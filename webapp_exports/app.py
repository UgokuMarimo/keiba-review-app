import streamlit as st
import os
import sys
import json
import subprocess
import chromadb
import google.generativeai as genai
import sqlite3
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
import itertools
import re
# --- プロジェクトパス設定 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))

from utils import analytics  # Import the new analytics module

# .envファイルの読み込み
load_dotenv()

# --- モジュールインポート ---
# try-except block to handle potential import errors during initial setup
try:
    from explanation_templates import EXPLANATION_TEMPLATES, get_original_value_display
    from code.utils.schedule_scraper import get_race_schedule_for_date
except ImportError:
    st.error("必要なモジュールが見つかりません。パス設定を確認してください。")
    st.stop()

# --- APIキーとモデル設定 ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("環境変数 'GOOGLE_API_KEY' が設定されていません。")
    st.stop()

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"

# --- 定数 ---
DB_PATH = os.path.join(PROJECT_ROOT, 'predictions.db')
BET_AMOUNT = 100

# --- Streamlit UI設定 ---
st.set_page_config(page_title="競馬AI 統合プラットフォーム", layout="wide")
st.title("🐎 競馬AI 統合プラットフォーム")

# --- サイドバー ---
st.sidebar.header("メニュー")
page = st.sidebar.radio("モード選択", ["レース予測 & 解説", "回収率分析", "📊 予測精度分析"])

# --- データベース接続 (ベクトルDB) ---
@st.cache_resource
def load_vector_db():
    vector_db_path = os.path.join(PROJECT_ROOT, "vector_db")
    if not os.path.exists(vector_db_path):
        return None
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        return client.get_collection(name="race_results")
    except Exception as e:
        st.error(f"ベクトルデータベースの読み込みに失敗しました: {e}")
        return None

collection = load_vector_db()

# --- 関数定義: 回収率分析 (Old) ---
def analyze_recovery_rate():
    st.header("回収率分析")

    # --- 結果更新ボタン ---
    if st.button("レース結果を更新する (未確定レースの取得)"):
        with st.spinner("レース結果を取得・更新中... (数分かかる場合があります)"):
            script_path = os.path.join(PROJECT_ROOT, 'code', 'a4_prediction', 'm05_update_results.py')
            result = subprocess.run(
                [sys.executable, script_path], 
                capture_output=True, 
                text=True, 
                encoding='utf-8'
            )
            if result.returncode == 0:
                st.success("レース結果の更新が完了しました！")
                with st.expander("更新ログ詳細"):
                    st.text(result.stdout)
            else:
                st.error("更新中にエラーが発生しました。")
                st.text(result.stderr)

    if not os.path.exists(DB_PATH):
        st.error("データベースファイルが見つかりません。")
        return

    # --- データベース接続 ---
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 必要なデータを結合して取得
        query = """
        SELECT
            p.race_id,
            p.race_number,
            p.umaban,
            p.pred_rank,
            p.result_rank,
            pay.tansho_payout, pay.tansho_numbers,
            pay.fukusho_payouts,
            pay.umaren_payout, pay.umaren_numbers,
            pay.wide_payouts,
            pay.umatan_payout, pay.umatan_numbers,
            pay.sanrenpuku_payout, pay.sanrenpuku_numbers,
            pay.sanrentan_payout, pay.sanrentan_numbers
        FROM
            predictions p
        LEFT JOIN payouts pay ON p.race_id = pay.race_id
        WHERE
            p.result_rank IS NOT NULL
        ORDER BY
            p.race_id, p.pred_rank;
        """
        try:
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            return

    if df.empty:
        st.info("分析対象のデータがありません。予測を実行し、結果を更新してください。")
        return

    # --- サイドバー設定 (分析条件) ---
    st.sidebar.subheader("📊 分析条件設定")
    
    # 1. レースフィルター
    st.sidebar.markdown("### 1. レースフィルター")
    race_num_range = st.sidebar.slider("レース番号範囲", 1, 12, (1, 12))
    
    # 2. 賭け式設定
    st.sidebar.markdown("### 2. シミュレーション設定")
    bet_type = st.sidebar.selectbox("券種", ["単勝", "複勝", "馬連", "ワイド", "馬単", "3連複", "3連単"])
    strategy = st.sidebar.selectbox("買い方", ["ボックス", "流し(1頭軸) - 未実装", "フォーメーション - 未実装"], index=0)
    
    if strategy == "ボックス":
        top_n = st.sidebar.number_input("予測上位何頭を買うか (Box)", min_value=1, max_value=18, value=5)
    else:
        top_n = 1 # Placeholder
        st.sidebar.warning("現在ボックス買いのみサポートしています。")

    bet_amount = st.sidebar.number_input("1点あたりの投資額 (円)", min_value=100, step=100, value=100)

    # --- 分析ロジック ---
    
    # 1. フィルター適用
    filtered_df = df[
        (df['race_number'] >= race_num_range[0]) & 
        (df['race_number'] <= race_num_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("条件に一致するレースがありません。")
        return

    # レースごとに処理
    race_ids = filtered_df['race_id'].unique()
    
    total_investment = 0
    total_return = 0
    hit_count = 0
    race_results_list = []

    for race_id in race_ids:
        race_data = filtered_df[filtered_df['race_id'] == race_id]
        
        # 予測上位N頭を取得
        top_horses = race_data.nsmallest(top_n, 'pred_rank')
        selected_umabans = top_horses['umaban'].tolist()
        
        # 買い目生成
        combinations = []
        if bet_type == "単勝":
            combinations = [(u,) for u in selected_umabans]
        elif bet_type == "複勝":
            combinations = [(u,) for u in selected_umabans]
        elif bet_type == "馬連":
            combinations = list(itertools.combinations(selected_umabans, 2))
        elif bet_type == "ワイド":
            combinations = list(itertools.combinations(selected_umabans, 2))
        elif bet_type == "馬単":
            combinations = list(itertools.permutations(selected_umabans, 2))
        elif bet_type == "3連複":
            combinations = list(itertools.combinations(selected_umabans, 3))
        elif bet_type == "3連単":
            combinations = list(itertools.permutations(selected_umabans, 3))

        # 投資額加算
        investment = len(combinations) * bet_amount
        total_investment += investment

        # 払い戻し判定
        payout = 0
        hit_flag = False
        
        # 1行目のデータから払い戻し情報を取得 (同じレースなら同じはず)
        row = race_data.iloc[0]
        
        # 払い戻しデータがない場合はスキップ (未確定など)
        if pd.isna(row['tansho_numbers']):
            continue

        # 的中判定ロジック
        def check_hit(bet_combo, result_numbers_str, payout_val):
            if not result_numbers_str: return 0
            bet_set = set(map(str, bet_combo))
            res_nums = re.findall(r'\d+', str(result_numbers_str))
            res_set = set(res_nums)
            
            if bet_type in ["馬単", "3連単"]:
                if tuple(map(str, bet_combo)) == tuple(res_nums):
                    return payout_val
            else:
                if bet_set == res_set:
                    return payout_val
            return 0

        # 券種ごとの判定
        race_payout = 0
        
        if bet_type == "単勝":
            if str(row['tansho_numbers']) in [str(c[0]) for c in combinations]:
                race_payout += row['tansho_payout']

        elif bet_type == "複勝":
            try:
                fuku_dict = json.loads(row['fukusho_payouts'])
                for bet in combinations:
                    u = str(bet[0])
                    if u in fuku_dict:
                        race_payout += fuku_dict[u]
            except: pass

        elif bet_type == "馬連":
            for bet in combinations:
                race_payout += check_hit(bet, row['umaren_numbers'], row['umaren_payout'])

        elif bet_type == "ワイド":
            try:
                wide_dict = json.loads(row['wide_payouts'])
                for bet in combinations:
                    bet_set = set(map(str, bet))
                    for key, pay in wide_dict.items():
                        key_set = set(re.findall(r'\d+', key))
                        if bet_set == key_set:
                            race_payout += pay
            except: pass

        elif bet_type == "馬単":
            for bet in combinations:
                race_payout += check_hit(bet, row['umatan_numbers'], row['umatan_payout'])

        elif bet_type == "3連複":
            for bet in combinations:
                race_payout += check_hit(bet, row['sanrenpuku_numbers'], row['sanrenpuku_payout'])

        elif bet_type == "3連単":
            for bet in combinations:
                race_payout += check_hit(bet, row['sanrentan_numbers'], row['sanrentan_payout'])

        if race_payout > 0:
            hit_count += 1
            hit_flag = True
        
        total_return += race_payout
        
        race_results_list.append({
            "race_id": race_id,
            "race_num": row['race_number'],
            "investment": investment,
            "return": race_payout,
            "hit": "🎯" if hit_flag else "-"
        })

    # --- 結果表示 ---
    total_races = len(race_ids)
    hit_rate = (hit_count / total_races) * 100 if total_races > 0 else 0
    recovery_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

    st.markdown("---")
    st.subheader(f"分析結果 ({bet_type} / {strategy} / Top {top_n}頭)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("対象レース数", f"{total_races} レース")
    col2.metric("的中率", f"{hit_rate:.1f}%")
    col3.metric("回収率", f"{recovery_rate:.1f}%", delta=f"{recovery_rate - 100:.1f}%")
    col4.metric("払戻", f"{total_return - total_investment:,} 円")
    st.write(f"総投資: {total_investment:,} 円 / 総払戻: {total_return:,} 円")

    # 詳細テーブル
    with st.expander("レース別詳細を見る"):
        st.dataframe(pd.DataFrame(race_results_list))


# --- メインロジック分岐 ---

if page == "レース予測 & 解説":
    # 1. 日付選択
    col_date, col_btn = st.columns([2, 1])
    with col_date:
        target_date = st.date_input("日付を選択", date.today())
    with col_btn:
        # 改行でボタン位置を調整
        st.write("") 
        st.write("")
        fetch_btn = st.button("レース情報を取得", type="primary")

    # 2. レーススケジュール取得 (ボタン押下時のみ)
    if fetch_btn:
        with st.spinner(f"{target_date} のレース情報を取得中..."):
            schedule_df = get_race_schedule_for_date(target_date.strftime("%Y-%m-%d"))
            st.session_state['schedule_df'] = schedule_df
            st.session_state['target_date'] = target_date
            # 日付が変わったら選択状態をリセット
            st.session_state['selected_race_id'] = None 
    
    # 以前取得したデータがあり、かつ日付が同じなら表示
    schedule_df = st.session_state.get('schedule_df')
    current_target_date = st.session_state.get('target_date')

    if schedule_df is not None and not schedule_df.empty and current_target_date == target_date:
        
        # --- レース選択画面 (Grid Layout) ---
        if st.session_state.get('selected_race_id') is None:
            st.subheader(f"{target_date} の開催レース")

            # --- 一括予測 (Batch Prediction) ---
            with st.expander("🚀 一括予測 (Batch Prediction)", expanded=False):
                st.write("複数のレースをまとめて予測・解説生成します。")
                all_races_label = schedule_df.apply(lambda x: f"{x['venue_name']} {x['race_number']}R ({x['race_name'] or '名無し'})", axis=1).tolist()
                race_id_map = {f"{x['venue_name']} {x['race_number']}R ({x['race_name'] or '名無し'})": x['race_id'] for _, x in schedule_df.iterrows()}
                
                # 全選択ボタン
                if st.button("全レースを選択"):
                    st.session_state['batch_selected_races'] = all_races_label
                    # multiselectのkeyに対応するsession_stateも更新する必要がある
                    st.session_state['batch_race_selector'] = all_races_label
                
                # 既存の選択状態を、現在の選択肢に含まれるものだけにフィルタリング (Bug Fix)
                current_selection = st.session_state.get('batch_selected_races', [])
                valid_selection = [x for x in current_selection if x in all_races_label]
                
                selected_labels = st.multiselect(
                    "予測するレースを選択", 
                    all_races_label, 
                    default=valid_selection,
                    key="batch_race_selector"
                )
                
                # 解説生成オプション
                enable_explanation = st.checkbox("解説も同時に生成する (時間がかかります)", value=False)
                
                if st.button("選択したレースを予測する", type="primary", key="batch_predict_btn"):
                    if not selected_labels:
                        st.warning("レースを選択してください。")
                    else:
                        selected_ids = [race_id_map[label] for label in selected_labels]
                        st.info(f"{len(selected_ids)} レースの予測を開始します...")
                        
                        # プログレスバー
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # バッチスクリプト実行
                        script_path = os.path.join(PROJECT_ROOT, 'code', 'a4_prediction', 'm06_batch_predict.py')
                        # IDをカンマ区切りで渡す
                        ids_str = ",".join(selected_ids)
                        
                        cmd = [sys.executable, script_path, '--race_ids', ids_str, '--model_type', 'B']
                        if enable_explanation:
                            cmd.append('--explanation')
                        
                        with st.spinner("AIが予測を実行中..."):
                            try:
                                # Windows環境でのエンコーディング問題を回避するために環境変数を設定
                                env = os.environ.copy()
                                env["PYTHONIOENCODING"] = "utf-8"
                                
                                result = subprocess.run(
                                    cmd,
                                    capture_output=True,
                                    text=True,
                                    encoding='utf-8',
                                    env=env
                                )
                                
                                if result.returncode == 0:
                                    st.success("✅ 一括予測が完了しました！")
                                    with st.expander("実行ログ"):
                                        st.text(result.stdout)
                                        
                                    # スキップされたレースの警告
                                    if "[SKIPPED]" in result.stdout:
                                        st.warning("一部のレースは予測対象外（新馬戦・障害戦）のためスキップされました。ログを確認してください。")
                                else:
                                    st.error("❌ 予測中にエラーが発生しました。")
                                    st.text(result.stderr)
                            except Exception as e:
                                st.error(f"実行エラー: {e}")
                                
            st.markdown("---")
            st.markdown("---")
            st.subheader("個別レース選択")
            
            # 開催地ごとにグルーピング
            venues = schedule_df['venue_name'].unique()
            cols = st.columns(len(venues))
            
            for i, venue in enumerate(venues):
                with cols[i]:
                    st.markdown(f"### {venue}")
                    venue_races = schedule_df[schedule_df['venue_name'] == venue]
                    
                    for _, row in venue_races.iterrows():
                        race_name = row['race_name'] if row['race_name'] else "レース名なし"
                        
                        # 予測済みかチェック
                        race_id = str(row['race_id'])
                        shap_dir = os.path.join(PROJECT_ROOT, 'shap_results', race_id)
                        summary_path = os.path.join(shap_dir, "prediction_summary.json")
                        is_predicted = os.path.exists(summary_path)
                        
                        status_mark = "✅ " if is_predicted else ""
                        label = f"{status_mark}{row['race_number']}R {row['start_time']} {race_name}"
                        
                        # ボタンでレース選択
                        if st.button(label, key=f"btn_{row['race_id']}", use_container_width=True):
                            st.session_state['selected_race_id'] = row['race_id']
                            st.rerun()

        # --- レース詳細・予測画面 ---
        else:
            race_id = st.session_state['selected_race_id']
            race_row = schedule_df[schedule_df['race_id'] == race_id].iloc[0]
            
            # ヘッダーと戻るボタン
            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("← 一覧に戻る"):
                    st.session_state['selected_race_id'] = None
                    st.rerun()
            with col_title:
                st.subheader(f"{race_row['venue_name']} {race_row['race_number']}R {race_row['race_name']}")
                st.caption(f"発走: {race_row['start_time']} / ID: {race_id}")

            # 4. 予測実行ボタン
            if st.button("予測を実行する", type="primary"):
                with st.spinner("AIが予測を実行し、分析しています... (最大1分程度)"):
                    script_path = os.path.join(PROJECT_ROOT, 'code', 'a4_prediction', 'm04_predict.py')
                    # subprocessで実行
                    result = subprocess.run(
                        [sys.executable, script_path, race_id, 'B'], 
                        capture_output=True, 
                        text=True, 
                        encoding='utf-8'
                    )
                    
                    # デバッグ用: 実行ログを表示
                    with st.expander("実行ログを確認する", expanded=False):
                        st.text("STDOUT:")
                        st.text(result.stdout)
                        st.text("STDERR:")
                        st.text(result.stderr)

                    if result.returncode != 0:
                        st.error("予測実行中にエラーが発生しました。")
                    else:
                        st.success("予測とSHAP分析が完了しました！")

            # 5. 結果表示 & 解説選択
            shap_dir = os.path.join(PROJECT_ROOT, 'shap_results', race_id)
            if os.path.exists(shap_dir):
                horses_data = []
                summary_path = os.path.join(shap_dir, "prediction_summary.json")
                
                if os.path.exists(summary_path):
                    # 新しい形式: 全頭サマリーJSONからロード
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        horses_data = json.load(f)
                else:
                    # 旧形式: 個別のSHAP JSONからロード (互換性維持)
                    files = [f for f in os.listdir(shap_dir) if f.startswith("shap_rank_") and f.endswith(".json")]
                    for f in files:
                        with open(os.path.join(shap_dir, f), 'r', encoding='utf-8') as json_file:
                            data = json.load(json_file)
                            horses_data.append(data)
                
                # 予測順位でソート
                horses_data.sort(key=lambda x: x['pred_rank'])
                
                # 結果テーブル表示
                st.subheader("予測結果一覧")
                result_df = pd.DataFrame([{
                    "順位": h['pred_rank'],
                    "馬番": h['umaban'],
                    "馬名": h['horse_name'],
                    "勝率予測": f"{h['pred_win_prob']:.1%}",
                    "解説": "✅" if h.get("explanation") else "-"
                } for h in horses_data])
                st.table(result_df)

                # 解説対象の選択
                st.subheader("詳細解説")
                horse_options = {f"{h['pred_rank']}位 {h['horse_name']}": h for h in horses_data}
                selected_horse_label = st.selectbox("解説を見たい馬を選択してください", list(horse_options.keys()))
                
                target_horse_data = horse_options[selected_horse_label]
                
                # 既に解説がある場合は表示
                if target_horse_data.get("explanation"):
                    st.markdown(f"### {target_horse_data['horse_name']} の解説")
                    st.info("自動生成された解説を表示します。")
                    st.markdown(target_horse_data["explanation"])
                else:
                    if st.button("解説を生成"):
                        target_horse_data = horse_options[selected_horse_label]
                        
                        with st.spinner(f"{target_horse_data['horse_name']} の解説を生成中..."):
                            # RAG: ベクトルDB検索
                            context_docs = ""
                            if collection:
                                search_query = f"{target_horse_data['horse_name']}の最近のレース内容"
                                retrieved = collection.query(query_texts=[search_query], n_results=3)
                                if retrieved['documents']:
                                    context_docs = "\n".join(retrieved['documents'][0])

                            # 特徴量の整理 (相対評価用)
                            # _race_ を含む特徴量を抽出
                            race_level_factors = []
                            normal_factors = []
                            
                            all_factors = target_horse_data['positive_factors'] + target_horse_data['negative_factors']
                            # 貢献度絶対値でソート
                            all_factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)

                            for f in all_factors:
                                if "_race_" in f['feature']:
                                    race_level_factors.append(f)
                                else:
                                    normal_factors.append(f)

                            # プロンプト作成
                            prompt = f"""あなたはデータ重視の冷静な競馬分析家です。
提供されたAI分析データ（SHAP値）と過去のレース情報を基に、競走馬「{target_horse_data['horse_name']}」の能力と今回のレースにおける期待度を論理的に解説してください。

# 分析データ
- **予測順位**: {target_horse_data['pred_rank']}位 (勝率: {target_horse_data['pred_win_prob']:.1%})

## 1. 重要な評価指標 (Key Factors)
{chr(10).join([f"- {f['feature']} (値: {get_original_value_display(f['feature'], f['value'])}, 貢献度: {f['shap_value']:.3f})" for f in normal_factors[:5]])}

## 2. 他馬との比較 (Relative Context)
レースメンバー平均との乖離など、相対的な立ち位置づけ！
{chr(10).join([f"- {f['feature']} (値: {get_original_value_display(f['feature'], f['value'])}, 貢献度: {f['shap_value']:.3f})" for f in race_level_factors[:3]])}

## 3. 過去の実績 (Background)
{context_docs}

# 解説のガイドライン
1.  **トーン＆マナー**:
    - 「明るげな本命」のような過度な表現は避け、データに基づいた客観的・専門的な口調で記述してください。
    - ユーザーに見えていない内部変数名（例: `past_5_race_dev`）をそのまま使わず、自然な日本語に翻訳して説明してください（例: 「過去5走のレースレベルとの乖離」）。
    - `_race_dev` は「レース平均に対する優位性（プラスなら平均以上）」を意味します。
    - 騎手や調教師のデータに言及する際は、必ず「提示された数値（勝率など）」を根拠に挙げてください。一般的なイメージではなく、今回のデータセットの数値を優先してください。

2.  **構成**:
    - **結論**: 評価を一言で（例：「有力候補」「紐候補」「過剰人気」）。
    - **根拠**: 「重要な評価指標」の数値を具体的に引用しながら解説。「値が〜〜であるため、〜〜と評価できる」という形式を推奨。
    - **相対比較**: 「他馬との比較」データを使い、メンバー内での立ち位置（スピード、安定感など）を説明。
    - **総評**: 馬券的な推奨度合い。

3.  **注意点**:
    - クラスやグレードの数値（例: Class 1:7）の意味が不明確な場合は、勝手に「G1」などと断定せず、「クラス指数が高く」のように数値そのものを使って説明してください。
    - 文字数は300文字〜500文字程度で、情報を充実させてください。
"""
                            # LLM実行
                            model = genai.GenerativeModel(GENERATION_MODEL)
                            response = model.generate_content(prompt)
                            
                            st.markdown(response.text)
        
    elif schedule_df is None:
        st.info("「レース情報を取得」ボタンを押して、レース一覧を表示してください。")
    else:
        st.warning("指定された日付のレース情報が見つかりませんでした。")

elif page == "回収率分析":
    analyze_recovery_rate()

elif page == "📊 予測精度分析":
    analytics.render_analysis_page(DB_PATH)
