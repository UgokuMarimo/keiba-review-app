import argparse
import traceback
import json
import lightgbm as lgb
import shap
from joblib import load
from dotenv import load_dotenv
import sys
import os
import pandas as pd
import numpy as np

# --- Windows環境でのUnicode出力エラー対策 (重要) ---
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))

# .envファイルの読み込み
load_dotenv()

# --- モジュールインポート ---
import config
from utils.feature_pipeline import (
    preprocess_and_clean, add_past_race_features, 
    engineer_advanced_features, add_race_level_features, encode_and_finalize
)
from utils.scraper import scrape_shutuba_table, load_past_race_data, load_past_race_data_with_overseas
from utils.db_utils import save_prediction_to_db, send_discord_webhook, format_for_discord

# 解説生成用モジュール
try:
    import google.generativeai as genai
    import chromadb
    from explanation_templates import get_original_value_display
    
    # APIキー設定
    if "GOOGLE_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        GENERATION_MODEL = "gemini-2.5-flash"
        ENABLE_EXPLANATION = True
    else:
        print("[WARN] GOOGLE_API_KEY not found. Automated explanation disabled.")
        ENABLE_EXPLANATION = False

except ImportError as e:
    print(f"[WARN] Failed to import explanation modules: {e}. Automated explanation disabled.")
    ENABLE_EXPLANATION = False

def load_vector_db():
    """ベクトルDBクライアントをロード"""
    vector_db_path = os.path.join(PROJECT_ROOT, "vector_db")
    if not os.path.exists(vector_db_path):
        return None
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        return client.get_collection(name="race_results")
    except Exception as e:
        print(f"[WARN] Failed to load vector DB: {e}")
        return None

def generate_explanation(horse_data, collection):
    """
    上位馬の解説を生成する関数 (app.pyのロジックを移植)
    """
    if not ENABLE_EXPLANATION:
        return None

    try:
        # RAG: ベクトルDB検索
        context_docs = ""
        if collection:
            search_query = f"{horse_data['horse_name']}の最近のレース内容"
            try:
                retrieved = collection.query(query_texts=[search_query], n_results=3)
                if retrieved['documents']:
                    context_docs = "\n".join(retrieved['documents'][0])
            except Exception as e:
                print(f"[WARN] Vector DB query failed: {e}")

        # 特徴量の整理
        race_level_factors = []
        normal_factors = []
        
        all_factors = horse_data['positive_factors'] + horse_data['negative_factors']
        all_factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        for f in all_factors:
            if "_race_" in f['feature']:
                race_level_factors.append(f)
            else:
                normal_factors.append(f)

        # プロンプト作成
        prompt = f"""あなたはデータ重視の冷静な競馬分析家です。
提供されたAI分析データ（SHAP値）と過去のレース情報を基に、競走馬「{horse_data['horse_name']}」の能力と今回のレースにおける期待度を論理的に解説してください。

# 分析データ
- **予測順位**: {horse_data['pred_rank']}位 (勝率: {horse_data['pred_win_prob']:.1%})

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
    - ユーザーに見えていない内部変数名（例: `past_5_race_dev`）をそのまま使わず、自然な日本語に翻訳して説明してください。
    - `_race_dev` は「レース平均に対する優位性（プラスなら平均以上）」を意味します。

2.  **構成**:
    - **結論**: 評価を一言で（例：「有力候補」「紐候補」「過剰人気」）。
    - **根拠**: 「重要な評価指標」の数値を具体的に引用しながら解説。
    - **相対比較**: 「他馬との比較」データを使い、メンバー内での立ち位置を説明。
    - **総評**: 馬券的な推奨度合い。

3.  **注意点**:
    - 文字数は300文字〜500文字程度。
"""
        # LLM実行
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"[WARN] Failed to generate explanation for {horse_data['horse_name']}: {e}")
        return None

def load_models(model_type: str):
    """
    モデルと関連アーティファクトを読み込む関数。
    バッチ処理などで再利用可能にするために分離。
    """
    print(f"[INFO] Loading models for type: {model_type}")
    model_conf = config.MODEL_CONFIGS[model_type]
    
    # トラックタイプごとにモデルをロード (芝/ダート)
    models = {}
    artifacts = {}
    
    for track_type in ['turf', 'dirt']:
        artifacts_base_dir = os.path.join(config.ARTIFACTS_DIR, f'{model_type}_artifacts')
        model_dir = os.path.join(config.MODEL_DIR_BASE, f'{model_type}_models', config.EXPERIMENT_VERSION)
        
        # アーティファクト読み込み
        try:
            time_scaler = load(os.path.join(artifacts_base_dir, f'time_scaler_{track_type}.joblib'))
            label_encoders = load(os.path.join(artifacts_base_dir, f'label_encoders_{track_type}.joblib'))
            
            # 統計情報の読み込み
            stats_to_load = model_conf['stats_to_save']
            loaded_stats = {}
            for key in stats_to_load:
                path = os.path.join(artifacts_base_dir, f'{key}_{track_type}.joblib')
                if os.path.exists(path):
                    loaded_stats[key] = load(path)
                else:
                    print(f"[WARN] Artifact not found: {path}")
            
            artifacts[track_type] = {
                'time_scaler': time_scaler,
                'label_encoders': label_encoders,
                'loaded_stats': loaded_stats
            }
            
            # モデル読み込み
            lgb_model = lgb.Booster(model_file=os.path.join(model_dir, f'lgb_model_{track_type}_win.txt'))
            imputer = load(os.path.join(model_dir, f'imputer_{track_type}_win.joblib'))
            
            models[track_type] = {
                'lgb_model': lgb_model,
                'imputer': imputer
            }
            
        except Exception as e:
            print(f"[WARN] Failed to load models/artifacts for {track_type}: {e}")
            # 該当するトラックタイプのレースが予測できないだけなので、続行
            
    return models, artifacts, model_conf

def predict_race(race_id: str, model_type: str, run_shap: bool, use_overseas: bool, enable_explanation: bool = True, models=None, artifacts=None, model_conf=None):
    """
    1レース分の予測を実行する関数。
    models, artifacts が渡されない場合は内部でロードする。
    """
    print(f"--- [START] Prediction for race_id: {race_id} (Model: {model_type}) ---")
    
    try:
        # モデルが渡されていない場合はロード
        if models is None or artifacts is None or model_conf is None:
            models, artifacts, model_conf = load_models(model_type)

        # --- [Phase 1/5] データ取得 ---
        shutuba_df = scrape_shutuba_table(race_id)
        if shutuba_df is None or shutuba_df.empty:
            print("[FATAL] Failed to scrape shutuba data. Exiting.")
            return None
        
        # --- [Phase 2/5] 過去走データ取得 ---
        horse_ids = shutuba_df['horse_id'].astype(str).unique().tolist()
        race_date = shutuba_df['日付'].iloc[0]
        
        if use_overseas:
            print("[INFO] Using overseas/local race data integration mode.")
            try:
                past_race_df = load_past_race_data_with_overseas(
                    horse_ids,
                    race_date=race_date,
                    num_past_races=config.NUM_PAST_RACES,
                    use_horse_page=True,
                    save_to_cache=True
                )
            except Exception as e:
                print(f"[WARN] Failed to load overseas data: {e}. Falling back to traditional method.")
                import traceback
                traceback.print_exc()
                past_race_df = load_past_race_data(horse_ids)
        else:
            past_race_df = load_past_race_data(horse_ids)
        
        track_surface = shutuba_df['芝・ダート'].iloc[0]
        if '障' in track_surface or '新馬' in shutuba_df['レース名'].iloc[0]:
            print(f"[INFO] Skipping prediction for steeplechase or debut race.")
            print("[SKIPPED] This race is a debut or steeplechase race and is not supported.")
            return None

        # --- [Phase 3/5] 特徴量生成 ---
        print("\n--- [Phase 3/5] Feature Generation ---")
        track_type = 'turf' if '芝' in track_surface else 'dirt'
        
        if track_type not in models:
            print(f"[FATAL] Model for {track_type} not loaded.")
            return None
            
        current_artifacts = artifacts[track_type]
        time_scaler = current_artifacts['time_scaler']
        label_encoders = current_artifacts['label_encoders']
        loaded_stats = current_artifacts['loaded_stats']

        # 特徴量パイプライン実行
        if past_race_df is not None and not past_race_df.empty:
            combined_df = pd.concat([shutuba_df, past_race_df], ignore_index=True, sort=False)
        else:
            combined_df = shutuba_df.copy()

        processed_df, _ = preprocess_and_clean(combined_df, time_scaler=time_scaler)
        df_with_past = add_past_race_features(processed_df, config.NUM_PAST_RACES, config.PAST_RACE_FEATURES)
        df_featured, _ = engineer_advanced_features(df_with_past, config.NUM_PAST_RACES, jockey_rates=loaded_stats)
        df_race_level = add_race_level_features(df_featured)
        
        predict_target_df = df_race_level[df_race_level['race_id'] == str(race_id)].copy()
        if predict_target_df.empty:
            print("[FATAL] No target race data to process after feature engineering.")
            return None
            
        categorical_features = model_conf['categorical_features']
        features_df, _ = encode_and_finalize(predict_target_df, categorical_features, label_encoders=label_encoders)
        features_df.reset_index(drop=True, inplace=True)

        # --- [Phase 4/5] 予測実行 ---
        print("\n--- [Phase 4/5] Prediction ---")
        
        current_model = models[track_type]
        lgb_model = current_model['lgb_model']
        imputer = current_model['imputer']
        model_columns = lgb_model.feature_name()

        X_predict = pd.DataFrame(columns=model_columns)
        for col in model_columns:
            if col in features_df.columns:
                X_predict[col] = features_df[col]
            else:
                X_predict[col] = np.nan

        leakage_cols = model_conf.get('leakage_features', [])
        cols_to_drop_for_pred = [col for col in X_predict.columns if col in leakage_cols]
        X_predict_cleaned = X_predict.drop(columns=cols_to_drop_for_pred, errors='ignore')

        X_predict_imputed = pd.DataFrame(imputer.transform(X_predict_cleaned), columns=X_predict_cleaned.columns)
        pred_win = lgb_model.predict(X_predict_imputed)

        # 結果を整形
        base_info_df = shutuba_df[['馬番', '馬']].copy().rename(columns={'馬': '馬名'})
        pred_df = pd.DataFrame({'pred_win': pred_win, '馬番': features_df['馬番'].values})
        
        base_info_df['馬番'] = pd.to_numeric(base_info_df['馬番'], errors='coerce')
        pred_df['馬番'] = pd.to_numeric(pred_df['馬番'], errors='coerce')
        
        final_result_df = pd.merge(base_info_df, pred_df, on='馬番', how='left')
        final_result_df['rank_win'] = final_result_df['pred_win'].rank(ascending=False, method='first').astype(int)
        final_result_df.sort_values('rank_win', inplace=True)

        # --- [Phase 5/5] SHAP 分析 & 結果保存 ---
        shap_output_dir = os.path.join(PROJECT_ROOT, 'shap_results', race_id)
        os.makedirs(shap_output_dir, exist_ok=True)
        
        # 全頭の結果サマリーを作成
        summary_data = []
        
        # ベクトルDBのロード (解説生成用)
        vector_db_collection = load_vector_db() if ENABLE_EXPLANATION else None
        
        if run_shap:
            print("\n--- [Phase 5/5] SHAP Analysis ---")
            try:
                explainer = shap.TreeExplainer(lgb_model)
                shap_values_list = explainer.shap_values(X_predict_imputed)
                if isinstance(shap_values_list, list):
                    shap_values = shap_values_list[1]
                else:
                    shap_values = shap_values_list

                # 全頭分のデータを処理
                for i in range(len(final_result_df)): 
                    horse_info = final_result_df.iloc[i]
                    horse_umaban = int(horse_info['馬番'])
                    
                    original_idx_list = features_df.index[features_df['馬番'] == horse_umaban].tolist()
                    if not original_idx_list: continue
                    original_idx = original_idx_list[0]

                    shap_df = pd.DataFrame({
                        'feature': X_predict_imputed.columns,
                        'shap_value': shap_values[original_idx],
                        'value': X_predict_imputed.iloc[original_idx].values
                    })
                    
                    shap_df['value'] = shap_df['value'].astype(float)
                    shap_df['shap_value'] = shap_df['shap_value'].astype(float)

                    positive_features_all = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False)
                    negative_features_all = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value', ascending=True)

                    # データ構築
                    horse_data = {
                        "race_id": race_id,
                        "horse_name": horse_info['馬名'],
                        "umaban": horse_umaban,
                        "pred_win_prob": float(horse_info['pred_win']),
                        "pred_rank": int(horse_info['rank_win']),
                        "positive_factors": positive_features_all.to_dict('records'),
                        "negative_factors": negative_features_all.to_dict('records'),
                        "explanation": None # 初期値
                    }
                    
                    # 上位3頭は解説を自動生成
                    if horse_info['rank_win'] <= 3 and ENABLE_EXPLANATION and enable_explanation:
                        print(f"  Generating explanation for Rank {horse_info['rank_win']}: {horse_info['馬名']}...")
                        explanation = generate_explanation(horse_data, vector_db_collection)
                        horse_data["explanation"] = explanation

                    summary_data.append(horse_data)

                    # 上位3頭は個別のJSONも保存（互換性維持のため）
                    if horse_info['rank_win'] <= 3:
                        save_path = os.path.join(shap_output_dir, f"shap_rank_{horse_info['rank_win']}.json")
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(horse_data, f, ensure_ascii=False, indent=4)
                        
                        # コンソール表示 (Top 3のみ)
                        print(f"\n🐴 予測{horse_info['rank_win']}位: {horse_umaban}番 {horse_info['馬名']} (予測値: {horse_info['pred_win']:.4f})")
                        print("  ✅ 好材料 TOP5")
                        for _, row in positive_features_all.head(5).iterrows():
                            print(f"    - {row['feature']:<30} (値: {row['value']:.2f}, 貢献度: {row['shap_value']:.4f})")
                        print("  ❌ 不安材料 TOP5")
                        for _, row in negative_features_all.head(5).iterrows():
                            print(f"    - {row['feature']:<30} (値: {row['value']:.2f}, 貢献度: {row['shap_value']:.4f})")

            except Exception as e:
                print(f"\n[SHAP ERROR] An error occurred: {e}")
                traceback.print_exc()
        else:
            # SHAPなしの場合でもサマリーは作成
             for i in range(len(final_result_df)):
                horse_info = final_result_df.iloc[i]
                summary_data.append({
                    "race_id": race_id,
                    "horse_name": horse_info['馬名'],
                    "umaban": int(horse_info['馬番']),
                    "pred_win_prob": float(horse_info['pred_win']),
                    "pred_rank": int(horse_info['rank_win']),
                    "positive_factors": [],
                    "negative_factors": [],
                    "explanation": None
                })

        # 全頭サマリーJSONの保存
        summary_path = os.path.join(shap_output_dir, "prediction_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
        print(f"-> Full prediction summary saved to: {summary_path}")

        # --- [完了] 通知とDB保存 ---
        discord_message = format_for_discord(race_id, shutuba_df.iloc[0], final_result_df)
        send_discord_webhook(discord_message)
        
        try:
            save_prediction_to_db(final_result_df, shutuba_df, race_id)
        except Exception as e:
            print(f"[DB ERROR] Failed to save to DB: {e}")

        print("\n--- [SUCCESS] All processes complete. ---")
        return final_result_df
        
    except Exception as e:
        print("\n--- [FATAL ERROR] An unexpected error occurred in main process ---")
        traceback.print_exc()

def main(race_id: str, model_type: str, run_shap: bool, use_overseas: bool = False, enable_explanation: bool = True):
    predict_race(race_id, model_type, run_shap, use_overseas, enable_explanation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and explain a horse race.")
    parser.add_argument('race_id', help='Target 12-digit race ID')
    parser.add_argument('model_type', help="Model type to use (e.g., 'B' or 'C')")
    parser.add_argument('--no-shap', action='store_false', dest='run_shap', help="Disable SHAP analysis.")
    parser.add_argument('--use-overseas', action='store_true', dest='use_overseas', help="Include overseas and local race data.")
    parser.add_argument('--no-explanation', action='store_false', dest='enable_explanation', help="Disable automated LLM explanation.")
    parser.set_defaults(run_shap=True, use_overseas=False, enable_explanation=True)
    
    args = parser.parse_args()
    main(args.race_id, args.model_type.upper(), args.run_shap, args.use_overseas, args.enable_explanation)