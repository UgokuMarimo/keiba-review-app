import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import os

# --- プロジェクトパス設定 (feature_pipeline.py が utils フォルダ内にある想定) ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)

import config # config.py をインポート

# --- ヘルパー関数群 ---
def safe_float_convert(series: pd.Series) -> pd.Series:
    """安全に数値型に変換し、変換不能な値はNaNにする"""
    return pd.to_numeric(series, errors='coerce')

def class_mapping(row: pd.Series) -> int:
    """
    データフレームの行を受け取り、レースの格付けを数値に変換する。
    「レース名」列に含まれる多様な表記を正規表現で解析し、正確なクラス分類を行う。
    (GII, GIII のローマ数字表記に対応するよう修正済み)
    """
    # 各列から情報を安全に取得
    race_name = str(row.get('レース名', ''))
    class_info = str(row.get('クラス', '')) # 補助情報としてクラス列も利用
    track_type = str(row.get('芝・ダート', ''))

    # 1. 障害レースは学習対象外なので、明確に区別できる値を返す
    if '障' in track_type:
        return -99

    # 2. レース名とクラス情報を結合して判定材料を一つにする
    combined_info = race_name + " " + class_info

    # 3. 判定ロジック (より格上のレースから順番に判定)
    
    # [A] 重賞レースの判定
    #    正規表現を修正し、"GII", "GIII" のようなローマ数字表記に完全対応
    if re.search(r'[（\(]G(I|Ⅰ|1)[）\)]', combined_info): return 8  # G1
    if re.search(r'[（\(]G(II|Ⅱ|2)[）\)]', combined_info): return 7  # G2
    if re.search(r'[（\(]G(III|Ⅲ|3)[）\)]', combined_info): return 6  # G3
    if re.search(r'[（\(]重賞[）\)]', combined_info): return 6 # 「(重賞)」という表記もG3相当として扱う

    # [B] オープンクラス / リステッド の判定
    if 'オープン' in combined_info or '(OP)' in combined_info or re.search(r'[（\(]L[）\)]', combined_info):
        return 5

    # [C] 条件戦の判定 (3勝クラス -> 2勝クラス -> 1勝クラスの順に)
    if re.search(r'3勝|３勝|1600万', combined_info): return 4
    if re.search(r'2勝|２勝|1000万|900万', combined_info): return 3
    if re.search(r'1勝|１勝|500万', combined_info): return 2

    # [D] 新馬・未勝利・未出走の判定
    if '新馬' in combined_info or '未出走' in combined_info: return 0
    if '未勝利' in combined_info: return 1

    # [E] 上記のいずれにも該当しない特別レース (例: 「UHB賞」など)
    #     明示的なクラス表記がない特別レースは、オープンクラスとして扱う
    return 5

def process_passing_order(series: pd.Series) -> pd.DataFrame:
    def parse_single_order(order_str):
        if pd.isna(order_str) or not isinstance(order_str, str): return [np.nan, np.nan]
        try:
            positions = [int(p) for p in order_str.split('-')]
            if not positions: return [np.nan, np.nan]
            return [np.mean(positions), positions[-1] - positions[0]]
        except (ValueError, IndexError): return [np.nan, np.nan]
    parsed_data = series.apply(parse_single_order)
    return pd.DataFrame(parsed_data.tolist(), index=series.index, columns=['通過順_平均', '通過順_変動'])

def classify_running_style(passing_order_str: str, num_horses: int) -> str:
    if pd.isna(passing_order_str) or not isinstance(passing_order_str, str) or num_horses == 0: return 'unknown'
    try:
        positions = [int(p) for p in passing_order_str.split('-')]
        if not positions: return 'unknown'
        first_pos_normalized = positions[0] / num_horses
        if first_pos_normalized <= 0.25: return '逃げ'
        elif first_pos_normalized <= 0.5: return '先行'
        elif first_pos_normalized <= 0.75: return '差し'
        else: return '追込'
    except (ValueError, IndexError): return 'unknown'

def create_dynamic_gate_group(df: pd.DataFrame) -> pd.Series:
    if df.empty or not all(col in df.columns for col in ['race_id', '馬番', '出走頭数']):
        return pd.Series(['不明'] * len(df), index=df.index, name='馬番グループ')
    all_gate_groups = pd.Series(index=df.index, dtype=str, name='馬番グループ')
    for race_id, group_df in df.groupby('race_id'):
        num_horses = group_df['出走頭数'].iloc[0]
        if pd.isna(num_horses) or num_horses < 3:
            labels = pd.Series(['少頭数'] * len(group_df), index=group_df.index)
        else:
            third = num_horses / 3
            q1 = int(np.ceil(third)); q2 = int(np.ceil(2 * third))
            conditions = [group_df['馬番'] <= q1, (group_df['馬番'] > q1) & (group_df['馬番'] <= q2), group_df['馬番'] > q2]
            choices = ['内', '中', '外']
            labels = pd.Series(np.select(conditions, choices, default='不明'), index=group_df.index)
        all_gate_groups.loc[group_df.index] = labels
    return all_gate_groups.fillna('不明')

# --- パイプライン関数群 ---

def preprocess_and_clean(df: pd.DataFrame, time_scaler: dict = None) -> Tuple[pd.DataFrame, dict]:
    """
    データの前処理とクレンジングを行う。
    走破時間をMM:SS.S形式から秒数に変換し、さらに標準化して '走破時間_scaled' を生成する。
    """
    if df.empty: return pd.DataFrame(), time_scaler
    df_copy = df.copy()

    # --- 1. データ形式の統一 ---
    if '性齢' in df_copy.columns:
        df_copy['性'] = df_copy['性齢'].str[0]
        df_copy['齢'] = pd.to_numeric(df_copy['性齢'].str[1:], errors='coerce')
        df_copy.drop('性齢', axis=1, inplace=True, errors='ignore')

    if '日付' in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy['日付']):
            df_copy['日付'] = pd.to_datetime(df_copy['日付'], format='%Y年%m月%d日', errors='coerce')
        if df_copy['日付'].isnull().all():
            print("[ERROR] All date values are null. Aborting.")
            return pd.DataFrame(), time_scaler
    else:
        print("[ERROR] '日付' column not found. Aborting.")
        return pd.DataFrame(), time_scaler

    # --- 2. 既存の前処理 ---
    numeric_cols = ['馬番', '着順', '体重', '体重変化', '齢', '斤量', '上がり', '人気', '距離', 'オッズ']
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # --- 走破時間の処理: MM:SS.S -> 秒, 標準化, 外れ値処理, 新規特徴量追加 ---
    if '走破時間' in df_copy.columns:
        # まず、走破時間を「秒」に変換して新しいカラム '走破時間_seconds' を作成
        df_copy['走破時間_seconds'] = np.nan
        
        # '走破時間' がNaNでない、かつ ':' を含むもののみ処理
        mask_valid_time_str = df_copy['走破時間'].notna() & df_copy['走破時間'].astype(str).str.contains(':')
        
        if mask_valid_time_str.any():
            # 有効な文字列を持つ行のみ抽出して処理
            df_with_time_str = df_copy.loc[mask_valid_time_str, '走破時間'].astype(str)
            time_parts = df_with_time_str.str.split(':', expand=True)
            
            # 分を秒に変換し、秒と合算
            minutes = pd.to_numeric(time_parts[0], errors='coerce')
            seconds_part = pd.to_numeric(time_parts[1], errors='coerce')
            
            total_seconds = minutes * 60 + seconds_part
            
            # '走破時間_seconds' に結果を代入
            df_copy.loc[mask_valid_time_str, '走破時間_seconds'] = total_seconds

        # ここから、'走破時間_seconds' を使って '走破時間_scaled' を作成
        # 走破時間は小さいほど良いので、負の値にしてスケーリングすることで「大きいほど良い」特徴量にする
        # NaNを除外してスケーリングを計算し、後でNaNを戻す
        valid_times_for_scaling = df_copy['走破時間_seconds'].dropna()
        
        if not valid_times_for_scaling.empty:
            inverted_times = -valid_times_for_scaling # 速いタイムほど大きな値になる
            
            # time_scalerがNoneの場合（学習時）はパラメータをフィット
            if time_scaler is None:
                mean_val = inverted_times.mean()
                std_val = inverted_times.std()
                # 標準偏差が0の場合は1として扱う（ゼロ除算対策、スケーリング効果なし）
                if std_val == 0:
                    std_val = 1.0 
                
                # 外れ値処理のクリッピング範囲を設定
                clip_lower_bound = -3.0 
                clip_upper_bound = 3.0 

                time_scaler = {
                    'running_time_scaled_mean': mean_val,
                    'running_time_scaled_std': std_val,
                    'running_time_scaled_clip_lower': clip_lower_bound,
                    'running_time_scaled_clip_upper': clip_upper_bound
                }
            
            # スケーリング適用
            # time_scaler が存在しない場合（初回の呼び出しでNoneの場合）に備える
            # stdが0の場合はスケーリングせず0にする（全ての値が平均と同じと見なす）
            if time_scaler and time_scaler['running_time_scaled_std'] != 0:
                 scaled_inverted_times = (inverted_times - time_scaler['running_time_scaled_mean']) / time_scaler['running_time_scaled_std']
            else: 
                 scaled_inverted_times = inverted_times * 0 # 全て0にする
            
            # 外れ値処理 (クリッピング)
            clipped_scaled_times = scaled_inverted_times.clip(
                lower=time_scaler['running_time_scaled_clip_lower'],
                upper=time_scaler['running_time_scaled_clip_upper']
            )
            
            # '走破時間_scaled' カラムに結果を代入
            df_copy.loc[clipped_scaled_times.index, '走破時間_scaled'] = clipped_scaled_times
        
        # 元の '走破時間' (文字列) カラムは、'走破時間_seconds'と'走破時間_scaled'ができたため、削除
        df_copy.drop('走破時間', axis=1, inplace=True, errors='ignore')
        
    CATEGORY_MAPPINGS = {'性': {'牡': 0, '牝': 1, 'セ': 2}, '芝・ダート': {'芝': 0, 'ダ': 1, '障': 2}, '回り': {'右': 0, '左': 1, '芝': 2, '直': 2}}
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in df_copy.columns: df_copy[col] = df_copy[col].astype(str).str.strip().map(mapping)
    if '天気' in df_copy.columns:
        tenki_map = {'晴': 0, '曇': 1, '小': 2, '雨': 3, '雪': 4}; df_copy['天気'] = df_copy['天気'].str.strip().str[0].map(tenki_map)
    if '馬場' in df_copy.columns:
        baba_map = {'良': 0, '稍': 1, '重': 2, '不': 3}; df_copy['馬場'] = df_copy['馬場'].str.strip().str[0].map(baba_map)

    if '通過順' in df_copy.columns: df_copy = pd.concat([df_copy, process_passing_order(df_copy['通過順'].astype(str))], axis=1)
    
    if 'レース名' in df_copy.columns:
        df_copy['クラス'] = df_copy.apply(class_mapping, axis=1)

    df_copy['year'] = df_copy['日付'].dt.year
    df_copy['出走頭数'] = df_copy.groupby('race_id')['馬'].transform('count')
    df_copy['is_niigata_1000m'] = ((df_copy['場名'] == '新潟') & (df_copy['距離'] == 1000) & (df_copy['芝・ダート'] == 0)).astype(int)
    
    if '通過順' in df_copy.columns and '出走頭数' in df_copy.columns:
        df_copy['脚質'] = df_copy.apply(lambda r: classify_running_style(r['通過順'], r['出走頭数']), axis=1)
    else:
        df_copy['脚質'] = 'unknown' 
    df_copy['馬番グループ'] = create_dynamic_gate_group(df_copy) 
    
    return df_copy, time_scaler

def add_past_race_features(df: pd.DataFrame, num_past_races: int, past_race_features: List[str]) -> pd.DataFrame:
    """
    指定された過去走特徴量を、馬と日付に基づいてシフトして追加する。
    """
    if df.empty: return pd.DataFrame(columns=df.columns)
    if '馬' not in df.columns or '日付' not in df.columns: return df 
    df_copy = df.copy() 
    df_copy.sort_values(by=['馬', '日付'], ascending=[True, False], inplace=True) 
    for i in range(1, num_past_races + 1):
        df_copy[f'日付{i}'] = df_copy.groupby('馬')['日付'].shift(-i)
        for feature in past_race_features:
            if feature == '通過順': 
                if f'{feature}_平均' in df_copy.columns: df_copy[f'{feature}_平均{i}'] = df_copy.groupby('馬')[f'{feature}_平均'].shift(-i)
                if f'{feature}_変動' in df_copy.columns: df_copy[f'{feature}_変動{i}'] = df_copy.groupby('馬')[f'{feature}_変動'].shift(-i)
            elif feature in df_copy.columns: df_copy[f'{feature}{i}'] = df_copy.groupby('馬')[feature].shift(-i)
            else: df_copy[f'{feature}{i}'] = np.nan
    return df_copy

def engineer_advanced_features(df: pd.DataFrame, num_past_races: int, jockey_rates: Dict[str, pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: # time_scaler 引数を削除
    """
    高度な特徴量エンジニアリング（日付差、距離差、過去走統計量、ベイジアン勝率）を行う。
    走破時間スケーリングは preprocess_and_clean で既に実施済み。
    """
    if df.empty: return pd.DataFrame(columns=df.columns), {}
    df_copy = df.copy() 
    df_copy.replace('---', np.nan, inplace=True)
    
    # 日付差・長期休養明けフラグ
    if '日付' in df_copy.columns and '日付1' in df_copy.columns: df_copy['日付差1'] = (df_copy['日付'] - df_copy['日付1']).dt.days
    else: df_copy['日付差1'] = np.nan
    df_copy['長期休養明けフラグ'] = (df_copy['日付差1'] > 180).astype(int)

    # 距離差
    if '距離' in df_copy.columns: df_copy['距離'] = safe_float_convert(df_copy['距離'])
    if '距離1' in df_copy.columns: df_copy['距離1'] = safe_float_convert(df_copy['距離1'])
    if '距離' in df_copy.columns and '距離1' in df_copy.columns: df_copy['距離差1'] = df_copy['距離'] - df_copy['距離1']
    else: df_copy['距離差1'] = np.nan

    for i in range(2, num_past_races + 1):
        if f'日付{i-1}' in df_copy.columns and f'日付{i}' in df_copy.columns: df_copy[f'日付差{i}'] = (df_copy[f'日付{i-1}'] - df_copy[f'日付{i}']).dt.days
        else: df_copy[f'日付差{i}'] = np.nan
        if f'距離{i-1}' in df_copy.columns: df_copy[f'距離{i-1}'] = safe_float_convert(df_copy[f'距離{i-1}'])
        if f'距離{i}' in df_copy.columns: df_copy[f'距離{i}'] = safe_float_convert(df_copy[f'距離{i}'])
        if f'距離{i-1}' in df_copy.columns and f'距離{i}' in df_copy.columns: df_copy[f'距離差{i}'] = df_copy[f'距離{i-1}'] - df_copy[f'距離{i}']
        else: df_copy[f'距離差{i}'] = np.nan
    
    date_diff_cols = [f'日付差{i}' for i in range(1, num_past_races + 1)]; distance_diff_cols = [f'距離差{i}' for i in range(1, num_past_races + 1)] 
    df_copy[date_diff_cols] = df_copy[date_diff_cols].fillna(999); df_copy[distance_diff_cols] = df_copy[distance_diff_cols].fillna(0) 
    
    # 過去走統計量の計算
    stats_features_general = ['着順', '上がり', '斤量', '体重', '体重変化', '通過順_平均', '通過順_変動', '距離差', '走破時間_seconds', '走破時間_scaled'] 
    
    for feature in stats_features_general:
        past_cols = [f'{feature}{i}' for i in range(1, num_past_races + 1) if f'{feature}{i}' in df_copy.columns]
        if past_cols:
            for col in past_cols: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy[f'過去{num_past_races}走_{feature}_平均'] = df_copy[past_cols].mean(axis=1)
            df_copy[f'過去{num_past_races}走_{feature}_最大'] = df_copy[past_cols].max(axis=1)
            df_copy[f'過去{num_past_races}走_{feature}_最小'] = df_copy[past_cols].min(axis=1)
            # 標準偏差も追加
            df_copy[f'過去{num_past_races}走_{feature}_標準偏差'] = df_copy[past_cols].std(axis=1)

    # --- 新規追加: 距離を考慮した走破時間_scaledの過去走統計量 ---
    # 既存の '距離' カラムと '走破時間_scaled{i}' カラムが存在するかチェック
    if '距離' in df_copy.columns and f'走破時間_scaled{num_past_races}' in df_copy.columns: 
        
        # 各過去走の距離変化の絶対値を計算 (現在のレース距離との差)
        for i in range(1, num_past_races + 1):
            if f'距離{i}' in df_copy.columns:
                df_copy[f'距離差_現在_過去{i}'] = (df_copy['距離'] - df_copy[f'距離{i}']).abs()
            else:
                df_copy[f'距離差_現在_過去{i}'] = np.nan

        # 各行に対して、条件に合う走破時間_scaledのリストを作成
        def get_conditional_scaled_times_for_stats(row, num_past_races_val):
            current_distance = row['距離']
            
            # 優先順位1: 同じ距離のレースの走破時間_scaled を収集
            same_distance_times = []
            for i in range(1, num_past_races_val + 1):
                if pd.notna(row[f'距離{i}']) and row[f'距離{i}'] == current_distance and pd.notna(row[f'走破時間_scaled{i}']):
                    same_distance_times.append(row[f'走破時間_scaled{i}'])
            
            if same_distance_times:
                return same_distance_times # 同じ距離のレースがあればそれらを返す

            # 優先順位2: 同じ距離のレースがない場合、距離変化が最小で最も直近のレースを探す
            min_dist_diff = np.inf
            closest_time_found = np.nan
            closest_race_idx_found = -1 # 最も直近のレースを特定するためのインデックス (1が最も直近)

            for i in range(1, num_past_races_val + 1):
                if pd.notna(row[f'距離差_現在_過去{i}']) and pd.notna(row[f'走破時間_scaled{i}']):
                    dist_diff = row[f'距離差_現在_過去{i}']
                    
                    if dist_diff < min_dist_diff:
                        min_dist_diff = dist_diff
                        closest_time_found = row[f'走破時間_scaled{i}']
                        closest_race_idx_found = i 
                    elif dist_diff == min_dist_diff:
                        # 距離差が同じ場合は、より直近のレースを採用
                        if i < closest_race_idx_found: # 小さいiの方が直近
                             closest_time_found = row[f'走破時間_scaled{i}']
                             closest_race_idx_found = i
            
            if pd.notna(closest_time_found):
                return [closest_time_found] # 最も直近の距離差最小レースのタイムをリストで返す
            
            return [] # 該当するレースがない場合

        # 各行に適用して条件に合うタイムのリストを取得
        df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_times'] = df_copy.apply(
            lambda row: get_conditional_scaled_times_for_stats(row, num_past_races), axis=1
        )

        # 統計量を計算
        # NaNが含まれている可能性があるため、PandasのSeriesメソッドを活用
        df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_平均'] = df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_times'].apply(lambda x: pd.Series(x).mean())
        df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_最大'] = df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_times'].apply(lambda x: pd.Series(x).max())
        df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_最小'] = df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_times'].apply(lambda x: pd.Series(x).min())
        df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_標準偏差'] = df_copy[f'過去{num_past_races}走_条件_走破時間_scaled_times'].apply(lambda x: pd.Series(x).std())
        
        # 中間生成カラムを削除
        df_copy.drop(columns=[f'距離差_現在_過去{i}' for i in range(1, num_past_races + 1) if f'距離差_現在_過去{i}' in df_copy.columns], errors='ignore', inplace=True)
        df_copy.drop(columns=[f'過去{num_past_races}走_条件_走破時間_scaled_times'], errors='ignore', inplace=True)


    # --- ベイジアンレートの計算とマージ ---
    def calculate_bayesian_rate_internal(sub_df: pd.DataFrame, group_cols: List[str], target_col_name: str, C_val: int = 20, prior_rate_fallback: float = None) -> pd.DataFrame:
        if sub_df.empty or '着順' not in sub_df.columns or not all(c in sub_df.columns for c in group_cols): return pd.DataFrame(columns=group_cols + [target_col_name])
        prior_rate = (sub_df['着順'] == 1).mean() if prior_rate_fallback is None else prior_rate_fallback
        stats = sub_df.groupby(group_cols, observed=False).agg(wins=('着順', lambda x: (x == 1).sum()), races=('着順', 'size')).reset_index()
        stats[target_col_name] = (stats['wins'] + C_val * prior_rate) / (stats['races'] + C_val)
        return stats[group_cols + [target_col_name]]
    
    calculated_stats = {}
    if jockey_rates is None: # 学習時のみ統計量を計算
        df_temp = df_copy.copy()
        if '馬番グループ' not in df_temp.columns: df_temp['馬番グループ'] = create_dynamic_gate_group(df_temp)

        # 開催・馬場・馬番グループ別バイアス
        if all(c in df_temp.columns for c in ['開催', '場名', '芝・ダート', '馬番グループ']): 
            calculated_stats['venue_bias_by_gate_group'] = calculate_bayesian_rate_internal(df_temp, ['開催', '場名', '芝・ダート', '馬番グループ'], '開催バイアス_馬番グループ', C_val=10)
        
        # 前日バイアス（計算自体は行うが、予測時には別途結合）
        if all(c in df_temp.columns for c in ['日付', '場名', '芝・ダート', '馬番グループ', 'race_id']):
            df_temp_sorted = df_temp.sort_values(by=['日付', 'race_id']).copy()
            prev_day_bias_raw = calculate_bayesian_rate_internal(df_temp_sorted, ['日付', '場名', '芝・ダート', '馬番グループ'], '前日バイアス_馬番グループ', C_val=5)
            if pd.api.types.is_datetime64_any_dtype(prev_day_bias_raw['日付']): prev_day_bias_raw['日付'] = prev_day_bias_raw['日付'] + pd.to_timedelta(1, unit='D')
            prev_day_bias_raw.rename(columns={'日付': 'レース日付_prev'}, inplace=True)
            calculated_stats['prev_day_bias_by_gate_group'] = prev_day_bias_raw

        # 騎手勝率・競馬場勝率
        if 'jockey_id' in df_copy.columns:
            calculated_stats['jockey_rate'] = calculate_bayesian_rate_internal(df_copy, ['jockey_id'], '騎手勝率')
            calculated_stats['jockey_venue_rate'] = calculate_bayesian_rate_internal(df_copy, ['jockey_id', '場名'], '騎手競馬場勝率')
        
        # 体重関連統計量
        if '体重' in df_copy.columns and '体重1' in df_copy.columns:
            df_copy['体重差1_for_stats'] = df_copy['体重'] - safe_float_convert(df_copy['体重1'])
            if '馬' in df_copy.columns:
                 calculated_stats['horse_weight_stats'] = df_copy.groupby('馬').agg(weight_diff_mean=('体重差1_for_stats', 'mean'), weight_diff_std=('体重差1_for_stats', 'std')).reset_index()
        
        # 体重増減適性
        if '体重変化1' in df_copy.columns and '馬' in df_copy.columns:
            df_copy['体重増減カテゴリ'] = np.select([df_copy['体重変化1'] > 2, df_copy['体重変化1'] < -2], ['増加', '減少'], default='不変')
            calculated_stats['weight_change_suitability'] = calculate_bayesian_rate_internal(df_copy.dropna(subset=['体重増減カテゴリ']), ['馬', '体重増減カテゴリ'], '体重増減適性')
        
        # 馬場脚質適性
        if all(c in df_copy.columns for c in ['馬場', '脚質']): calculated_stats['track_running_style_suitability'] = calculate_bayesian_rate_internal(df_copy, ['馬場', '脚質'], '馬場脚質適性')
        
        # 騎手馬場適性
        if all(c in df_copy.columns for c in ['jockey_id', '馬場']): calculated_stats['jockey_track_suitability'] = calculate_bayesian_rate_internal(df_copy, ['jockey_id', '馬場'], '騎手馬場適性')
    
    stats_to_merge = calculated_stats if jockey_rates is None else {k: v for k, v in jockey_rates.items() if v is not None}
    
    if jockey_rates is not None and '体重変化1' in df_copy.columns: # 予測時にはカテゴリ再作成
        df_copy['体重増減カテゴリ'] = np.select([df_copy['体重変化1'] > 2, df_copy['体重変化1'] < -2], ['増加', '減少'], default='不変')

    merge_configs = {
        'jockey_rate': {'on': 'jockey_id'}, 
        'jockey_venue_rate': {'on': ['jockey_id', '場名']}, 
        'horse_weight_stats': {'on': '馬'},
        'weight_change_suitability': {'on': ['馬', '体重増減カテゴリ']}, 
        'track_running_style_suitability': {'on': ['馬場', '脚質']}, 
        'jockey_track_suitability': {'on': ['jockey_id', '馬場']},
        'venue_bias_by_gate_group': {'on': ['開催', '場名', '芝・ダート', '馬番グループ']},
    }

    for key, conf in merge_configs.items():
        if key in stats_to_merge and not stats_to_merge[key].empty:
            on_cols = conf['on'] if isinstance(conf['on'], list) else [conf['on']]
            if all(c in df_copy.columns for c in on_cols):
                if 'jockey_id' in on_cols and 'jockey_id' in stats_to_merge[key].columns:
                    stats_to_merge[key]['jockey_id'] = stats_to_merge[key]['jockey_id'].astype(str).fillna('nan')
                df_copy = pd.merge(df_copy, stats_to_merge[key], on=on_cols, how='left')
    
    if 'prev_day_bias_by_gate_group' in stats_to_merge and not stats_to_merge['prev_day_bias_by_gate_group'].empty:
        df_copy = pd.merge(df_copy, stats_to_merge['prev_day_bias_by_gate_group'], left_on=['日付', '場名', '芝・ダート', '馬番グループ'], right_on=['レース日付_prev', '場名', '芝・ダート', '馬番グループ'], how='left').drop(columns=['レース日付_prev'], errors='ignore')
    
    # 体重の前走比標準化
    if '体重' in df_copy.columns and '体重1' in df_copy.columns:
        df_copy['体重差1_for_stats'] = df_copy['体重'] - safe_float_convert(df_copy['体重1'])
        if 'weight_diff_std' in df_copy.columns and 'weight_diff_mean' in df_copy.columns:
            df_copy['体重_前走比_標準化'] = np.where(df_copy['weight_diff_std'].fillna(0) == 0, 0, (df_copy['体重差1_for_stats'].fillna(0) - df_copy['weight_diff_mean'].fillna(0)) / df_copy['weight_diff_std'])
    
    return df_copy, calculated_stats


def add_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース内の相対的な特徴量（平均、最大、最小、偏差など）を追加する。
    """
    if df.empty: return pd.DataFrame(columns=df.columns)
    if 'race_id' not in df.columns or '馬' not in df.columns: return df
    df_copy = df.copy() 
    if '出走頭数' not in df_copy.columns: df_copy['出走頭数'] = df_copy.groupby('race_id')['馬'].transform('count')
    
    # ここでは、新しい条件付き走破時間_scaledの統計量を使って相対特徴量を作成
    relative_features = [
        '斤量', '騎手勝率', '騎手競馬場勝率', 
        f'過去{config.NUM_PAST_RACES}走_着順_平均', f'過去{config.NUM_PAST_RACES}走_上がり_平均', 
        '開催バイアス_馬番グループ', '前日バイアス_馬番グループ', 
        f'過去{config.NUM_PAST_RACES}走_条件_走破時間_scaled_平均', # ★ 新しい特徴量を使用 ★
        f'過去{config.NUM_PAST_RACES}走_条件_走破時間_scaled_最大', # ★ 新しい特徴量を使用 ★
        f'過去{config.NUM_PAST_RACES}走_条件_走破時間_scaled_最小', # ★ 新しい特徴量を使用 ★
    ] 
    
    for feature in relative_features:
        if feature in df_copy.columns:
            df_copy[feature] = pd.to_numeric(df_copy[feature], errors='coerce')
            race_stats = df_copy.groupby('race_id')[feature].agg(['mean', 'max', 'min']).rename(columns={'mean': f'{feature}_race_mean', 'max': f'{feature}_race_max', 'min': f'{feature}_race_min'})
            df_copy = pd.merge(df_copy, race_stats, on='race_id', how='left')
            df_copy[f'{feature}_race_dev'] = df_copy[f'{feature}_race_mean'] - df_copy[feature]
            df_copy[f'{feature}_race_max_diff'] = df_copy[f'{feature}_race_max'] - df_copy[feature]
            df_copy[f'{feature}_race_min_diff'] = df_copy[feature] - df_copy[f'{feature}_race_min']
        else:
            for suffix in ['_race_mean', '_race_max', '_race_min', '_dev', '_max_diff', '_min_diff']: df_copy[f'{feature}{suffix}'] = np.nan
    return df_copy

def encode_and_finalize(df: pd.DataFrame, categorical_features: List[str], label_encoders: Dict[str, LabelEncoder] = None) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    カテゴリカル特徴量のエンコード（Label Encoding）と最終処理を行う。
    """
    if df.empty: return pd.DataFrame(columns=categorical_features), {} if label_encoders is None else label_encoders
    df_copy = df.copy() 
    if label_encoders is None:
        label_encoders = {}
        for col in categorical_features:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).fillna('unknown')
                le = LabelEncoder()
                # 学習時（label_encodersがNone）は、ユニークな値に'unknown'も加えてfitする
                classes = np.append(df_copy[col].unique(), 'unknown')
                le.fit(np.unique(classes)) 
                df_copy[col] = le.transform(df_copy[col])
                label_encoders[col] = le
            else:
                df_copy[col] = np.nan # カラムがない場合はNaN
    else: # 予測時（学習済みのlabel_encodersがある場合）
        for col in categorical_features:
            if col in df_copy.columns:
                le = label_encoders.get(col)
                if le is None: continue # そのカラムのエンコーダがなければスキップ
                
                # エンコーダにないラベルは 'unknown' として扱う
                known_labels = set(le.classes_)
                df_copy[col] = df_copy[col].astype(str).fillna('unknown').apply(lambda x: x if x in known_labels else 'unknown')
                df_copy[col] = le.transform(df_copy[col])
            else:
                df_copy[col] = np.nan # カラムがない場合はNaN

    return df_copy, label_encoders