# C:\KeibaAI\code\a2_build_features\m02_build_training_data.py

"""
生のレース結果データを元に、特徴量エンジニアリングを行い、
指定されたモデルタイプ（A, B, C）用の学習データと変換器を生成する統合スクリプト。

■ このスクリプトの役割
- m02A, m02B, m02C の機能を一つに統合し、コードの重複を排除する。
- config.py の MODEL_CONFIGS 設定に基づき、モデルごとに異なる特徴量の選択や処理を行う。
- モデルA: 予測時に利用不可能な当日情報（馬場、天気など）を含まないモデル。
- モデルB: オッズや人気などを除外し、純粋な能力値を測ることを目指したモデル。
- モデルC: 当日のリアルタイムな馬場バイアスなどを特徴量に加えた、最も情報量の多いモデル。

■ 使い方
コマンドライン引数で、作成したいモデルタイプを指定する。
- モデルAのデータを生成: python code/a2_build_features/m02_build_training_data.py A
- モデルBのデータを生成: python code/a2_build_features/m02_build_training_data.py B
- モデルCのデータを生成: python code/a2_build_features/m02_build_training_data.py C
"""
# --- プロジェクトパスとライブラリのインポート ---
import sys
import os
import pandas as pd
from joblib import dump
import warnings
from typing import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))

# --- モジュールインポート ---
import config
from utils.feature_pipeline import(
    preprocess_and_clean, add_past_race_features, engineer_advanced_features, 
    add_race_level_features, encode_and_finalize
)

def load_and_combine_data(start_year: int, end_year: int) -> pd.DataFrame:
    """指定された期間の生データを読み込み、一つのDataFrameに結合する（共通関数）"""
    print("--- Loading and combining raw data ---")
    df_list = []
    dtype_spec = {'race_id': str, 'horse_id': str, 'jockey_id': str}
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(config.DATA_DIR, f"{year}.csv")
        if not os.path.exists(file_path): continue
        try:
            df = pd.read_csv(file_path, encoding="SHIFT-JIS", header=0, low_memory=False, 
                            parse_dates=['日付'], 
                            date_parser=lambda x: pd.to_datetime(x, format='%Y年%m月%d日', errors='coerce'),
                            dtype=dtype_spec)
            df_list.append(df)
        except Exception as e:
            print(f"ERROR: Failed to read {file_path}: {e}")
            
    if not df_list:
        print("FATAL: No data to process."); return pd.DataFrame()
        
    raw_df = pd.concat(df_list, ignore_index=True)
    raw_df['race_id'] = pd.to_numeric(raw_df['race_id'], errors='coerce')
    print(f"--- Data loading complete ({len(raw_df)} rows) ---")
    return raw_df

def main(model_type: str):
    """指定されたモデルタイプの学習データを生成するメイン関数"""
    
    if model_type not in config.MODEL_CONFIGS:
        print(f"[FATAL] Model type '{model_type}' is not defined in config.MODEL_CONFIGS.")
        return

    model_conf = config.MODEL_CONFIGS[model_type]
    
    # --- 保存先ディレクトリ設定 ---
    output_base_dir = os.path.join(config.ENCODED_DIR, f'{model_type}_encoded')
    artifacts_base_dir = os.path.join(config.ARTIFACTS_DIR, f'{model_type}_artifacts')
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(artifacts_base_dir, exist_ok=True)
    
    print(f"\n{'='*20} Building Training Data for Model {model_type} {'='*20}")
    print(f"    Output directory: {output_base_dir}")
    print(f"    Artifacts directory: {artifacts_base_dir}")
        
    # --- データ読み込み ---
    raw_df_combined = load_and_combine_data(config.BUILD_START_YEAR, config.BUILD_END_YEAR)
    if raw_df_combined.empty: return

    # time_scaler を初期化
    global_time_scaler: Dict[str, float] = None 

    for track in config.TRACK_TYPES:
        print(f"\n--- Processing for {track.upper()} track ---")
        track_map = {'turf': '芝', 'dirt': 'ダ'}
        df_track = raw_df_combined[raw_df_combined['芝・ダート'].str.strip() == track_map[track]].copy()
        if df_track.empty: 
            print(f"[WARN] Raw data for {track.upper()} track is empty. Skipping.")
            continue
        
        # --- パイプライン実行 (共通処理) ---
        # preprocess_and_clean で time_scaler をフィットさせ、'走破時間_scaled' を生成する
        # ここでフィットされた time_scaler を global_time_scaler として取得
        df_processed_all_data, global_time_scaler = preprocess_and_clean(df_track.copy(), time_scaler=None) 
        
        # フィットされた time_scaler を保存
        dump(global_time_scaler, os.path.join(artifacts_base_dir, f'time_scaler_{track}.joblib'))
        
        if df_processed_all_data.empty:
            print(f"[WARN] Preprocessed data for {track.upper()} is empty. Skipping.")
            continue
            
        df_for_stats = df_processed_all_data[df_processed_all_data['year'] <= config.JOCKEY_RATE_BUILD_END_YEAR].copy()
        if df_for_stats.empty:
            print(f"[WARN] Stats calculation data for {track.upper()} is empty. Skipping.")
            continue
            
        df_with_past_for_stats = add_past_race_features(df_for_stats, config.NUM_PAST_RACES, config.PAST_RACE_FEATURES)
        df_with_past_for_stats = df_with_past_for_stats.groupby(['race_id', '馬'], as_index=False).last()
        
        # engineer_advanced_features は time_scaler を引数として受け取らない (preprocess_and_cleanで処理済みのため)
        # そのため、返り値も計算された統計量のみとなる
        df_for_stats_featured, calculated_stats = engineer_advanced_features(df_with_past_for_stats, config.NUM_PAST_RACES, jockey_rates=None)
        
        # engineer_advanced_features で計算された新しい走破時間_scaledの統計量が df_for_stats_featured に含まれていることを確認
        # ただし、現状 calculated_stats には走破時間_scaledの統計量は含まれない想定

        # --- モデル別 artifact 保存 ---
        stats_to_save = model_conf.get('stats_to_save', [])
        print(f"[INFO] Saving {len(stats_to_save)} types of artifacts for Model {model_type}.")
        for key in stats_to_save:
            if key in calculated_stats:
                dump(calculated_stats[key], os.path.join(artifacts_base_dir, f'{key}_{track}.joblib'))
        
        # --- 全期間データへの特徴量適用 ---
        df_with_past_all_data = add_past_race_features(df_processed_all_data, config.NUM_PAST_RACES, config.PAST_RACE_FEATURES)
        df_with_past_all_data = df_with_past_all_data.groupby(['race_id', '馬'], as_index=False).last()
        
        # 全期間データに対して engineer_advanced_features を適用
        # ここでは calculated_stats (ベイジアン勝率など) を適用し、time_scaler は不要 (既にpreprocess_and_cleanで処理済み)
        df_featured, _ = engineer_advanced_features(df_with_past_all_data, config.NUM_PAST_RACES, jockey_rates=calculated_stats)
        
        df_race_level = add_race_level_features(df_featured)
        if df_race_level.empty:
            print(f"[WARN] Final data is empty after feature engineering for {track.upper()}. Skipping.")
            continue

        # --- モデルごとの分岐処理 ---
        categorical_features = model_conf['categorical_features']
        
        # モデルC のみリアルタイムバイアス計算の特殊処理
        if model_type == 'C':
            print("[INFO] Model C: Calculating real-time daily track bias...")
            df_encoded = process_realtime_features_for_c(df_race_level, categorical_features, artifacts_base_dir, track)
            if df_encoded.empty:
                print(f"[WARN] Model C processing resulted in empty data for {track.upper()}. Skipping.")
                continue
        
        # モデルA, B の共通処理
        else:
            drop_conf = model_conf.get('features_to_drop', {})
            cols_to_drop = drop_conf.get('common', []) + drop_conf.get(track, [])
            df_final = df_race_level.drop(columns=[c for c in cols_to_drop if c in df_race_level.columns], errors='ignore').copy()
            print(f"[INFO] Dropped {len(cols_to_drop)} columns for Model {model_type} ({track}). Final shape: {df_final.shape}")
            
            df_encoded, label_encoders_calculated = encode_and_finalize(df_final, categorical_features, label_encoders=None)
            dump(label_encoders_calculated, os.path.join(artifacts_base_dir, f'label_encoders_{track}.joblib'))
        
        # --- 最終データの保存 ---
        output_path = os.path.join(output_base_dir, f'encoded_data_{track}.csv')
        df_encoded.to_csv(output_path, index=False)
        print(f"\nSUCCESS: Full training data for Model {model_type} ({track}) saved to -> {output_path}")
        print(f"  - Total rows: {len(df_encoded)}")
        print(f"  - Total features (columns): {len(df_encoded.columns)}")
        print(f"  - Example features: {list(df_encoded.columns[:5])} ... {list(df_encoded.columns[-5:])}\n")
            
    print(f"\n--- All training data building processes for Model {model_type} finished. ---")


def process_realtime_features_for_c(df: pd.DataFrame, categorical_features: List[str], artifacts_base_dir: str, track: str) -> pd.DataFrame:
    """モデルC専用のリアルタイム特徴量（当日バイアス）を計算・結合する関数"""
    
    # モデルCでのみ使うヘルパー関数
    def calculate_bayesian_rate_m02(sub_df: pd.DataFrame, group_cols: List[str], target_col: str, prior_rate: float) -> pd.DataFrame:
        if sub_df.empty or '着順' not in sub_df.columns: return pd.DataFrame(columns=group_cols + [target_col])
        C = 5 # 当日バイアスなのでCは小さめ
        stats = sub_df.groupby(group_cols, observed=False).agg(wins=('着順', lambda x: (x == 1).sum()), races=('着順', 'size')).reset_index()
        stats[target_col] = (stats['wins'] + C * prior_rate) / (stats['races'] + C)
        return stats[group_cols + [target_col]]

    global_prior_rate = (df['着順'] == 1).mean()
    final_encoded_df_list = []
    df.sort_values(by=['日付', 'race_id'], inplace=True)

    # LabelEncoderを先に全データで学習させ、保存
    df_encoded_temp, label_encoders_calculated = encode_and_finalize(df.copy(), categorical_features, label_encoders=None)
    dump(label_encoders_calculated, os.path.join(artifacts_base_dir, f'label_encoders_{track}.joblib'))
    
    unique_dates = df['日付'].unique()
    for i, current_date_np in enumerate(unique_dates):
        if i % 100 == 0:
             print(f"\r    Processing date {pd.to_datetime(current_date_np).strftime('%Y-%m-%d')} ({i+1}/{len(unique_dates)})...", end="")

        daily_races_all_venues = df[df['日付'] == current_date_np].copy()
        for venue_name, daily_races_at_venue in daily_races_all_venues.groupby('場名'):
            races_so_far_today = pd.DataFrame()
            for race_id_current, current_race_df in daily_races_at_venue.sort_values('race_id').groupby('race_id'):
                
                bias_stats = calculate_bayesian_rate_m02(races_so_far_today, ['馬番グループ'], '当日馬番バイアス', global_prior_rate)
                merged_df = pd.merge(current_race_df, bias_stats, on='馬番グループ', how='left')
                
                # バイアスのNaNを前日→開催→グローバルの優先度で埋める
                merged_df['当日馬番バイアス'].fillna(merged_df['前日バイアス_馬番グループ'], inplace=True)
                merged_df['当日馬番バイアス'].fillna(merged_df['開催バイアス_馬番グループ'], inplace=True)
                merged_df['当日馬番バイアス'].fillna(global_prior_rate, inplace=True)
                
                encoded_race_df, _ = encode_and_finalize(merged_df, categorical_features, label_encoders_calculated)
                final_encoded_df_list.append(encoded_race_df)

                races_so_far_today = pd.concat([races_so_far_today, current_race_df[['着順', '馬番グループ']]], ignore_index=True)
    
    print() # 改行
    if not final_encoded_df_list: return pd.DataFrame()
    return pd.concat(final_encoded_df_list, ignore_index=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code/a2_build_features/m02_build_training_data.py [A|B|C]")
        sys.exit(1)
    
    model_type_arg = sys.argv[1].upper()
    if model_type_arg not in ['A', 'B', 'C']:
        print(f"Invalid model type '{model_type_arg}'. Please choose from 'A', 'B', or 'C'.")
        sys.exit(1)
        
    main(model_type_arg)