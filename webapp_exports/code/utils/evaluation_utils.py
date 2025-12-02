# C:\KeibaAI\code\utils\evaluation_utils.py (最終版)

"""
評価スクリプト群で使用する共通の関数を定義するモジュール。
"""
import os
import sys
import pandas as pd
from typing import Tuple

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..')); sys.path.append(PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
import config

def load_data_for_evaluation(track: str, year: int) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """評価に必要な特徴量データと、正解・オッズデータを読み込む"""
    print(f"\n--- Loading data for {track.upper()} track evaluation for year {year} ---")
    
    try:
        # モデル入力用の特徴量データをロード
        encoded_path = os.path.join(config.ENCODED_DIR, f'encoded_data_{track}.csv')
        encoded_data = pd.read_csv(encoded_path, low_memory=False, parse_dates=['日付'])
        test_features = encoded_data[encoded_data['year'] == year].copy()
        
        # 回収率計算用の、生のレース結果データをロード
        raw_path = os.path.join(config.DATA_DIR, f'{year}.csv')
        raw_data = pd.read_csv(raw_path, encoding="SHIFT-JIS", low_memory=False)
        test_raw = raw_data.copy()
        
        if test_features.empty or test_raw.empty:
            print(f"[WARN] No test data found for year {year}. Skipping.")
            return None, None
            
        return test_features, test_raw
        
    except FileNotFoundError as e:
        print(f"[ERROR] Required data file not found for evaluation: {e}")
        return None, None

def calculate_return_rate(predictions_df: pd.DataFrame, raw_data_df: pd.DataFrame) -> Tuple[float, float, int]: 
    """単勝回収率と的中率を計算する (堅牢化・最終版)"""
    
    if predictions_df.empty:
        return 0.0, 0.0, 0

    # 渡されたデータフレームに必要なカラムだけを抽出する
    bet_targets_df = predictions_df[['race_id', '馬番']].copy()
    raw_data_df = raw_data_df.copy()
    
    # マージの前に、両方のデータフレームのキーの型を安全な文字列型に統一する
    bet_targets_df['race_id'] = bet_targets_df['race_id'].astype(str)
    raw_data_df['race_id'] = raw_data_df['race_id'].astype(str)
    bet_targets_df['馬番'] = bet_targets_df['馬番'].astype(int)
    raw_data_df['馬番'] = pd.to_numeric(raw_data_df['馬番'], errors='coerce')
    
    # 賭け対象の馬に、生のオッズ・着順データをマージ
    # raw_data_df から必要なカラムだけを選択
    raw_subset = raw_data_df[['race_id', '馬番', 'オッズ', '着順']].dropna(subset=['馬番'])
    merged_df = pd.merge(bet_targets_df, raw_subset, on=['race_id', '馬番'], how='left')

    # 投資対象となったレースの数を投資回数とする (マージに成功したもののみ)
    betting_df = merged_df.dropna(subset=['オッズ', '着順'])
    if betting_df.empty:
        return 0.0, 0.0, 0

    total_investment = len(betting_df) * 100
    
    # 1着だったレースに絞り込む
    win_df = betting_df[pd.to_numeric(betting_df['着順'], errors='coerce') == 1].copy()
    
    # 的中数
    total_wins = len(win_df)
    
    # 払い戻し合計を計算
    win_df['payout'] = pd.to_numeric(win_df['オッズ'], errors='coerce') * 100
    total_return = win_df['payout'].sum()

    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0
    win_rate = (total_wins / len(betting_df)) * 100 if len(betting_df) > 0 else 0
    
    return return_rate, win_rate, len(betting_df)