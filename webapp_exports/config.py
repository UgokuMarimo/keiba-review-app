# C:\KeibaAI\config.py (最終確定版)

import os
import ast # このファイル内で関数を使うため、astを再度インポート

# --- プロジェクトパス設定 ---
# このconfig.pyファイル自身があるディレクトリをプロジェクトのルートとする
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# データベースの保存先を 'C:\KeibaAI\predictions.db' に変更
DB_PATH = os.path.join(PROJECT_ROOT, 'predictions.db')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
MODEL_DIR_BASE = os.path.join(PROJECT_ROOT, 'models')
TUNING_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'tuning_results')
ENCODED_DIR = os.path.join(PROJECT_ROOT, 'encoded')
PREDICT_DATA_DIR = os.path.join(PROJECT_ROOT, 'predict_data')

WEBAPP_EXPORTS_DIR = os.path.join(PROJECT_ROOT, 'webapp_exports')


# ★★★ ここからが修正箇所 ★★★
# ヘルパー関数をこのファイル内に戻す
def load_best_params_from_file(filepath: str) -> dict:
    """Optunaのスタディ結果のようなテキストファイルから最適なパラメータを読み込む。"""
    if not os.path.exists(filepath):
        # print(f"[WARN] Hyperparameter file not found: {filepath}. Returning empty dict.")
        return {}
    params = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Best params:" in line: break
            for line in f:
                line = line.strip()
                if not line.startswith("'"): continue
                key, value = line.split(':', 1)
                key = ast.literal_eval(key)
                value = ast.literal_eval(value.rstrip(','))
                params[key] = value
    except Exception as e:
        print(f"[ERROR] Failed to parse hyperparameter file {filepath}: {e}")
        return {}
    return params
# ★★★ 修正ここまで ★★★


# --- 期間設定 & バージョン管理 ---
EXPERIMENT_VERSION = "B_prod_model"
SCRAPE_START_YEAR = 2025
SCRAPE_END_YEAR = 2026
BUILD_START_YEAR = 1990
BUILD_END_YEAR = 2026
TRAINING_START_YEAR = 1993
EVALUATION_YEAR = 2025
JOCKEY_RATE_BUILD_END_YEAR = 2024

# --- LightGBM ハイパーパラメータ設定 ---
LGB_BASE_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1, 
    "boosting_type": "gbdt", "class_weight": "balanced"
}

LGB_PARAMS_WIN_TURF = {**LGB_BASE_PARAMS, **load_best_params_from_file(os.path.join(TUNING_RESULTS_DIR, 'best_params_lgbm_turf_win.txt'))}
LGB_PARAMS_WIN_DIRT = {**LGB_BASE_PARAMS, **load_best_params_from_file(os.path.join(TUNING_RESULTS_DIR, 'best_params_lgbm_dirt_win.txt'))}
LGB_PARAMS_PLACE_TURF = {**LGB_BASE_PARAMS, **load_best_params_from_file(os.path.join(TUNING_RESULTS_DIR, 'best_params_lgbm_turf_place.txt'))}
LGB_PARAMS_PLACE_DIRT = {**LGB_BASE_PARAMS, **load_best_params_from_file(os.path.join(TUNING_RESULTS_DIR, 'best_params_lgbm_dirt_place.txt'))}


# --- 特徴量エンジニアリング設定 ---
# --- 特徴量エンジニアリング設定 ---
TRACK_TYPES = ['turf', 'dirt']
NUM_PAST_RACES = 5
PAST_RACE_FEATURES = [
    '馬番', 'jockey_id', '斤量', 'オッズ', '人気', '体重', '体重変化', 
    '上がり', '通過順', '着順', '距離', 'クラス', 
    '走破時間_seconds', 
    '走破時間_scaled', # ここは維持
    '芝・ダート', '天気', '馬場',
    '脚質', 
]

# --- モデル別設定 ---
MODEL_CONFIGS = {
    'A': {
        'features_to_drop': {
            'common': [
                '馬場', '天気', '開催バイアス_馬番グループ', '前日バイアス_馬番グループ', '当日馬番バイアス',
                '馬場_race_mean', '馬場_race_max', '馬場_race_min', '馬場_race_dev', '馬場_race_max_diff', '馬場_race_min_diff',
                '天気_race_mean', '天気_race_max', '天気_race_min', '天気_race_dev', '天気_race_max_diff', '天気_race_min_diff',
                '開催バイアス_馬番グループ_race_mean', '開催バイアス_馬番グループ_race_max', '開催バイアス_馬番グループ_race_min', 
                '開催バイアス_馬番グループ_race_dev', '開催バイアス_馬番グループ_race_max_diff', '開催バイアス_馬番グループ_race_min_diff',
                '前日バイアス_馬番グループ_race_mean', '前日バイアス_馬番グループ_race_max', '前日バイアス_馬番グループ_race_min', 
                '前日バイアス_馬番グループ_race_dev', '前日バイアス_馬番グループ_race_max_diff', '前日バイアス_馬番グループ_race_min_diff',
            ],
            'turf': [], 'dirt': []
        },
        'categorical_features': [
            '馬', 'レース名', '開催', '場名', '脚質', '体重増減カテゴリ', '馬番グループ',
            *[f'脚質{i}' for i in range(1, NUM_PAST_RACES + 1)]
        ],
        'stats_to_save': ['jockey_rate', 'jockey_venue_rate', 'horse_weight_stats', 'weight_change_suitability', 'track_running_style_suitability', 'jockey_track_suitability']
    },
    'B': {
        'leakage_features': ['オッズ', '人気'], 
        'features_to_drop': {
            'common': [
                '前日バイアス_馬番グループ', '当日馬番バイアス', 
                '前日バイアス_馬番グループ_race_mean', '前日バイアス_馬番グループ_race_max', 
                '前日バイアス_馬番グループ_race_min', '前日バイアス_馬番グループ_race_dev', 
                '前日バイアス_馬番グループ_race_max_diff', '前日バイアス_馬番グループ_race_min_diff',
                'grade'
            ],
            'turf': ['体重_前走比_標準化', '体重増減適性', '馬場脚質適性', '騎手馬場適性', 'is_niigata_1000m'],
            'dirt': []
        },
        'categorical_features': [
            '馬', 'レース名', '開催', '場名', '脚質', '体重増減カテゴリ', '馬番グループ', '馬場', '天気',
            *[f'脚質{i}' for i in range(1, NUM_PAST_RACES + 1)]
        ],
        'stats_to_save': ['jockey_rate', 'jockey_venue_rate', 'horse_weight_stats', 'weight_change_suitability', 'track_running_style_suitability', 'jockey_track_suitability', 'venue_bias_by_gate_group']
    },
    'C': {
        'features_to_drop': {'common': [], 'turf': [], 'dirt': []},
        'categorical_features': [
            '馬', 'レース名', '開催', '場名', '脚質', '体重増減カテゴリ', '馬番グループ',
            *[f'脚質{i}' for i in range(1, NUM_PAST_RACES + 1)]
        ],
        'stats_to_save': ['jockey_rate', 'jockey_venue_rate', 'jockey_rides', 'horse_weight_stats', 'weight_change_suitability', 'track_running_style_suitability', 'jockey_track_suitability', 'venue_bias_by_gate_group', 'prev_day_bias_by_gate_group']
    }
}

# --- その他 ---
PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}

# --- 通知設定 ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1422401164821008394/bZHt-71EUIaRDCEZgv1DHL9NPgZwax58xjIT0SVhIBY5JyWZN-uKlfVGeV9pUwGNBLEn"


# JRAの競馬場IDマッピング (race_idの4-6桁目)
PLACE_MAP_IDS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}

# JRAの競馬場名 (場名からJRAを判定するため)
JRA_PLACE_NAMES = set([
    "札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"
])

# 過去走特徴量を生成する際の過去レース数
NUM_PAST_RACES = 5 
# 個別ページからスクレイピングする過去走の最大数 (少し多めに取得)
NUM_PAST_RACES_TO_SCRAPE = 10 
