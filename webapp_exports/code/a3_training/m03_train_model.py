# C:\KeibaAI\code\a3_training\m03_train_model.py

"""
モデル学習、評価、本番モデル構築を行うための統合スクリプト。

■ 機能
このスクリプトは、コマンドライン引数に応じて3つの主要な動作モードを切り替える。
1. 評価モード (`--mode eval`):
   - 指定されたテスト年で性能を評価する。
   - 1着率(win)または3着内率(place)モデルのAUCスコアを出力する。

2. 本番モデル構築モード (`--mode prod`):
   - 利用可能な全データを使用して、本番予測用のモデルを学習・保存する。

3. ハイパーパラメータチューニングモード (`--mode tune`):
   - Optunaを使い、LightGBMの最適なハイパーパラメータを探索する。

■ 使い方
基本的なコマンドは以下の要素で構成されます。
`python code/a3_training/m03_train_model.py --mode [モード] --model-type [モデル] --target [予測対象] --track [コース]`

---
[モード1: 性能評価 の例]
# 2025年のデータで、モデルB (芝) の1着率モデルの性能を評価する場合
python code/a3_training/m03_train_model.py --mode eval --model-type B --target win --track turf --evaluation-year 2025
---
[モード2: 本番モデル構築 の例]
# モデルB (芝) の1着率予測モデルを全データで学習・保存する場合
python code/a3_training/m03_train_model.py --mode prod --model-type B --target win --track turf

# モデルB (ダート) の1着率予測モデルも同様に作成
python code/a3_training/m03_train_model.py --mode prod --model-type B --target win --track dirt

---
[モード3: パラメータ探索 の例]
# モデルB (芝) の1着率モデルのハイパーパラメータを100回試行して探索する場合
python code/a3_training/m03_train_model.py --mode tune --model-type B --target win --track turf --n-trials 100

---
[応用: ループを使った一括実行の例]
# モデルBの本番モデル(win/place, turf/dirt)を全て構築する
for track in turf dirt; do
  for target in win place; do
    echo "--- Building Model B: ${track} / ${target} ---"
    python code/a3_training/m03_train_model.py --mode prod --model-type B --target ${target} --track ${track}
  done
done
"""
import sys
import os
import pandas as pd
import lightgbm as lgb
import optuna
import argparse
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump
from typing import Tuple, Dict, Any

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
# ---
import config

def get_model_params(track: str, target_type: str) -> Dict[str, Any]:
    """configから指定されたモデルのハイパーパラメータを取得する"""
    param_key = f"LGB_PARAMS_{target_type.upper()}_{track.upper()}"
    params = getattr(config, param_key, None)
    if params is None:
        raise ValueError(f"Parameters for {param_key} not found in config.py")
    return params

def load_data(model_type: str, track: str) -> pd.DataFrame | None:
    """指定されたモデルとトラックのエンコード済みデータを読み込む"""
    file_path = os.path.join(config.ENCODED_DIR, f'{model_type}_encoded', f'encoded_data_{track}.csv')
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found: {file_path}"); return None
    print(f"--- Loading data from: {file_path} ---")
    return pd.read_csv(file_path, low_memory=False)

def prepare_features_and_labels(df: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """特徴量(X)と目的変数(y)を準備する"""
    model_conf = config.MODEL_CONFIGS.get(model_type, {})
    
    # 共通の削除対象カラム
    cols_to_drop = [
        'race_id', 'horse_id', '騎手', '馬', '日付', 'レース名', '開催', 'year', '着順', '通過順', 
        '芝・ダート', # このカラムはモデル学習のトラック分けで不要になるため削除
        f'過去{config.NUM_PAST_RACES}走_条件_走破時間_scaled_times' # engineer_advanced_features での中間生成カラム
    ]
    
    # レース結果に紐づくリーク情報を削除
    # これらは予測時点では知り得ない情報であり、現在のレースの学習データからは除外する
    # 過去データ (例: 上がり1, 走破時間_seconds1 など) は残す
    leak_features_current_race = [
        '上がり', '走破時間_seconds', '走破時間_scaled', '通過順_平均', 
        '通過順_変動', '脚質'
    ]
    cols_to_drop.extend(leak_features_current_race)

    # 過去の日付カラムも削除（特徴量としては不要な日付データ）
    past_date_cols = [f'日付{i}' for i in range(1, config.NUM_PAST_RACES + 2)]
    cols_to_drop.extend(past_date_cols)

    # `走破時間_scaled` を直接使って生成されていた古いレースレベル特徴量を削除
    # これらは、新しい条件付き走破時間_scaledの統計量に置き換えられたため、削除する
    old_scaled_race_level_features = [
        '走破時間_scaled_race_mean', '走破時間_scaled_race_max', '走破時間_scaled_race_min', 
        '走破時間_scaled_race_dev', '走破時間_scaled_race_max_diff', '走破時間_scaled_race_min_diff'
    ]
    cols_to_drop.extend(old_scaled_race_level_features)
    
    # モデルBの場合、configで定義された追加のリーク情報も削除
    if model_type == 'B':
        leakage_cols_from_config = model_conf.get('leakage_features', [])
        cols_to_drop.extend([col for col in leakage_cols_from_config if col not in cols_to_drop])
        
    # ドロップするカラムを実際にデータフレームから削除
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 目的変数
    y_raw = df['着順']
    return X, y_raw

# --- Mode 1: Evaluation ---
def run_evaluation(model_type: str, target_type: str, track: str, evaluation_year: int):
    print(f"\n--- [EVALUATION MODE] Model: {model_type}, Target: {target_type}, Track: {track}, Test Year: {evaluation_year} ---")
    df = load_data(model_type, track)
    if df is None: return

    train_df = df[df['year'] < evaluation_year].copy()
    test_df = df[df['year'] == evaluation_year].copy()
    if train_df.empty or test_df.empty:
        print(f"[WARN] Not enough data for evaluation. Train: {len(train_df)}, Test: {len(test_df)}"); return

    X_train, y_train_raw = prepare_features_and_labels(train_df, model_type)
    X_test, y_test_raw = prepare_features_and_labels(test_df, model_type)

    y_train = (y_train_raw == 1) if target_type == 'win' else (y_train_raw <= 3)
    y_test = (y_test_raw == 1) if target_type == 'win' else (y_test_raw <= 3)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    params = get_model_params(track, target_type)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_imputed, y_train)

    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n>>> RESULT for {track.upper()} {target_type.upper()}:")
    print(f"    Test Set AUC: {auc_score:.4f}")
    print("-" * 50)

# --- Mode 2: Production Model Building ---
def run_production(model_type: str, target_type: str, track: str):
    print(f"\n--- [PRODUCTION MODE] Building model for {model_type} / {target_type} / {track} ---")
    df = load_data(model_type, track)
    if df is None: return

    output_dir = os.path.join(config.MODEL_DIR_BASE, f'{model_type}_models', config.EXPERIMENT_VERSION)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Models will be saved to: {output_dir}")

    X, y_raw = prepare_features_and_labels(df, model_type)
    y = (y_raw == 1) if target_type == 'win' else (y_raw <= 3)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    params = get_model_params(track, target_type)
    # 本番モデルはGPUを使って高速化
    params['device'] = 'gpu'
    model = lgb.LGBMClassifier(**params)
    model.fit(X_imputed, y)

    dump(imputer, os.path.join(output_dir, f'imputer_{track}_{target_type}.joblib'))
    model.booster_.save_model(os.path.join(output_dir, f'lgb_model_{track}_{target_type}.txt'))
    
    print(f"\n>>> SUCCESS: Production model and imputer for {track.upper()}/{target_type.upper()} saved.")
    print("-" * 50)

# --- Mode 3: Hyperparameter Tuning ---
def run_tuning(model_type: str, target_type: str, track: str, n_trials: int):
    print(f"\n--- [TUNING MODE] Tuning for {model_type} / {target_type} / {track} ---")
    df = load_data(model_type, track)
    if df is None: return

    train_df = df[df['year'] < config.TEST_YEAR].copy()
    X, y_raw = prepare_features_and_labels(train_df, model_type)
    y = (y_raw == 1) if target_type == 'win' else (y_raw <= 3)
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    def objective(trial: optuna.trial.Trial) -> float:
        param = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'boosting_type': 'gbdt', 'class_weight': 'balanced',
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }
        X_train, X_valid, y_train, y_valid = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(10, verbose=False)])
        return roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Optimization Finished! ---")
    print(f"Best trial AUC: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  '{key}': {value},")

    os.makedirs(config.TUNING_RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(config.TUNING_RESULTS_DIR, f'best_params_lgbm_{track}_{target_type}.txt')
    with open(result_path, 'w') as f:
        f.write(f"Best trial AUC: {study.best_value}\nBest params:\n")
        for key, value in study.best_params.items():
            f.write(f"  '{key}': {value},\n")
    print(f"\n>>> SUCCESS: Best parameters saved to {result_path}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Unified model training and evaluation script.")
    parser.add_argument('--mode', type=str, required=True, choices=['eval', 'prod', 'tune'], help="Operating mode.")
    parser.add_argument('--model-type', type=str, required=True, choices=['B', 'C'], help="Model type to use (B or C).")
    parser.add_argument('--target', type=str, required=True, choices=['win', 'place'], help="Prediction target (win or place).")
    
    # Mode-specific arguments
    parser.add_argument('--track', type=str, choices=['turf', 'dirt'], help="Track type (required for all modes).")
    parser.add_argument('--evaluation-year', type=int, default=config.EVALUATION_YEAR, help="Test year for eval mode.")
    parser.add_argument('--n-trials', type=int, default=100, help="Number of trials for tune mode.")
    
    args = parser.parse_args()

    # 引数チェック
    if args.track is None:
        print("[ERROR] --track argument is required for all modes.")
        sys.exit(1)
        
    if args.mode == 'eval':
        run_evaluation(args.model_type, args.target, args.track, args.evaluation_year)
    elif args.mode == 'prod':
        run_production(args.model_type, args.target, args.track)
    elif args.mode == 'tune':
        run_tuning(args.model_type, args.target, args.track, args.n_trials)

if __name__ == "__main__":
    main()