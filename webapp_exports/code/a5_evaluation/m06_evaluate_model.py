# C:\KeibaAI\code\a5_evaluation\m06_evaluate_model.py

"""
オンライン学習シミュレーションを行い、各モデル(A, B, C)の性能を評価する統合スクリプト。

■ 機能
- コマンドライン引数で指定されたモデルタイプ(A, B, or C)のシミュレーションを実行。
- モデルA/B: 開催日単位での追加学習をシミュレート。
- モデルC: レース単位でのリアルタイム追加学習をシミュレート。
- シミュレーション完了後、AUC、単勝回収率、的中率などの性能指標を出力する。

■ 使い方
# モデルAのオンライン学習シミュレーションを実行
python code/a5_evaluation/m06_evaluate_model.py A

# モデルBのオンライン学習シミュレーションを実行
python code/a5_evaluation/m06_evaluate_model.py B
"""
import pandas as pd
import numpy as np
import os
import sys
from joblib import load
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import argparse

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..')); sys.path.append(PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
import config
from utils.evaluation_utils import calculate_return_rate

def get_params_for_eval(track: str, target: str):
    param_key = f"LGB_PARAMS_{target.upper()}_{track.upper()}"
    return getattr(config, param_key, {"objective":"binary", "metric":"auc", "verbosity":-1})

def main(model_type: str):
    print(f"===== START ONLINE LEARNING SIMULATION (MODEL {model_type}) =====")
    print(f"  - Simulation Year: {config.EVALUATION_YEAR}")
    
    encoded_base_dir = os.path.join(config.ENCODED_DIR, f'{model_type}_encoded')
    model_conf = config.MODEL_CONFIGS[model_type]
    
    for track in config.TRACK_TYPES:
        print(f"\n--- Processing for {track.upper()} track ---")
        
        try:
            encoded_path = os.path.join(encoded_base_dir, f'encoded_data_{track}.csv')
            encoded_data = pd.read_csv(encoded_path, low_memory=False, parse_dates=['日付'])
            
            raw_path = os.path.join(config.DATA_DIR, f'{config.EVALUATION_YEAR}.csv')
            raw_data_eval_year = pd.read_csv(raw_path, encoding="SHIFT-JIS", usecols=['race_id', '馬番', 'オッズ', '着順'])
            raw_data_eval_year['race_id'] = raw_data_eval_year['race_id'].astype(str)
            raw_data_eval_year['オッズ'] = pd.to_numeric(raw_data_eval_year['オッズ'], errors='coerce')

        except FileNotFoundError as e:
            print(f"[ERROR] Data file not found: {e}. Please run the appropriate m02 script first."); continue

        # --- データ準備 ---
        initial_train_data = encoded_data[encoded_data['year'] < config.EVALUATION_YEAR].copy()
        simulation_data = encoded_data[encoded_data['year'] == config.EVALUATION_YEAR].copy()
        if initial_train_data.empty or simulation_data.empty:
            print(f"[WARN] Not enough data for simulation. Skipping {track}."); continue

        y_initial_train = (initial_train_data['着順'] == 1).astype(int)
        
        # --- 特徴量選択 ---
        base_drop_cols = ['race_id', 'horse_id', '騎手', '馬', '日付', 'レース名', '開催', 'year', '着順', '通過順']

        # 過去の日付カラムも削除対象に明示的に追加する
        past_date_cols = [f'日付{i}' for i in range(1, config.NUM_PAST_RACES + 2)]
        base_drop_cols.extend(past_date_cols)

        leakage_cols = model_conf.get('leakage_features', []) # リーク情報
        drop_conf = model_conf.get('features_to_drop', {}) # モデルごとの不要特徴量
        cols_to_drop = list(set(base_drop_cols + leakage_cols + drop_conf.get('common', []) + drop_conf.get(track, [])))

        X_initial_train = initial_train_data.drop(columns=[c for c in cols_to_drop if c in initial_train_data.columns])
        training_features = X_initial_train.columns.tolist()

        # --- 初期モデル学習 ---
        print("[INFO] Training initial model...")
        imputer = SimpleImputer(strategy='mean')
        X_initial_train_imputed = pd.DataFrame(imputer.fit_transform(X_initial_train), columns=training_features)
        
        lgb_params = get_params_for_eval(track, 'win')
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        categorical_feature_names = [c for c in model_conf['categorical_features'] if c in training_features]
        lgb_model.fit(X_initial_train_imputed, y_initial_train, categorical_feature=categorical_feature_names)
        
        # --- シミュレーション実行 ---
        simulation_data.sort_values(by=['日付', 'race_id'], inplace=True)
        # モデルCはレース単位、A/Bは日付単位でループ
        group_key = ['日付', 'race_id'] if model_type == 'C' else '日付'
        
        all_predictions = []
        total_groups = len(simulation_data.groupby(group_key))
        print(f"[INFO] Starting simulation for {total_groups} groups...")
        
        for i, (group_id, group_data) in enumerate(simulation_data.groupby(group_key)):
            print(f"\r  -> Processing group {i+1}/{total_groups}", end="", flush=True)

            X_group = group_data[training_features]
            X_group_imputed = pd.DataFrame(imputer.transform(X_group), columns=training_features)
            
            group_scores = lgb_model.predict_proba(X_group_imputed)[:, 1]
            
            group_pred_df = group_data[['race_id', '馬番', '着順']].copy()
            group_pred_df['score_win'] = group_scores
            all_predictions.append(group_pred_df)

            # モデルCとA/Bで学習単位が異なる
            y_group_update = (group_data['着順'] == 1).astype(int)
            lgb_model.fit(X_group_imputed, y_group_update, init_model=lgb_model.booster_, categorical_feature=categorical_feature_names)
        
        print("\n[INFO] Simulation loop finished.")
        if not all_predictions:
            print("[WARN] No predictions made. Skipping evaluation."); continue
            
        final_predictions_df = pd.concat(all_predictions, ignore_index=True)
        final_predictions_df['race_id'] = final_predictions_df['race_id'].astype(str)

        # --- 評価指標の計算 ---
        print(f"\n--- Online Learning Simulation Results for Model {model_type} ({track.upper()}) ---")
        final_auc = roc_auc_score((final_predictions_df['着順'] == 1).astype(int), final_predictions_df['score_win'])
        print(f"  - Overall AUC: {final_auc:.4f}")
        
        final_predictions_df['race_id'] = final_predictions_df['race_id'].astype(str)
        raw_data_eval_year['race_id'] = raw_data_eval_year['race_id'].astype(str)
        final_predictions_df['馬番'] = final_predictions_df['馬番'].astype(int)
        raw_data_eval_year['馬番'] = raw_data_eval_year['馬番'].astype(int)
        
        # 予測スコア1位の馬に単勝賭け
        pred_rank_1_df = final_predictions_df.loc[final_predictions_df.groupby('race_id')['score_win'].idxmax()]
        return_rate_rank, win_rate_rank, _ = calculate_return_rate(pred_rank_1_df, raw_data_eval_year)
        
        # 期待値1位の馬に単勝賭け
        merged_preds_for_ev = pd.merge(final_predictions_df, raw_data_eval_year[['race_id', '馬番', 'オッズ']], on=['race_id', '馬番'], how='left')
        merged_preds_for_ev['expected_value'] = merged_preds_for_ev['score_win'] * merged_preds_for_ev['オッズ']
        ev_rank_1_df = merged_preds_for_ev.loc[merged_preds_for_ev.groupby('race_id')['expected_value'].idxmax()]
        return_rate_ev, win_rate_ev, _ = calculate_return_rate(ev_rank_1_df, raw_data_eval_year)
        
        print("\n  [Strategy 1: Bet on Top Predicted Horse]")
        print(f"  - Return Rate: {return_rate_rank:.2f}%")
        print(f"  - Win Rate (Accuracy): {win_rate_rank:.2f}%")
        
        print("\n  [Strategy 2: Bet on Top Expected Value Horse]")
        print(f"  - Return Rate: {return_rate_ev:.2f}%")
        print(f"  - Win Rate (Accuracy): {win_rate_ev:.2f}%")

    print("\n===== ALL SIMULATION PROCESSES FINISHED. =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online learning simulation for a specified model.")
    parser.add_argument('model_type', type=str, choices=['A', 'B', 'C'], help="Model type to evaluate (A, B, or C).")
    args = parser.parse_args()
    main(args.model_type)