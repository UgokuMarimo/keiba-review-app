import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import time

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))

# --- モジュールインポート ---
from a4_prediction.m04_predict import load_models, predict_race
from utils.schedule_scraper import get_race_schedule_for_date

def batch_predict(target_date: str = None, race_ids: list = None, model_type: str = 'B', run_shap: bool = True, use_overseas: bool = False, enable_explanation: bool = False):
    """
    複数レースを一括予測する関数。
    """
    print(f"--- [START] Batch Prediction (Model: {model_type}) ---")
    
    # 1. 対象レースIDの特定
    target_race_ids = []
    if race_ids:
        target_race_ids = race_ids
    elif target_date:
        print(f"Fetching race schedule for date: {target_date}")
        try:
            schedule_df = get_race_schedule_for_date(target_date)
            if schedule_df is not None and not schedule_df.empty:
                target_race_ids = schedule_df['race_id'].tolist()
            else:
                print(f"[WARN] No races found for date: {target_date}")
                return
        except Exception as e:
            print(f"[ERROR] Failed to fetch schedule: {e}")
            return
    else:
        print("[ERROR] Either target_date or race_ids must be provided.")
        return

    if not target_race_ids:
        print("[WARN] No race IDs to process.")
        return

    print(f"Target Races: {len(target_race_ids)} races")
    print(f"IDs: {target_race_ids}")

    # 2. モデルのロード (1回だけ)
    try:
        models, artifacts, model_conf = load_models(model_type)
    except Exception as e:
        print(f"[FATAL] Failed to load models: {e}")
        return

    # 3. 各レース予測実行
    success_count = 0
    fail_count = 0
    
    for race_id in tqdm(target_race_ids, desc="Predicting Races"):
        print(f"\nProcessing Race ID: {race_id}...")
        try:
            result = predict_race(
                race_id=str(race_id),
                model_type=model_type,
                run_shap=run_shap,
                use_overseas=use_overseas,
                models=models,
                artifacts=artifacts,
                model_conf=model_conf,
                enable_explanation=enable_explanation
            )
            
            if result is not None:
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"[ERROR] Failed to predict race {race_id}: {e}")
            fail_count += 1
            # Continue to next race
            
    print(f"\n--- [BATCH COMPLETE] Success: {success_count}, Failed: {fail_count} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict horse races.")
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)')
    parser.add_argument('--race_ids', help='Comma-separated list of race IDs')
    parser.add_argument('--model_type', default='B', help="Model type (default: B)")
    parser.add_argument('--no-shap', action='store_false', dest='run_shap', help="Disable SHAP analysis.")
    parser.add_argument('--use-overseas', action='store_true', dest='use_overseas', help="Include overseas data.")
    parser.add_argument('--explanation', action='store_true', dest='enable_explanation', help="Enable automated LLM explanation.")
    parser.set_defaults(run_shap=True, use_overseas=False, enable_explanation=False)
    
    args = parser.parse_args()
    
    race_id_list = None
    if args.race_ids:
        race_id_list = args.race_ids.split(',')
        
    batch_predict(
        target_date=args.date,
        race_ids=race_id_list,
        model_type=args.model_type,
        run_shap=args.run_shap,
        use_overseas=args.use_overseas,
        enable_explanation=args.enable_explanation
    )
