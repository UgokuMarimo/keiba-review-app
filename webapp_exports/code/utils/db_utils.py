import sqlite3
import pandas as pd
import requests
import json
import os
import sys

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..')); sys.path.append(PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
import config

def save_prediction_to_db(result_df: pd.DataFrame, shutuba_df: pd.DataFrame, race_id: str):
    """予測結果をSQLiteデータベースに保存する (新DB設計対応版)"""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS predictions (
                race_id TEXT, umaban INTEGER, horse_name TEXT, kaisai_date TEXT, 
                keibajo TEXT, race_number INTEGER, track_type TEXT, 
                pred_win REAL, pred_rank INTEGER, 
                tansho_odds REAL, tansho_ninki INTEGER, 
                result_rank INTEGER,  -- 結果更新用に残す
                prediction_timestamp TEXT, 
                PRIMARY KEY (race_id, umaban)
            );"""
            conn.execute(create_table_query)

            save_target_df = shutuba_df[['馬番', 'オッズ', '人気']].copy()
            save_target_df.rename(columns={'オッズ': '単勝オッズ'}, inplace=True)
            save_target_df['馬番'] = pd.to_numeric(save_target_df['馬番'], errors='coerce')
            save_df = pd.merge(result_df, save_target_df, on='馬番', how='left')
            
            race_info = shutuba_df.iloc[0]
            save_df['race_id'] = race_id
            save_df['kaisai_date'] = pd.to_datetime(race_info['日付'], format='%Y年%m月%d日', errors='coerce').strftime('%Y-%m-%d')
            save_df['keibajo'] = race_info['場名']
            save_df['race_number'] = int(str(race_id)[-2:])
            save_df['track_type'] = 'turf' if '芝' in race_info['芝・ダート'] else 'dirt'
            save_df['prediction_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            save_df.rename(columns={
                '馬名': 'horse_name', '馬番': 'umaban', 
                'pred_win': 'pred_win', 'rank_win': 'pred_rank', 
                '単勝オッズ': 'tansho_odds', '人気': 'tansho_ninki'
            }, inplace=True)

            final_cols = ['race_id', 'umaban', 'horse_name', 'kaisai_date', 'keibajo', 'race_number', 'track_type', 'pred_win', 'pred_rank', 'tansho_odds', 'tansho_ninki', 'prediction_timestamp']
            final_save_df = save_df[[col for col in final_cols if col in save_df.columns]]
            
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE race_id = ?", (race_id,))
            final_save_df.to_sql('predictions', conn, if_exists='append', index=False)
            conn.commit()
            print(f"-> Prediction for race_id {race_id} saved to clean 'predictions' table successfully.")

    except Exception as e:
        print(f"[DB ERROR] Failed to save prediction to database: {e}")

def send_discord_webhook(message: str):
    if not hasattr(config, 'DISCORD_WEBHOOK_URL') or not config.DISCORD_WEBHOOK_URL: return
    try:
        requests.post(config.DISCORD_WEBHOOK_URL, json={"content": message, "username": "競馬AI予測"})
        print("-> Message sent to Discord successfully.")
    except requests.exceptions.RequestException as e: print(f"[DISCORD ERROR]: {e}")

def format_for_discord(race_id, race_info, result_df):
    race_name = race_info.get('レース名', '不明'); venue = race_info.get('場名', '不明')
    race_number = str(race_id)[-2:].lstrip('0')
    header = f"🐴 **{venue}{race_number}R {race_name} AI予測** 🐴\n" + "="*30 + "\n"
    top5_df = result_df.head(5)
    body = "```\n" + "{:^4} {:^4} {:<12} {:^8}\n".format("順位", "馬番", "馬名", "予測値") + "-"*32 + "\n" # ヘッダーを変更
    for _, row in top5_df.iterrows():
        pred_win_val = row.get('pred_win', 0)
        # ★★★ 修正開始: %表示から小数点表示に変更 ★★★
        body += "{:^4} {:^4} {:<12s} {:>7.4f}\n".format(row['rank_win'], str(int(row['馬番'])), row['馬名'][:11], pred_win_val)
        # ★★★ 修正終了 ★★★
    body += "```"
    return header + body