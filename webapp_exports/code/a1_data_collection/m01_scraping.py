"""
netkeiba.comのデータベースから、指定された期間の全レース結果をスクレイピングし、
年ごとのCSVファイルとして '../data' ディレクトリに保存するスクリプト。
または、特定のレースIDベース（YYYYPPCCDD）に紐づく1Rから12Rまでのレース結果のみをスクレイピングし、既存の年ごとのCSVファイルに追記する機能も持つ。

■ 主な処理
- **引数なしの場合**: config.pyで定義されたYEAR_STARTからYEAR_END-1までの全期間のデータを収集。
- **レースIDベースを引数に与えた場合**: そのIDベースに紐づく1R〜12Rのレースデータを収集し、年ごとのCSVに追記。

■ 使い方
- 全期間をスクレイピング: `python code/a1_data_collection/m01_scraping.py`
- 特定のレースIDベースをスクレイピング（例: 2025年9月7日の中山開催IDベース `2025060403`）:
  `python code/a1_data_collection/m01_scraping.py 2025060403`

■ 注意事項
- netkeibaのサーバーに過度な負荷をかけないよう、リクエスト間には適切な待機時間 (REQUEST_WAIT_TIME) を設けています。
"""

import requests
from bs4 import BeautifulSoup
import time
import csv
import os
import re
from datetime import datetime, timedelta
import sys
from typing import Optional, List
import pandas as pd

# --- プロジェクトルートをパスに追加 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
# ---

import config # configモジュールをインポート (修正点)
from utils.feature_pipeline import class_mapping # feature_pipelineからclass_mappingをインポート (修正点)

# --- SETTINGS ---
YEAR_START = 2025
YEAR_END = 2026 
SAVE_DIR = os.path.join(PROJECT_ROOT, 'data') 
REQUEST_WAIT_TIME = 1.0 
RETRY_WAIT_TIME = 5     

CSV_HEADER = [
    'race_id', '馬', 'horse_id', '騎手', 'jockey_id', '馬番', '走破時間', 'オッズ', 
    '通過順', '着順', '体重', '体重変化', '性', '齢', '斤量', '上がり', '人気', 
    'レース名', '日付', '開催', 'クラス', '芝・ダート', '距離', '回り', '馬場', '天気', 
    '場id', '場名'
]

# PLACE_MAP は config.py に移動したため、このファイルからは削除 (修正点)
# PLACE_MAP = {
#     "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
#     "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
# }
# --- END SETTINGS ---


def safe_get_text(element, strip=True):
    """BeautifulSoupの要素から安全にテキストを取得するヘルパー関数"""
    return element.get_text(strip=True) if element else ""

def _scrape_race_data_from_url(current_race_id: str, url: str, place_id: str, place_name: str) -> Optional[list]:
    """
    単一レースIDのページからデータをスクレイピングする内部ヘルパー関数
    成功すればレースデータ行のリスト（複数馬分）を返し、失敗すればNoneを返す。
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        time.sleep(REQUEST_WAIT_TIME)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None # レースが見つからない場合はNoneを返す
        else:
            print(f"\n[ERROR] Request failed for {current_race_id}: {e}. Retrying after {RETRY_WAIT_TIME} seconds.")
            time.sleep(RETRY_WAIT_TIME)
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
            except requests.exceptions.RequestException as e2:
                print(f"\n[FATAL] Retry failed for {current_race_id}. Skipping race: {e2}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed for {current_race_id}: {e}. Retrying after {RETRY_WAIT_TIME} seconds.")
        time.sleep(RETRY_WAIT_TIME)
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
        except requests.exceptions.RequestException as e2:
            print(f"\n[FATAL] Retry failed for {current_race_id}. Skipping race: {e2}")
            return None
    
    soup = BeautifulSoup(r.content.decode("euc-jp", "ignore"), "html.parser")

    race_table = soup.find("table", class_="race_table_01")
    if not race_table:
        return None
    
    # レース情報抽出
    title, date_str_from_page, detail, clas = ('', '', '', '')
    sur, rou, dis, wed, con = ('', '', '', '', '')

    data_intro = soup.find("div", class_="data_intro")
    if data_intro:
        title = safe_get_text(data_intro.find("h1"))
        smalltxt = safe_get_text(data_intro.find("p", class_="smalltxt")).split()
        date_str_from_page = smalltxt[0] if smalltxt else ""
        detail = smalltxt[1] if len(smalltxt) > 1 else ""
        clas = smalltxt[2].replace('\xa0', ' ') if len(smalltxt) > 2 else ""

        diary_snap_text = ""
        all_spans = data_intro.find_all("span")
        for span in all_spans:
            text = safe_get_text(span)
            if "m" in text and "天候" in text:
                diary_snap_text = text.replace('\xa0', ' ')
                break
        
        if diary_snap_text:
            parts = diary_snap_text.split('/')
            if len(parts) >= 3:
                sur = parts[0].strip()[0] # 芝・ダート
                rou = parts[0].strip()[1] # 回り
                dis = ''.join(filter(str.isdigit, parts[0])) # 距離
                wed = parts[1].split(':')[1].strip() if ':' in parts[1] else '' # 天候
                con = parts[2].split(':')[1].strip() if ':' in parts[2] else '' # 馬場

    # 各馬のデータ抽出
    horse_data_rows = []
    horse_rows = race_table.find_all("tr")[1:]
    for row_idx, row in enumerate(horse_rows): # row_idx に修正
        cells = row.find_all("td")
        if len(cells) < 15: continue

        try:
            rank = safe_get_text(cells[0])
            umaban = safe_get_text(cells[2])
            horse_link = cells[3].find("a")
            horse_name = safe_get_text(horse_link)
            horse_id = horse_link['href'].split('/')[-2] if horse_link else ""
            sex_age = safe_get_text(cells[4])
            sex, age = (sex_age[0], sex_age[1:]) if sex_age else ('', '')
            kinryo = safe_get_text(cells[5])
            jockey_link = cells[6].find("a")
            jockey_name = safe_get_text(jockey_link)
            jockey_id_match = re.search(r'/jockey/result/recent/(\d+)/', jockey_link['href']) if jockey_link else None
            jockey_id = jockey_id_match.group(1) if jockey_id_match else ""
            runtime = safe_get_text(cells[7])
            pas = safe_get_text(cells[10])
            last = safe_get_text(cells[11])
            odds = safe_get_text(cells[12])
            pop = safe_get_text(cells[13])
            weight_text = safe_get_text(cells[14])
            weight, weight_dif = weight_text.replace(')', '').split('(') if '(' in weight_text else (weight_text, '0')

            race_data_row = [
                current_race_id, horse_name, horse_id, jockey_name, jockey_id, umaban, runtime, odds, pas, rank,
                weight, weight_dif, sex, age, kinryo, last, pop, title, date_str_from_page, detail, clas,
                sur, dis, rou, con, wed, place_id, place_name
            ]
            horse_data_rows.append(race_data_row)

        except Exception as e:
            print(f"\n[ERROR] Error parsing horse data for {current_race_id}: {e}. Skipping this horse (row {row_idx}).")
            return None # 1頭でもエラーが出たらそのレースはスキップ

    return horse_data_rows


def run_single_race_id_base_scraping(race_id_base: str):
    """
    指定されたrace_id_base (例: 2025060403) に紐づく1Rから12Rまでの全レースデータをスクレイピングする。
    """
    if not re.fullmatch(r'\d{10}', race_id_base):
        print(f"[ERROR] Invalid race_id_base format: {race_id_base}. Expected YYYYPPCCDD (10 digits).")
        return

    year = int(race_id_base[0:4])
    place_id = race_id_base[4:6]
    kai = int(race_id_base[6:8]) # 開催回
    nichi = int(race_id_base[8:10]) # 開催日

    if place_id not in config.PLACE_MAP: # config.PLACE_MAP を使用 (修正点)
        print(f"[ERROR] Unknown place_id: {place_id} in {race_id_base}.")
        return
    place_name = config.PLACE_MAP[place_id] # config.PLACE_MAP を使用
    
    output_path = os.path.join(SAVE_DIR, f'{year}.csv')
    file_exists = os.path.exists(output_path)
    
    existing_race_ids = set()
    if file_exists:
        try:
            existing_df = pd.read_csv(output_path, encoding="SHIFT-JIS", header=0, low_memory=False, usecols=['race_id'])
            existing_race_ids = set(existing_df['race_id'].astype(str).tolist())
            print(f"[INFO] Loaded {len(existing_race_ids)} existing race_ids from {output_path} for duplicate check.")
        except Exception as e:
            print(f"[WARN] Failed to load existing data from {output_path} for duplicate check: {e}. Proceeding without strict duplicate check for existing data.")

    print(f"--- Processing races for {year} at {place_name} (開催{kai}回{nichi}日目) ---")
    race_data_all_for_base_id = []

    day_has_no_race_counter = 0

    for race_num in range(1, 13): # レース番号 (1Rから12R)
        current_race_id = f"{race_id_base}{race_num:02d}"
        url = f"https://db.netkeiba.com/race/{current_race_id}"

        if current_race_id in existing_race_ids:
            print(f"\r[INFO] Race {race_num:02d} (ID: {current_race_id}) already exists. Skipping.                                   ", end="", flush=True) # スペースで埋める
            day_has_no_race_counter = 0 
            continue

        scraped_rows = _scrape_race_data_from_url(current_race_id, url, place_id, place_name)
        
        if scraped_rows is None: 
            day_has_no_race_counter += 1
            if day_has_no_race_counter >= 3: 
                print(f"\r[INFO] No more races found after {race_num-1}R for {race_id_base}. Breaking loop.                          ", end="", flush=True) # スペースで埋める
                break 
            continue
        else:
            day_has_no_race_counter = 0 
            race_data_all_for_base_id.extend(scraped_rows)

        print(f"\r[INFO] Scraped {race_num:02d}R for {year}/{place_id}/{kai}回{nichi}日目. Total horses collected: {len(race_data_all_for_base_id)}.", end="", flush=True)


    if race_data_all_for_base_id:
        with open(output_path, 'a' if file_exists else 'w', newline='', encoding="SHIFT-JIS", errors='replace') as f:
            writer = csv.writer(f)
            if not file_exists: 
                writer.writerow(CSV_HEADER)
            writer.writerows(race_data_all_for_base_id)
        print(f"\n--- Appended {len(race_data_all_for_base_id)} records for {race_id_base} to {output_path} ---\n")
    else:
        print(f"\n--- No races found for {race_id_base}. No data appended. ---\n")

    print(f"Scraping for {race_id_base} finished.")


def run_full_period_scraping(start_year: int, end_year: int):
    """YEAR_STARTからYEAR_END-1までの全期間のレース結果をスクレイピングする"""
    for year_to_scrape in range(start_year, end_year):
        print(f"--- Processing data for year: {year_to_scrape} ---")
        race_data_all_for_year = [] 
        
        output_path = os.path.join(SAVE_DIR, f'{year_to_scrape}.csv')
        file_exists = os.path.exists(output_path)
        existing_race_ids = set()
        if file_exists:
            try:
                existing_df = pd.read_csv(output_path, encoding="SHIFT-JIS", header=0, low_memory=False, usecols=['race_id'])
                existing_race_ids = set(existing_df['race_id'].astype(str).tolist())
                print(f"[INFO] Loaded {len(existing_race_ids)} existing race_ids from {output_path} for duplicate check.")
            except Exception as e:
                print(f"[WARN] Failed to load existing data from {output_path} for duplicate check: {e}. Proceeding without strict duplicate check.")

        for place_id, place_name in config.PLACE_MAP.items(): # config.PLACE_MAP を使用 (修正点)
            print(f"== Started processing for {place_name} racetrack (Year: {year_to_scrape}) ==")
            for kai in range(1, 7): # 開催回
                for nichi in range(1, 13): # 開催日
                    day_has_no_race_counter = 0 # 連続404カウンター

                    for race_num in range(1, 13): # レース番号
                        current_race_id = f"{year_to_scrape}{place_id}{kai:02d}{nichi:02d}{race_num:02d}"
                        url = f"https://db.netkeiba.com/race/{current_race_id}"

                        if current_race_id in existing_race_ids:
                            day_has_no_race_counter = 0 
                            continue
                        
                        scraped_rows = _scrape_race_data_from_url(current_race_id, url, place_id, place_name)
                        
                        if scraped_rows is None: 
                            day_has_no_race_counter += 1
                            if day_has_no_race_counter >= 3: 
                                break
                            continue
                        else:
                            day_has_no_race_counter = 0 
                            race_data_all_for_year.extend(scraped_rows)
                            print(f"\r[INFO] Processed {race_num:02d}R for {current_race_id[:-2]}. Total collected: {len(race_data_all_for_year)} horses.", end="", flush=True)

                    if day_has_no_race_counter >= 3: 
                        pass 
                
        if race_data_all_for_year:
            output_path = os.path.join(SAVE_DIR, f'{year_to_scrape}.csv')
            print(f"\n--- Writing {len(race_data_all_for_year)} new records for year {year_to_scrape} to {output_path} ---\n")
            with open(output_path, 'a' if file_exists else 'w', newline='', encoding="SHIFT-JIS", errors='replace') as f:
                writer = csv.writer(f)
                if not file_exists: 
                    writer.writerow(CSV_HEADER)
                writer.writerows(race_data_all_for_year)
        else:
            print(f"\n--- No new data collected for year {year_to_scrape}. ---\n")

    print("All full period scraping processes finished.")


def main():
    """スクリプトのエントリーポイント。コマンドライン引数で race_id_base を受け取る。"""
    os.makedirs(SAVE_DIR, exist_ok=True) 
    
    if len(sys.argv) > 1:
        race_id_base_input = sys.argv[1]
        if re.fullmatch(r'\d{10}', race_id_base_input):
            print(f"--- Running specific race_id_base scraping: {race_id_base_input} ---")
            run_single_race_id_base_scraping(race_id_base_input)
        else:
            print(f"[ERROR] Invalid argument. Please provide a 10-digit race_id_base (YYYYPPCCDD) or no argument for full scraping.")
            sys.exit(1)
    else:
        print("--- Running full period scraping ---")
        run_full_period_scraping(YEAR_START, YEAR_END)

if __name__ == "__main__":
    main()