import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import re
import os
import sys
from typing import Optional, List, Dict
from tqdm import tqdm

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
# ---

# configモジュールは必要に応じてインポート
import config

REQUEST_WAIT_TIME = 0.3 # サーバー負荷軽減のための待機時間

# JRAの競馬場IDマッピング (m01_scraping.pyやconfig.pyから流用)
# ここで使うのは場名からJRAかどうかを判定するため
JRA_PLACE_NAMES = set([
    "札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"
])

def safe_get_text(element, strip=True):
    """BeautifulSoupの要素から安全にテキストを取得するヘルパー関数"""
    return element.get_text(strip=True) if element else ""

def to_numeric_or_nan(value):
    """文字列を数値に変換、変換不能ならNaN"""
    if value is None or str(value).strip() in ['', '**', '--', '---.-', '計不', '&nbsp;']:
        return pd.NA # pandasの欠損値型を使用
    try:
        cleaned_value = re.sub(r'[^\d.-]', '', str(value))
        if cleaned_value in ['.', '-', '']: return pd.NA
        return float(cleaned_value)
    except (ValueError, TypeError): return pd.NA


def scrape_shutuba_table_for_horse_urls(race_id: str) -> Optional[pd.DataFrame]:
    """
    指定されたレースIDの出馬表から、各馬のhorse_idと過去走ページURLを抽出する。
    """
    print(f"[SCRAPER] Fetching shutuba table for horse URLs for race_id: {race_id}")
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        r.raise_for_status()
        time.sleep(REQUEST_WAIT_TIME)
    except requests.exceptions.RequestException as e:
        print(f"\n[SCRAPER ERROR] Failed to fetch shutuba table: {e}"); return None
    
    soup = BeautifulSoup(r.content, "html.parser", from_encoding=r.apparent_encoding)
    table = soup.find("table", class_="Shutuba_Table")
    
    if not table:
        print(f"[SCRAPER WARN] Shutuba table not found for race_id: {race_id}"); return None
        
    horse_info_list = []
    for row in table.find_all("tr", class_="HorseList"):
        cols = row.find_all("td")
        if not cols or len(cols) < 4: continue # 馬名リンクを含む4列目まで必要
        
        horse_link_element = cols[3].find("a")
        if not horse_link_element or 'href' not in horse_link_element.attrs: continue
        
        horse_url_match = re.search(r'/horse/(\d+)', horse_link_element['href'])
        if horse_url_match:
            horse_id = horse_url_match.group(1)
            # 馬の個別プロフィールページではなく、直接「戦績」タブのURLを生成
            horse_past_race_url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
            horse_name = safe_get_text(horse_link_element)
            
            horse_info_list.append({
                'horse_id': horse_id,
                'horse_name': horse_name,
                'horse_past_race_url': horse_past_race_url
            })
            
    return pd.DataFrame(horse_info_list) if horse_info_list else None


def scrape_all_past_races_from_horse_page(
    horse_id: str, 
    horse_url: str, 
    max_races: int = 10
) -> Optional[pd.DataFrame]:
    """
    馬の個別ページ（戦績タブ）から過去のレース結果（JRA, 地方, 海外問わず）をスクレイピングする。
    最低限の基本情報を取得し、CSVとして出力できるよう整形する。
    """
    if not horse_url: return None
    
    records = []
    try:
        r = requests.get(horse_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        r.raise_for_status()
        time.sleep(REQUEST_WAIT_TIME)
    except requests.exceptions.RequestException as e:
        print(f"[SCRAPER ERROR] Failed to scrape horse page {horse_url}: {e}"); return None
    
    soup = BeautifulSoup(r.content, "html.parser", from_encoding=r.apparent_encoding)
    
    # ★★★ 修正開始 ★★★
    # 提供されたHTMLソースに合わせてセレクタを修正
    race_results_table = soup.find("table", class_="db_h_race_results nk_tb_common")
    # ★★★ 修正終了 ★★★

    if not race_results_table:
        # print(f"[SCRAPER WARN] Past race results table not found for {horse_url}");
        return None
        
    rows = race_results_table.find_all("tr")[1:] # ヘッダー行を除く
    
    for row_idx, row in enumerate(rows):
        if row_idx >= max_races: break # 指定された件数だけ取得
        
        cols = row.find_all("td")
        # 列数が足りない場合はスキップ（ヘッダーから数えて25列あるため、少し余裕を持って）
        if len(cols) < 20: continue 

        try:
            # 各カラムを抽出（提供されたHTMLの<th>タグの順序に合わせてインデックスを調整）
            
            # 日付 (0)
            date_str_raw = safe_get_text(cols[0])
            # 日付を標準形式に変換 (YYYY/MM/DD または YYYY-MM-DD → YYYY年MM月DD日)
            try:
                parsed_date = pd.to_datetime(date_str_raw, errors='coerce')
                if pd.notna(parsed_date):
                    date_str = parsed_date.strftime('%Y年%m月%d日')
                else:
                    date_str = date_str_raw  # 変換できない場合はそのまま
            except:
                date_str = date_str_raw  # エラーの場合はそのまま
            
            # 開催 (場名) (1) 例: '5東京3' や '大井'
            kaisai_text = safe_get_text(cols[1])
            # レース名 (4)
            race_name = safe_get_text(cols[4])
            
            # 頭数 (6)
            num_horses = safe_get_text(cols[6])
            # 馬番 (8)
            umaban = safe_get_text(cols[8])
            # オッズ (9)
            odds = safe_get_text(cols[9])
            # 人気 (10)
            pop = safe_get_text(cols[10])
            # 着順 (11)
            rank_str = safe_get_text(cols[11])
            
            # 騎手 (12)
            jockey_name = safe_get_text(cols[12])
            jockey_link = cols[12].find("a")
            jockey_id_match = re.search(r'/jockey/.*?/(\d+)', jockey_link['href']) if jockey_link else None
            jockey_id = jockey_id_match.group(1) if jockey_id_match else ''
            
            # 斤量 (13)
            kinryo = safe_get_text(cols[13])
            
            # 距離 (14) 例: 'ダ1600'
            distance_full_str = safe_get_text(cols[14])
            track_type_match = re.search(r'(芝|ダ|障)', distance_full_str)
            track_type_char = track_type_match.group(1) if track_type_match else ''
            distance_num_match = re.search(r'(\d+)m', distance_full_str)
            distance = distance_num_match.group(1) if distance_num_match else ''
            
            # 馬場 (16)
            track_condition = safe_get_text(cols[16])
            
            # タイム (18)
            runtime = safe_get_text(cols[18])
            
            # 着差 (19)
            margin = safe_get_text(cols[19])
            
            # 通過 (21)
            passing_order = safe_get_text(cols[21])
            
            # ペース (22)
            pace = safe_get_text(cols[22])
            
            # 上り (23)
            agari = safe_get_text(cols[23])
            
            # 馬体重 (24) 例: '514(-1)'
            weight_text = safe_get_text(cols[24])
            weight, weight_dif = (weight_text.replace(')', '').split('(')) if '(' in weight_text else (weight_text, pd.NA)

            # 天気はHTMLソースに直接ない場合があるので、取得を試みる
            weather = safe_get_text(cols[2]) # cols[2]が天気に該当

            # race_id (レース名リンクから抽出)
            race_link_element = cols[4].find('a') 
            if race_link_element and 'href' in race_link_element.attrs:
                race_id_match = re.search(r'/race/(\d+)', race_link_element['href'])
                extracted_race_id = race_id_match.group(1) if race_id_match else None
            else:
                extracted_race_id = None
            
            # 場名の抽出 (例: '5東京3' -> '東京')
            place_name_from_kaisai = ''
            kaisai_place_match = re.search(r'[^\d]+', kaisai_text) # 数字以外の部分を抽出
            if kaisai_place_match:
                place_name_from_kaisai = kaisai_place_match.group(0).replace('東京', '東京').replace('京都', '京都').replace('福島', '福島').strip() # 開催の頭の数字は含まれないように
                # 例：'大井' はそのまま
            
            # JRA, 地方, 海外の判定
            # race_idがあれば、そのprefixで判断するのが最も確実
            # なければ場名で簡易判定
            is_jra = False
            if extracted_race_id and len(extracted_race_id) >= 6:
                # JRAのrace_idはYYYYPPCCDDSSなので、PPの部分 (4-6桁目) で判断
                place_id_prefix = extracted_race_id[4:6]
                if place_id_prefix in config.PLACE_MAP_IDS: # config.pyにJRAのplace_idマップを定義済みと仮定
                    is_jra = True
            elif place_name_from_kaisai: # race_idから判断できない場合、場名で判定
                if place_name_from_kaisai in JRA_PLACE_NAMES: # 新規定義のJRA場名セットを使用
                    is_jra = True
            
            # 性齢は個別ページでは提供されないことが多いので、今回は取得しない
            sex = pd.NA; age = pd.NA
            
            record = {
                'horse_id': horse_id,
                'race_id': extracted_race_id,
                '日付': date_str,
                'レース名': race_name,
                '開催': kaisai_text, # 開催情報全体を保存
                '場名': place_name_from_kaisai,
                'num_horses': to_numeric_or_nan(num_horses),
                '馬番': to_numeric_or_nan(umaban),
                '着順': to_numeric_or_nan(rank_str),
                '距離': to_numeric_or_nan(distance),
                '芝・ダート': track_type_char,
                '天気': weather,
                '馬場': track_condition,
                '走破時間': runtime, # stringのまま
                '上がり': to_numeric_or_nan(agari),
                '斤量': to_numeric_or_nan(kinryo),
                '騎手': jockey_name,
                'jockey_id': jockey_id,
                'オッズ': to_numeric_or_nan(odds),
                '人気': to_numeric_or_nan(pop),
                '通過順': passing_order, # stringのまま
                '馬体重': to_numeric_or_nan(weight),
                '体重変化': to_numeric_or_nan(weight_dif),
                '着差': to_numeric_or_nan(margin),
                'ペース': pace, # stringのまま
                '性': sex,
                '齢': age,
                'is_jra_race': is_jra,
                'データソース': 'individual_page_result' # 出典を明記
            }
            records.append(record)

        except Exception as e:
            print(f"[SCRAPER WARN] Error parsing past race for {horse_url} (row {row_idx}): {e}. Skipping this race.")
            # 詳細なデバッグのため、traceback.print_exc() を一時的に有効にしても良い
            # import traceback; traceback.print_exc()
            continue
            
    return pd.DataFrame(records) if records else None


def main_scrape_horse_past_races(target_race_id: str, output_dir: str = os.path.join(PROJECT_ROOT, 'data', 'temp_horse_past_races')):
    """
    指定されたレースIDの出馬表から各馬の過去走をスクレイピングし、CSVとして出力するメイン関数。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 出馬表からhorse_idと個別ページURLを取得
    shutuba_info_df = scrape_shutuba_table_for_horse_urls(target_race_id)
    if shutuba_info_df is None or shutuba_info_df.empty:
        print(f"[FATAL] No horse info found from shutuba table for race_id: {target_race_id}. Exiting.")
        return

    all_horses_past_races = []
    
    print(f"\n--- [START] Scraping past races for {len(shutuba_info_df)} horses ---")
    for index, row in tqdm(shutuba_info_df.iterrows(), total=len(shutuba_info_df), desc="Scraping each horse's past races"):
        horse_id = row['horse_id']
        horse_name = row['horse_name']
        horse_url = row['horse_past_race_url']
        
        # 2. 各馬の個別ページから過去走をスクレイピング
        past_races_df = scrape_all_past_races_from_horse_page(horse_id, horse_url, max_races=config.NUM_PAST_RACES_TO_SCRAPE)
        
        if past_races_df is not None and not past_races_df.empty:
            all_horses_past_races.append(past_races_df)
        else:
            print(f"[WARN] No past race data found for {horse_name} (ID: {horse_id}) from individual page.")

    if not all_horses_past_races:
        print(f"[FATAL] No past race data collected for any horse in race {target_race_id}. Exiting.")
        return
        
    final_combined_df = pd.concat(all_horses_past_races, ignore_index=True)
    
    # 日付とレース名、着順、horse_idで重複を排除
    # race_idが簡易的なものの場合も考慮し、horse_id, 日付, レース名, 着順でdrop_duplicates
    final_combined_df.drop_duplicates(subset=['horse_id', '日付', 'レース名', '着順'], inplace=True)
    
    # 日付でソート（一時的にdatetime型に変換してソート、その後元の文字列形式を保持）
    final_combined_df['日付_temp'] = pd.to_datetime(final_combined_df['日付'], format='%Y年%m月%d日', errors='coerce')
    final_combined_df.sort_values(by=['horse_id', '日付_temp'], ascending=[True, False], inplace=True)
    final_combined_df.drop('日付_temp', axis=1, inplace=True)  # ソート用の一時カラムを削除
    
    output_file_path = os.path.join(output_dir, f"{target_race_id}_horse_past_races.csv")
    final_combined_df.to_csv(output_file_path, index=False, encoding='SHIFT-JIS')
    
    print(f"\n--- [SUCCESS] All past race data saved to: {output_file_path} ---")
    print(f"Total {len(final_combined_df)} records for {len(shutuba_info_df)} horses.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Scrape past race data for horses in a given race from individual pages.")
    parser.add_argument('race_id', help='Target 12-digit race ID (e.g., 202505041111)')
    args = parser.parse_args()
    
    # configファイルに NUM_PAST_RACES が定義されていることを確認
    if not hasattr(config, 'NUM_PAST_RACES'):
        print("[ERROR] config.py に NUM_PAST_RACES が定義されていません。デフォルト値を使用します。")
        config.NUM_PAST_RACES = 5 # デフォルト値
    
    # config.py に NUM_PAST_RACES_TO_SCRAPE が定義されていることを確認
    if not hasattr(config, 'NUM_PAST_RACES_TO_SCRAPE'):
        print("[ERROR] config.py に NUM_PAST_RACES_TO_SCRAPE が定義されていません。デフォルト値を使用します。")
        config.NUM_PAST_RACES_TO_SCRAPE = 10 # デフォルト値

    main_scrape_horse_past_races(args.race_id)