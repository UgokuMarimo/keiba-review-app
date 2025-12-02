# C:\KeibaAI\code\utils\scraper.py

import requests
from bs4 import BeautifulSoup
import time, re, os, sys
import pandas as pd
import numpy as np
from typing import Optional, List
from tqdm import tqdm

_current_dir = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..')); sys.path.append(PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
import config

REQUEST_WAIT_TIME = 0.3
def safe_get_text(element, strip=True): return element.get_text(strip=True) if element else ""
def to_numeric_or_nan(value):
    if value is None or str(value).strip() in ['', '**', '--', '---.-']: return np.nan
    try:
        cleaned_value = re.sub(r'[^\d.-]', '', str(value))
        if cleaned_value in ['.', '-', '']: return np.nan
        return float(cleaned_value)
    except (ValueError, TypeError): return np.nan

def scrape_shutuba_table(race_id: str) -> Optional[pd.DataFrame]:
    # (この関数は完成しているので、変更なし)
    print(f"[SCRAPER] Fetching shutuba table for race_id: {race_id}")
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15); r.raise_for_status(); time.sleep(REQUEST_WAIT_TIME)
    except requests.exceptions.RequestException as e: print(f"\n[SCRAPER ERROR]...: {e}"); return None
    soup = BeautifulSoup(r.content, "html.parser", from_encoding=r.apparent_encoding)
    race_info_header = soup.find("div", class_="RaceList_Item02");
    if not race_info_header: print(f"[SCRAPER WARN] Race info header not found for race_id: {race_id}"); return None
    race_name_h1 = race_info_header.find("h1", class_="RaceName"); race_name = safe_get_text(race_name_h1)
    grade_span = race_info_header.find("span", class_=re.compile(r'Icon_GradeType'))
    if grade_span:
        class_str = ' '.join(grade_span.get('class', [])); grade_match = re.search(r'Icon_GradeType(\d+)', class_str)
        if grade_match:
            grade_num = grade_match.group(1); grade_map = {'1': 'GI', '2': 'GII', '3': 'GIII'}
            if grade_num in grade_map: race_name += f" ({grade_map[grade_num]})"
    details01_text = safe_get_text(race_info_header.find("div", class_="RaceData01")); details02_text = safe_get_text(race_info_header.find("div", class_="RaceData02"))
    details02_spans = race_info_header.find("div", class_="RaceData02").find_all("span")
    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', soup.title.string); date_str = f"{date_match.group(1)}年{int(date_match.group(2))}月{int(date_match.group(3))}日" if date_match else ""
    track_type_match = re.search(r'(芝|ダ|障)', details01_text); track_type_char = track_type_match.group(1) if track_type_match else ''
    dist_match = re.search(r'(\d+)m', details01_text); distance = dist_match.group(1) if dist_match else ''
    turn_match = re.search(r'\((左|右|直)', details01_text); turn = turn_match.group(1) if turn_match else ''
    weather_match = re.search(r'天候:(\S+)', details01_text); weather = weather_match.group(1).split('/')[0].strip() if weather_match else ''
    track_cond_match = re.search(r'馬場:(\S+)', details01_text); track_condition = track_cond_match.group(1).strip() if track_cond_match else ''
    place_match = re.search(r'(東京|中山|阪神|京都|中京|新潟|福島|小倉|札幌|函館)', details02_text); place_name = place_match.group(1) if place_match else "不明"
    kaisai_match = re.search(r'(\d+回.+?\d+日目)', details02_text); kaisai = kaisai_match.group(1) if kaisai_match else ""
    race_class = f"{details02_spans[3].text.strip()} {details02_spans[4].text.strip()}" if len(details02_spans) > 4 else ""
    place_id = race_id[4:6] if len(race_id) >= 6 else ""
    table = soup.find("table", class_="ShutubaTable");
    if not table: return None
    records = []
    for row in table.find_all("tr", class_="HorseList"):
        cols = row.find_all("td");
        if not cols or len(cols) < 11: continue
        try:
            horse_link = cols[3].find("a"); horse_id = re.search(r'/horse/(\d+)', horse_link['href']).group(1) if horse_link and 'href' in horse_link.attrs else ""
            if not horse_id: continue
            jockey_link = cols[6].find("a"); j_match = re.search(r'/jockey/.*?/(\d+)', jockey_link['href']) if jockey_link and jockey_link.has_attr('href') else None; jockey_id = j_match.group(1) if j_match else ""
            weight_text = safe_get_text(cols[8]); w_match = re.match(r'(\d+)\((.+)\)', weight_text); weight, weight_dif = (w_match.groups()) if w_match else (weight_text, '0')
            sex_age_text = safe_get_text(cols[4])
            records.append({
                'race_id': race_id, '馬番': to_numeric_or_nan(safe_get_text(cols[1])), '馬': safe_get_text(cols[3]), 'horse_id': horse_id,
                '性': sex_age_text[0] if sex_age_text else '', '齢': to_numeric_or_nan(sex_age_text[1:]),
                '斤量': to_numeric_or_nan(safe_get_text(cols[5])), '騎手': safe_get_text(cols[6]), 'jockey_id': jockey_id,
                '体重': to_numeric_or_nan(weight), '体重変化': to_numeric_or_nan(weight_dif),
                'オッズ': to_numeric_or_nan(safe_get_text(cols[9].find('span'))), '人気': to_numeric_or_nan(safe_get_text(cols[10].find('span'))),
                'レース名': race_name, '日付': date_str, '芝・ダート': track_type_char, '距離': to_numeric_or_nan(distance),
                '場名': place_name, '場id': place_id, 'クラス': race_class,
                '回り': turn, '天気': weather, '馬場': track_condition,
            })
        except Exception as e: print(f"\n[SCRAPER ERROR]...: {e}"); continue
    return pd.DataFrame(records) if records else None

# ★★★ ここを修正 ★★★
def load_past_race_data(horse_ids: List[str], data_dir: str = config.DATA_DIR) -> Optional[pd.DataFrame]:
    """過去走データをローカルのCSVファイル群から読み込む (有効化バージョン)"""
    print(f"\n--- [Phase 2/5] Loading Past Race Data for {len(horse_ids)} horses from local CSVs ---")
    all_past_races = []
    # 検索範囲をconfigファイルから取得
    years = range(config.BUILD_END_YEAR, config.BUILD_START_YEAR - 1, -1)
    horse_id_set = set(map(str, horse_ids))
    
    for year in tqdm(years, desc="Loading past data by year"):
        file_path = os.path.join(data_dir, f"{year}.csv")
        if not os.path.exists(file_path): continue
        try:
            # low_memory=False を指定してDtypeWarningを抑制
            df = pd.read_csv(file_path, encoding="SHIFT-JIS", header=0, low_memory=False)
            if 'horse_id' not in df.columns: continue
            
            df['horse_id'] = df['horse_id'].astype(str)
            target_rows = df[df['horse_id'].isin(horse_id_set)]
            
            if not target_rows.empty:
                all_past_races.append(target_rows)
        except Exception as e:
            print(f"[DATA LOADER ERROR]...: {e}"); continue
            
    if not all_past_races:
        print("[DATA LOADER WARN] No past race data found for any of the specified horses in local files.")
        return None
        
    combined_df = pd.concat(all_past_races, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    print(f"-> Found {len(combined_df)} past race entries from local files.")
    return combined_df


# ★★★ 新規追加: 海外・地方競馬対応版 ★★★
def load_past_race_data_with_overseas(
    horse_ids: List[str],
    race_date: str,
    num_past_races: int = 5,
    use_horse_page: bool = True,
    save_to_cache: bool = True,
    data_dir: str = config.DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    海外・地方競馬を含む過去走データを取得
    
    Args:
        horse_ids: 馬IDのリスト
        race_date: レース日付 (YYYY年MM月DD日形式)
        num_past_races: 取得する過去走数
        use_horse_page: 各馬ページからスクレイピングするか
        save_to_cache: 海外・地方データをキャッシュに保存するか
        data_dir: データディレクトリ
    
    Returns:
        過去走データのDataFrame (JRA + 海外 + 地方)
    """
    print(f"\n--- [Phase 2/5] Loading Past Race Data (with overseas/local) for {len(horse_ids)} horses ---")
    
    if not use_horse_page:
        # フォールバック: 従来の方法
        return load_past_race_data(horse_ids, data_dir)
    
    # 各馬ページからスクレイピング
    from code.a1_data_collection.scrape_horse_past_races import scrape_all_past_races_from_horse_page
    
    all_past_races = []
    kaigai_races_to_save = []
    tihou_races_to_save = []
    
    # 地方競馬場のリスト
    LOCAL_TRACKS = {'大井', '船橋', '川崎', '浦和', '門別', '盛岡', '水沢', '金沢', '笠松', '名古屋', '園田', '姫路', '高知', '佐賀'}
    
    for horse_id in tqdm(horse_ids, desc="Scraping horse pages"):
        horse_url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
        
        try:
            # 各馬ページから過去走を取得
            past_races_df = scrape_all_past_races_from_horse_page(
                horse_id, 
                horse_url, 
                max_races=num_past_races * 2  # 余裕を持って多めに取得
            )
            
            if past_races_df is None or past_races_df.empty:
                continue
            
            # 日付でソート(最新順)
            past_races_df['日付_temp'] = pd.to_datetime(past_races_df['日付'], format='%Y年%m月%d日', errors='coerce')
            past_races_df = past_races_df.sort_values('日付_temp', ascending=False)
            past_races_df = past_races_df.drop('日付_temp', axis=1)
            
            # 直近N走を取得
            recent_races = past_races_df.head(num_past_races)
            all_past_races.append(recent_races)
            
            # 海外・地方レースを分類
            if save_to_cache and 'is_jra_race' in past_races_df.columns:
                non_jra_races = past_races_df[past_races_df['is_jra_race'] == False].copy()
                
                if not non_jra_races.empty:
                    # 地方と海外を判別
                    for _, race in non_jra_races.iterrows():
                        place_name = race.get('場名', '')
                        if place_name in LOCAL_TRACKS:
                            tihou_races_to_save.append(race)
                        else:
                            kaigai_races_to_save.append(race)
            
        except Exception as e:
            print(f"[SCRAPER ERROR] Failed to scrape horse {horse_id}: {e}")
            continue
    
    # データを保存
    if save_to_cache:
        if kaigai_races_to_save:
            _save_overseas_data(pd.DataFrame(kaigai_races_to_save), 'kaigai', data_dir)
        if tihou_races_to_save:
            _save_overseas_data(pd.DataFrame(tihou_races_to_save), 'tihou', data_dir)
    
    if not all_past_races:
        print("[DATA LOADER WARN] No past race data found from horse pages.")
        return None
    
    combined_df = pd.concat(all_past_races, ignore_index=True)
    combined_df.drop_duplicates(subset=['horse_id', '日付', 'レース名'], inplace=True)
    print(f"-> Found {len(combined_df)} past race entries (including overseas/local).")
    
    return combined_df


def _save_overseas_data(df: pd.DataFrame, data_type: str, data_dir: str):
    """
    海外・地方レースデータを年別CSVに保存
    
    Args:
        df: 保存するデータフレーム
        data_type: 'kaigai' または 'tihou'
        data_dir: データディレクトリ
    """
    if df.empty:
        return
    
    # 保存先ディレクトリを作成
    save_dir = os.path.join(data_dir, data_type)
    os.makedirs(save_dir, exist_ok=True)
    
    # 年別に分割して保存
    df['日付_temp'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日', errors='coerce')
    df['year'] = df['日付_temp'].dt.year
    df = df.drop('日付_temp', axis=1)
    
    for year, year_df in df.groupby('year'):
        if pd.isna(year):
            continue
        
        year = int(year)
        file_path = os.path.join(save_dir, f"{year}.csv")
        
        # 既存ファイルがあれば読み込んで結合
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path, encoding='SHIFT-JIS', low_memory=False)
                year_df = pd.concat([existing_df, year_df], ignore_index=True)
                year_df.drop_duplicates(subset=['horse_id', '日付', 'レース名'], inplace=True)
            except Exception as e:
                print(f"[SAVE ERROR] Failed to load existing {data_type} data for {year}: {e}")
        
        # 保存
        try:
            year_df.to_csv(file_path, index=False, encoding='SHIFT-JIS')
            print(f"[SAVE] Saved {len(year_df)} {data_type} races to {file_path}")
        except Exception as e:
            print(f"[SAVE ERROR] Failed to save {data_type} data for {year}: {e}")