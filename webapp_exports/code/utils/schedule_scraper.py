# C:\keibaAI\code\a1_data_collection\m00_get_race_schedule.py

"""
netkeiba.comから指定された日付のレーススケジュール（全レースのrace_idと発走時刻）
を取得するスクリプト。
実績のあるSeleniumベースのコードを元に、プロジェクト仕様に合わせて関数化。

■ 主な処理
1. Selenium WebDriverを起動し、指定された日付のレース一覧ページにアクセスする。
2. JavaScriptの実行が完了し、レース情報が表示されるまで待機する。
3. 完全にレンダリングされた後のHTMLソースから、各レースのrace_idと発走時刻を抽出。
4. 抽出した情報をpandas DataFrameにまとめて返す。

■ 使い方
- 初回実行前にライブラリのインストールが必要:
  pip install selenium webdriver-manager
- 他のスクリプトから:
  from code.a1_data_collection.m00_get_race_schedule import get_race_schedule_for_date
- 単体でデバッグ実行:
  python code/a1_data_collection/m00_get_race_schedule.py [YYYY-MM-DD]
"""

import sys
import os
import pandas as pd
import re
from datetime import datetime, date
import time
import logging

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))
sys.path.append(PROJECT_ROOT)
# ---

# --- Selenium関連のインポート --
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Seleniumのドライバマネージャのログを抑制
logging.getLogger('WDM').setLevel(logging.WARNING)

TARGET_URL_TEMPLATE = "https://race.netkeiba.com/top/race_list.html?kaisai_date={}"

def setup_driver() -> webdriver.Chrome:
    """Selenium WebDriverをセットアップする（先輩のコードを参考）"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # webdriver-manager を利用してchromedriverを自動管理
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def get_race_schedule_for_date(target_date_str: str = None) -> pd.DataFrame | None:
    """
    指定された日付（YYYY-MM-DD形式）のレーススケジュールを取得する。
    """
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] Invalid date format: {target_date_str}. Please use YYYY-MM-DD."); return None
    else:
        target_date = date.today()

    target_date_formatted = target_date.strftime('%Y%m%d')
    target_url = TARGET_URL_TEMPLATE.format(target_date_formatted)
    
    print(f"--- [START] Fetching race schedule for {target_date.strftime('%Y-%m-%d')} using Selenium ---")
    
    driver = None
    all_races = []
    try:
        driver = setup_driver()
        print(f"-> Accessing target URL: {target_url}")
        driver.get(target_url)

        # ページが完全に読み込まれるのを少し待つ
        time.sleep(3)
        
        # 開催場ごとの情報を取得 (先輩のコードのセレクタを参考)
        race_list_sections = driver.find_elements(By.CSS_SELECTOR, '.RaceList_DataList')

        if not race_list_sections:
            print(f"[INFO] No race venues found for {target_date_formatted}."); return pd.DataFrame(columns=['race_id', 'start_time', 'venue_name', 'race_number'])

        print(f"-> Found {len(race_list_sections)} venues.")

        for section in race_list_sections:
            # 競馬場名を取得
            title = section.find_element(By.CSS_SELECTOR, '.RaceList_DataTitle').text
            venue_name = title.split()[1]
            
            # 各レースの情報を取得
            race_items = section.find_elements(By.CSS_SELECTOR, '.RaceList_DataItem')
            for item in race_items:
                # 発走時刻を取得
                start_time = item.find_element(By.CSS_SELECTOR, '.RaceList_Itemtime').text
                # レース番号を取得
                race_number = item.find_element(By.CSS_SELECTOR, '.Race_Num').text.replace('R', '')
                
                # レース名を取得
                try:
                    race_name_element = item.find_element(By.CSS_SELECTOR, '.RaceList_ItemTitle')
                    race_name = race_name_element.text
                except:
                    race_name = "レース名なし"

                # race_id を取得 (aタグのhrefから抽出)
                link_element = item.find_element(By.TAG_NAME, 'a')
                race_href = link_element.get_attribute('href')
                
                race_id_match = re.search(r'race_id=([^&]+)', race_href)
                if race_id_match:
                    race_id = race_id_match.group(1)
                    
                    all_races.append({
                        'race_id': race_id,
                        'start_time': start_time,
                        'venue_name': venue_name,
                        'race_number': race_number,
                        'race_name': race_name
                    })

    except Exception as e:
        print(f"[FATAL] An error occurred during scraping: {e}"); return None
    finally:
        if driver:
            driver.quit()

    if not all_races:
        print(f"[INFO] No valid race schedules could be extracted for {target_date_str}."); return pd.DataFrame(columns=['race_id', 'start_time', 'venue_name', 'race_number'])

    schedule_df = pd.DataFrame(all_races)
    schedule_df = schedule_df.sort_values('start_time').reset_index(drop=True)
    
    print(f"--- [SUCCESS] Found {len(schedule_df)} races in total. ---")
    return schedule_df

if __name__ == '__main__':
    target_date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    schedule = get_race_schedule_for_date(target_date_arg)
    if schedule is not None:
        if schedule.empty:
            print("\nNo races scheduled for the target date.")
        else:
            print("\n>>> Race Schedule:")
            print(schedule)