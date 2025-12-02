# C:\KeibaAI\code\utils\build_vector_db.py

#

import pandas as pd
import os
import sys
import chromadb
import google.generativeai as genai
from tqdm import tqdm

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..')); sys.path.append(PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, 'code'))
import config

# APIキーを設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("環境変数 'GOOGLE_API_KEY' が設定されていません。")
genai.configure(api_key=GOOGLE_API_KEY)

# --- 設定 ---
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db") # DBの保存場所
COLLECTION_NAME = "race_results" # DB内のテーブル名
EMBEDDING_MODEL = "models/text-embedding-004" # Googleのベクトル化モデル

def main():
    print("--- [START] Building Vector Database ---")
    
    # 1. 全ての過去レースデータを読み込む
    all_dfs = []
    print("Loading all past race data...")
    for year in range(config.BUILD_START_YEAR, config.BUILD_END_YEAR + 1):
        file_path = os.path.join(config.DATA_DIR, f"{year}.csv")
        if os.path.exists(file_path):
            all_dfs.append(pd.read_csv(file_path, encoding="SHIFT-JIS", low_memory=False))
    
    if not all_dfs:
        print("[ERROR] No data found in /data folder.")
        return
        
    df = pd.concat(all_dfs, ignore_index=True)
    df.dropna(subset=['horse_id', '着順', 'レース名'], inplace=True)
    print(f"Loaded {len(df)} race entries.")

    # 2. データベースのセットアップ
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    # 既に存在する場合は削除して作り直す
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting and rebuilding.")
        client.delete_collection(name=COLLECTION_NAME)
    
    collection = client.create_collection(name=COLLECTION_NAME)

    # 3. データをベクトル化してDBに保存
    # 処理が重いので、馬ごとにまとめて処理する
    print("Generating embeddings for each race entry...")
    documents = []
    metadatas = []
    ids = []
    
    # tqdmを使って進捗を表示
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # LLMが検索しやすいように、各行を自然言語の「ドキュメント」に変換
        doc_text = (
            f"レース「{row['レース名']}」({row['日付']})で、馬「{row['馬']}」は{row['着順']}着でした。"
            f"騎手は{row['騎手']}、距離は{row['距離']}m、馬場は{row['馬場']}でした。"
        )
        documents.append(doc_text)
        
        # 後で参照できるように、元のデータをメタデータとして保存
        metadatas.append({
            "horse_id": str(row['horse_id']),
            "race_id": str(row['race_id']),
            "rank": str(row['着順'])
        })
        
        # 各ドキュメントの一意なID
        ids.append(f"{row['race_id']}_{row['horse_id']}")
    
    # GoogleのAPIを使って、全ドキュメントを一度にベクトルに変換
    print(f"Embedding {len(documents)} documents... (This may take a while)")
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=documents,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = result['embedding']
    
    # ベクトル化したデータをDBに保存 (ChromaDBは自動でバッチ処理してくれる)
    print("Adding embeddings to the database...")
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("\n--- [SUCCESS] Vector Database has been built. ---")
    print(f"Total documents indexed: {collection.count()}")
    print(f"Database saved at: {VECTOR_DB_PATH}")

if __name__ == "__main__":
    main()