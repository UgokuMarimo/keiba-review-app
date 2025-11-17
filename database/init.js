const path = require('path');
const fs = require('fs');
const Database = require('better-sqlite3');

const dbPath = process.env.DB_PATH || path.resolve(__dirname, 'database.db');
const dbDir = path.dirname(dbPath);

if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

const db = new Database(dbPath, { verbose: console.log });

const createTablesSql = `
CREATE TABLE IF NOT EXISTS races (
  race_id   TEXT PRIMARY KEY,
  date      TEXT NOT NULL,
  race_name TEXT NOT NULL,
  course    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS race_reviews (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  race_id     TEXT NOT NULL,
  horse_name  TEXT NOT NULL,
  jockey      TEXT,
  popularity  INTEGER,
  finish      INTEGER,
  comment     TEXT,
  FOREIGN KEY (race_id) REFERENCES races (race_id) ON DELETE CASCADE,
  UNIQUE (race_id, horse_name)
);
`;

try {
  console.log('データベースとテーブルの初期化を開始します...');
  db.exec(createTablesSql);
  console.log('テーブルが正常に作成されました。');
} catch (err) {
  console.error('テーブル作成中にエラーが発生しました:', err.message);
} finally {
  db.close();
  console.log('データベース接続を閉じました。');
}