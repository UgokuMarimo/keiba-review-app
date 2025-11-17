const express = require('express');
const router = express.Router();
const path = require('path');
const Database = require('better-sqlite3');

// データベースへの接続
const dbPath = process.env.DB_PATH || path.resolve(__dirname, '..', 'database', 'database.db');
const db = new Database(dbPath);

/*
 * 機能1: レース情報取得 (ダミーデータ版)
 * GET /api/races/dummy/:raceId
 */
router.get('/races/dummy/:raceId', (req, res) => {
  const raceId = req.params.raceId;

  // 将来的にはここでスクレイピング処理を行う
  // 今回は固定のダミーデータを返す
  const dummyData = {
    raceInfo: {
      race_id: raceId,
      date: '2025-11-16',
      race_name: 'ダミー記念',
      course: '芝 右 2000m'
    },
    horseList: [
      { horse_name: 'ウマ1', jockey: '騎手A', popularity: 1, finish: 1 },
      { horse_name: 'ウマ2', jockey: '騎手B', popularity: 2, finish: 2 },
      { horse_name: 'ウマ3', jockey: '騎手C', popularity: 3, finish: 3 },
      { horse_name: 'ウマ4', jockey: '騎手D', popularity: 4, finish: 4 },
      { horse_name: 'ウマ5', jockey: '騎手E', popularity: 5, finish: 5 },
    ]
  };
  res.json(dummyData);
});

/*
 * 機能3: レビューのデータベース保存
 * POST /api/reviews
 */
router.post('/reviews', (req, res) => {
  const { raceInfo, reviews } = req.body;

  // トランザクションを開始して、一連の処理をまとめる
  const insert = db.transaction(() => {
    // 1. races テーブルにレース情報を挿入 (存在する場合は無視)
    const stmtRaces = db.prepare('INSERT OR IGNORE INTO races (race_id, date, race_name, course) VALUES (?, ?, ?, ?)');
    stmtRaces.run(raceInfo.race_id, raceInfo.date, raceInfo.race_name, raceInfo.course);

    // 2. race_reviews テーブルに各馬のレビューを挿入 (存在する場合は更新)
    const stmtReviews = db.prepare('INSERT OR REPLACE INTO race_reviews (race_id, horse_name, jockey, popularity, finish, comment) VALUES (?, ?, ?, ?, ?, ?)');
    for (const review of reviews) {
      stmtReviews.run(review.race_id, review.horse_name, review.jockey, review.popularity, review.finish, review.comment);
    }
  });

  try {
    insert();
    res.status(201).json({ message: 'レビューが正常に保存されました。' });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'データベースへの保存中にエラーが発生しました。' });
  }
});

/*
 * 機能4: 過去データ閲覧 (レース単位)
 * GET /api/reviews/race/:raceId
 */
router.get('/reviews/race/:raceId', (req, res) => {
  try {
    const stmt = db.prepare('SELECT * FROM race_reviews WHERE race_id = ? ORDER BY finish');
    const reviews = stmt.all(req.params.raceId);
    res.json(reviews);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'データの取得中にエラーが発生しました。' });
  }
});


/*
 * 機能4: 過去データ閲覧 (馬ごと)
 * GET /api/reviews/horse/:horseName
 */
router.get('/reviews/horse/:horseName', (req, res) => {
  try {
    // races テーブルと JOIN して、レース情報も一緒に取得する
    const stmt = db.prepare(`
      SELECT r.date, r.race_name, r.course, rr.finish, rr.comment
      FROM race_reviews rr
      JOIN races r ON rr.race_id = r.race_id
      WHERE rr.horse_name = ?
      ORDER BY r.date DESC
    `);
    const reviews = stmt.all(req.params.horseName);
    res.json(reviews);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'データの取得中にエラーが発生しました。' });
  }
});

/*
 * 機能4: 過去データ閲覧 (日付検索)
 * GET /api/races/date/:date
 */
router.get('/races/date/:date', (req, res) => {
    try {
      const stmt = db.prepare('SELECT * FROM races WHERE date = ?');
      const races = stmt.all(req.params.date);
      res.json(races);
    } catch (err) {
      console.error(err.message);
      res.status(500).json({ error: 'データの取得中にエラーが発生しました。' });
    }
});

/*
 * 機能追加: 全データのエクスポート
 * GET /api/export
 */
router.get('/export', (req, res) => {
  try {
    // races テーブルから全データを取得
    const racesStmt = db.prepare('SELECT * FROM races');
    const races = racesStmt.all();

    // race_reviews テーブルから全データを取得
    const reviewsStmt = db.prepare('SELECT * FROM race_reviews');
    const reviews = reviewsStmt.all();

    // 2つのテーブルのデータを1つのオブジェクトにまとめて返す
    const exportData = {
      races: races,
      reviews: reviews,
    };

    // ダウンロード用のヘッダーを設定
    const date = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    res.setHeader('Content-Disposition', `attachment; filename="keiba_review_backup_${date}.json"`);
    res.setHeader('Content-Type', 'application/json');
    res.json(exportData);

  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'データのエクスポート中にエラーが発生しました。' });
  }
});

module.exports = router;