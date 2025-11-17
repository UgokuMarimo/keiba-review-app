const express = require('express');
const path = require('path');
const apiRoutes = require('./routes/api'); // 後ほど作成する API ルートファイルをインポート

const app = express();
const PORT = 3000; // アプリケーションが待機するポート番号

// JSON形式のリクエストボディを解析するためのミドルウェア
app.use(express.json());

// URLエンコードされたリクエストボディを解析するためのミドルウェア
app.use(express.urlencoded({ extended: true }));

// フロントエンドの静的ファイルを提供するためのミドルウェア
// 'public' ディレクトリ内のファイル（index.htmlなど）にアクセスできるようになる
app.use(express.static(path.join(__dirname, 'public')));

// '/api' で始まるリクエストを apiRoutes にルーティングする
app.use('/api', apiRoutes);

// サーバーを起動
app.listen(PORT, () => {
  console.log(`サーバーが http://localhost:${PORT} で起動しました`);
});