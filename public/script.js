document.addEventListener('DOMContentLoaded', () => {
  // === 要素の取得 ===
  const getRaceButton = document.getElementById('get-race-button');
  const raceIdInput = document.getElementById('race-id-input');
  const raceReviewSection = document.getElementById('race-review-section');
  
  const searchRaceButton = document.getElementById('search-race-button');
  const searchHorseButton = document.getElementById('search-horse-button');
  const searchDateButton = document.getElementById('search-date-button');
  
  const searchRaceIdInput = document.getElementById('search-race-id');
  const searchHorseNameInput = document.getElementById('search-horse-name');
  const searchDateInput = document.getElementById('search-date');
  
  const searchResultsContent = document.getElementById('search-results-content');

  const exportDataButton = document.getElementById('export-data-button');
  
  // === イベントリスナーの設定 ===

  // 「レース情報取得」ボタン
  getRaceButton.addEventListener('click', async () => {
    const raceId = raceIdInput.value;
    if (!raceId) {
      alert('レースIDを入力してください。');
      return;
    }
    // APIを叩いてダミーデータを取得
    try {
      const response = await fetch(`/api/races/dummy/${raceId}`);
      if (!response.ok) throw new Error('サーバーからの応答がありません。');
      const data = await response.json();
      renderReviewForm(data);
    } catch (error) {
      alert(`エラーが発生しました: ${error.message}`);
    }
  });

  // 「レースIDで検索」ボタン
  searchRaceButton.addEventListener('click', async () => {
    const raceId = searchRaceIdInput.value;
    if (!raceId) { alert('レースIDを入力してください。'); return; }
    try {
      const response = await fetch(`/api/reviews/race/${raceId}`);
      const data = await response.json();
      renderRaceSearchResults(data);
    } catch (error) {
      alert(`検索エラー: ${error.message}`);
    }
  });

  // 「馬名で検索」ボタン
  searchHorseButton.addEventListener('click', async () => {
    const horseName = searchHorseNameInput.value;
    if (!horseName) { alert('馬名を入力してください。'); return; }
    try {
      const response = await fetch(`/api/reviews/horse/${horseName}`);
      const data = await response.json();
      renderHorseSearchResults(data);
    } catch (error) {
      alert(`検索エラー: ${error.message}`);
    }
  });

  // 「日付で検索」ボタン
  searchDateButton.addEventListener('click', async () => {
    const date = searchDateInput.value;
    if (!date.match(/^\d{4}-\d{2}-\d{2}$/)) { 
        alert('日付を YYYY-MM-DD 形式で入力してください。'); 
        return; 
    }
    try {
      const response = await fetch(`/api/races/date/${date}`);
      const data = await response.json();
      renderDateSearchResults(data);
    } catch (error) {
      alert(`検索エラー: ${error.message}`);
    }
  });

  // 「全データをエクスポート」ボタン
  exportDataButton.addEventListener('click', async () => {
    if (!confirm('データベースの全データをJSONファイルとしてダウンロードします。よろしいですか？')) {
      return;
    }
    
    try {
      const response = await fetch('/api/export');
      if (!response.ok) {
        throw new Error('サーバーからの応答がありません。');
      }

      // レスポンスからBlobオブジェクト（ファイルデータ）を取得
      const blob = await response.blob();
      
      // ダウンロード用のURLを生成
      const url = window.URL.createObjectURL(blob);
      
      // aタグを動的に作成してクリックさせることでダウンロードを実行
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      
      // バックエンドで設定したファイル名をここで取得
      const contentDisposition = response.headers.get('content-disposition');
      let fileName = 'keiba_review_backup.json'; // デフォルトファイル名
      if (contentDisposition) {
        const fileNameMatch = contentDisposition.match(/filename="(.+)"/);
        if (fileNameMatch.length === 2)
          fileName = fileNameMatch[1];
      }
      a.download = fileName;

      document.body.appendChild(a);
      a.click();
      
      // 後片付け
      window.URL.revokeObjectURL(url);
      a.remove();
      
    } catch (error) {
      alert(`エクスポートに失敗しました: ${error.message}`);
    }
  });

  // === 画面描画関数 ===

  // レビュー入力フォームを描画する関数
  function renderReviewForm(data) {
    const { raceInfo, horseList } = data;
    let tableRows = horseList.map(horse => `
      <tr data-horse-name="${horse.horse_name}">
        <td>${horse.horse_name}</td>
        <td><input type="text" class="jockey" value="${horse.jockey}"></td>
        <td><input type="number" class="popularity" value="${horse.popularity}"></td>
        <td><input type="number" class="finish" value="${horse.finish}"></td>
        <td>
          <textarea class="comment"></textarea>
          <div class="template-buttons">
            <button type="button" class="template-btn">展開</button>
            <button type="button" class="template-btn">位置</button>
            <button type="button" class="template-btn">短評</button>
          </div>
        </td>
      </tr>
    `).join('');

    raceReviewSection.innerHTML = `
      <h3>${raceInfo.race_name} (${raceInfo.course})</h3>
      <p>日付: ${raceInfo.date}</p>
      <table id="review-table">
        <thead>
          <tr>
            <th>馬名</th>
            <th>騎手</th>
            <th>人気</th>
            <th>着順</th>
            <th>コメント</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
      <button id="save-reviews-button">この内容で保存する</button>
    `;
    
    // 保存ボタンにイベントリスナーを追加
    document.getElementById('save-reviews-button').addEventListener('click', () => saveReviews(raceInfo));

    // テンプレートボタンにイベントリスナーを追加
    document.querySelectorAll('.template-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const textarea = e.target.closest('td').querySelector('.comment');
            const templateText = `【${e.target.textContent}】\n`;
            textarea.value += templateText;
            textarea.focus();
        });
    });
  }

  // レース単位の検索結果を描画する関数
  function renderRaceSearchResults(reviews) {
    if (reviews.length === 0) {
      searchResultsContent.innerHTML = '<p>該当するレビューは見つかりませんでした。</p>';
      return;
    }
    const list = reviews.map(r => `
      <li>
        <p class="result-meta">
          <strong>${r.horse_name}</strong> (${r.finish}着 / ${r.popularity}人気 / ${r.jockey})
        </p>
        <p class="result-comment">${escapeHTML(r.comment)}</p>
      </li>
    `).join('');
    searchResultsContent.innerHTML = `<ul>${list}</ul>`;
  }
  
  // 馬名での検索結果を描画する関数
  function renderHorseSearchResults(reviews) {
    if (reviews.length === 0) {
      searchResultsContent.innerHTML = '<p>該当するレビューは見つかりませんでした。</p>';
      return;
    }
    const list = reviews.map(r => `
      <li>
        <p class="result-meta">
          <strong>${r.race_name}</strong> (${r.date} / ${r.course}) - <strong>${r.finish}着</strong>
        </p>
        <p class="result-comment">${escapeHTML(r.comment)}</p>
      </li>
    `).join('');
    searchResultsContent.innerHTML = `<ul>${list}</ul>`;
  }

  // 日付での検索結果を描画する関数
  function renderDateSearchResults(races) {
    if (races.length === 0) {
      searchResultsContent.innerHTML = '<p>該当するレースは見つかりませんでした。</p>';
      return;
    }
    const list = races.map(r => `
      <li>
        <p><strong>${r.race_name}</strong> (${r.course})</p>
        <p class="result-meta">レースID: ${r.race_id} | 日付: ${r.date}</p>
        <button onclick="document.getElementById('search-race-id').value='${r.race_id}'; document.getElementById('search-race-button').click();">このレースのレビューを見る</button>
      </li>
    `).join('');
    searchResultsContent.innerHTML = `<ul>${list}</ul>`;
  }

  // === データ保存・その他 ===

  // レビューを保存する関数
  async function saveReviews(raceInfo) {
    const reviews = [];
    const rows = document.querySelectorAll('#review-table tbody tr');
    rows.forEach(row => {
      reviews.push({
        race_id: raceInfo.race_id,
        horse_name: row.dataset.horseName,
        jockey: row.querySelector('.jockey').value,
        popularity: parseInt(row.querySelector('.popularity').value, 10) || null,
        finish: parseInt(row.querySelector('.finish').value, 10) || null,
        comment: row.querySelector('.comment').value,
      });
    });

    try {
      const response = await fetch('/api/reviews', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raceInfo, reviews }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      alert(result.message);
      raceReviewSection.innerHTML = ''; // 入力フォームをクリア
    } catch (error) {
      alert(`保存に失敗しました: ${error.message}`);
    }
  }

  // HTMLエスケープ関数 (XSS対策)
  function escapeHTML(str) {
    return str.replace(/[&<>"']/g, function(match) {
      return {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[match];
    });
  }

});