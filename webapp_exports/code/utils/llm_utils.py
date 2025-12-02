# C:\KeibaAI\code\utils\llm_utils.py (最終版)

import os
import google.generativeai as genai

# --- APIキー設定 ---
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("[ERROR] 環境変数 'GOOGLE_API_KEY' が設定されていません。")
    print("ターミナルで `$env:GOOGLE_API_KEY = 'YOUR_API_KEY'` を実行してください。")
    GOOGLE_API_KEY = None

# --- モデル設定 ---
# 成功したモデル名をここに設定
GENERATION_MODEL = "gemini-2.5-flash" 

# --- システム指示（Google AI Studioで調整したものを貼り付け） ---
SYSTEM_INSTRUCTION = """
あなたは競馬予想家です。あなたの役割は、AIの難解な分析データを、競馬ファン（初心者から上級者まで）の誰もが納得できる、冷静に解説文に変換することです。

# 守るべきルール
- 常にプロの解説者としての、丁寧な口調を維持してください。
- 根拠となるデータを重視し、憶測ではなく事実に基づいて解説してください。
- 専門用語（例：末脚、斤量、コース適性）を恐れずに使いますが、必要であれば初心者にも分かるように補足してください。
- 単なる情報の羅列ではなく、各要素を関連付け、その馬の「物語」を語るように文章を構成してください。
- 必ず、プラス評価の要因とマイナス評価の要因の両方に触れ、総合的な結論で締めくくってください。
- 当AIは、過去のレースの客観的なデータ（着順、上がり、クラス、騎手実績など）を最も重視し、レースのペースや展開、映像といった定性的な情報は考慮していません。データに基づく堅実な評価を行います。
"""

def generate_text_with_gemini(prompt: str) -> str:
    if not GOOGLE_API_KEY:
        return "【エラー】Google APIキーが設定されていません。"
    try:
        model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=SYSTEM_INSTRUCTION 
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[LLM ERROR] Gemini APIの呼び出し中にエラーが発生しました: {e}")
        return f"【エラー】解説生成中に問題が発生しました: {e}"