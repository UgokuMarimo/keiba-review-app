# C:\KeibaAI\code\a4_prediction\m04a_generate_explanation.py (SHAP値そのまま入力 & 数値→文字変換対応)
'''
google ai studioのAPIキーを設定後実行可能
 $env:GOOGLE_API_KEY = 'APIキー' 

python code/a4_prediction/m04a_generate_explanation.py [レースID] [モデルタイプ] [予測順位何位の馬の言語化するのか] 

python code/a4_prediction/m04a_generate_explanation.py 202508040411 B 1
'''



import os
import sys
import json
import argparse
import pandas as pd

# --- プロジェクトパス設定 ---
_current_dir = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..')); sys.path.append(PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, 'code'))

# --- モジュールインポート ---
# explanation_templatesから必要な関数をインポート
from explanation_templates import FEATURE_GROUPS, get_group_for_feature, get_original_value_display, EXPLANATION_TEMPLATES
from utils.llm_utils import generate_text_with_gemini

# --- ヘルパー関数: SHAPデータから特定のファクターの値を取得 (変更なし) ---
def _get_factor_value(shap_df: pd.DataFrame, feature_name: str):
    """shap_dfから特定の特徴量の値を取得する"""
    if feature_name in shap_df['feature'].values:
        return shap_df[shap_df['feature'] == feature_name]['value'].iloc[0]
    return None

def generate_prompt(shap_data: dict, shap_df: pd.DataFrame) -> str:
    """SHAPデータとテンプレートからLLMに与えるプロンプトを生成する"""

    # 1. 特徴量をグループ分けし、グループごとに貢献度を合算
    shap_df['group'] = shap_df.apply(lambda row: get_group_for_feature(row['feature'], row['shap_value']), axis=1)
    group_summary = shap_df.groupby('group')['shap_value'].sum().sort_values(ascending=False)

    positive_themes_list = []
    negative_themes_list = []

    for group_name, total_shap in group_summary.items():
        if abs(total_shap) < 0.05: continue # 影響の小さいグループは無視

        theme_body = f"\n## テーマ：{group_name} (総合貢献度: {total_shap:+.2f})\n"
        
        # このバージョンでは複合ルールや単一特徴量の特定のテンプレートは使用せず、
        # LLMに直接SHAPデータを渡し、自由な解釈を促します。
        group_features = shap_df[shap_df['group'] == group_name].sort_values('shap_value', ascending=False)

        # 個々の特徴量の内訳を記述
        for _, factor in group_features.iterrows():
            feature_name = factor['feature']
            numeric_value = factor['value']
            shap_value = factor['shap_value']

            # 数値データを元の文字データに変換
            value_display = get_original_value_display(feature_name, numeric_value)

            # デフォルトテンプレートを使用し、特徴量名、変換後の値、SHAP値をそのまま渡す
            if shap_value >= 0:
                reason_text = EXPLANATION_TEMPLATES["default_positive"](feature_name, value_display, shap_value)
            else:
                reason_text = EXPLANATION_TEMPLATES["default_negative"](feature_name, value_display, shap_value)

            theme_body += f"- {reason_text}\n"
        
        if total_shap > 0:
            positive_themes_list.append(theme_body)
        else:
            negative_themes_list.append(theme_body)

    # 3. 最終的なプロンプトを組み立てる
    prompt = f"""あなたはプロの競馬予想家です。与えられたAIの分析データを元に、競走馬「{shap_data['horse_name']}」の評価について、プロの視点から「データに基づいて」自然で分かりやすい解説文を200字から300字程度で作成してください。

# AIの総合評価
- 予測順位: {shap_data['pred_rank']}位
- 予測値: {shap_data['pred_win_prob']:.4f}


# 分析データサマリー
--- ポジティブ要因 ---
{chr(10).join(positive_themes_list) if positive_themes_list else "特になし"}

--- ネガティブ要因 ---
{chr(10).join(negative_themes_list) if negative_themes_list else "特になし"}

# 指示
- 上記の「分析データサマリー」に記載された個々の要因を統合し、単なる箇条書きの要約ではなく、一つの流暢な文章としてください。
- 「総合貢献度」が大きい（絶対値）テーマや、個々の要因の貢献度が特に高いものを、この馬を評価する上での重要な理由として解説に含めてください。
- プラス評価とマイナス評価のテーマを比較し、総合的にどちらが上回っているかを結論付けてください。
- 専門用語は避けず、しかし競馬初心者にも分かるように補足しながら解説してください。
- 丁寧語で記述してください。
- 「今回のレースでは、～と見られます。」のような形で結論を述べてください。
- 勝つ確率の低い馬に関して無理に高く評価せず「今回は厳しい」など正当に評価してください。また、評価が低くてもどのように恵まれれば勝てるかもと考えることのできる馬がそのことを記述してください
"""
    return prompt

def main(race_id: str, model_type: str, rank: int): # model_typeは使わないが引数は残しておく
    print(f"--- [START] Generating explanation for race {race_id}, rank {rank} ---") # ここも修正します
    shap_file_path = os.path.join(PROJECT_ROOT, 'shap_results', race_id, f"shap_rank_{rank}.json") # rank 変数を使用
    if not os.path.exists(shap_file_path): 
        print(f"[ERROR] SHAP result file not found: {shap_file_path}"); 
        return
    
    with open(shap_file_path, 'r', encoding='utf-8') as f: 
        shap_data = json.load(f)
    
    all_factors = shap_data.get('positive_factors', []) + shap_data.get('negative_factors', [])
    if not all_factors: 
        print("[ERROR] No SHAP factors found in the JSON file."); 
        return
    
    shap_df = pd.DataFrame(all_factors)
    
    print("\n--- Generating prompt for LLM... ---")
    prompt = generate_prompt(shap_data, shap_df)
    
    # プロンプトの内容を確認したい場合はコメントアウトを解除
    # print("\n--- Generated Prompt ---")
    # print(prompt)
    # print("--- End Prompt ---")

    print("\n--- Calling LLM API to generate explanation... ---")
    explanation_text = generate_text_with_gemini(prompt)
    
    print("\n" + "="*70)
    print(f"🐴 {shap_data['horse_name']} (予測{shap_data['pred_rank']}位) のAI解説")
    print("="*70)
    print(explanation_text)
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate natural language explanation from SHAP results.")
    parser.add_argument('race_id', help='Target 12-digit race ID')
    parser.add_argument('model_type', help="Model type to use (e.g., 'B' or 'C')") # 使わないが引数は維持
    #削除してみる
    # parser.add_argument('run_shap', type=bool, nargs='?', default=True, help="Dummy arg to match predict.py signature.")
    parser.add_argument('rank', type=int, nargs='?', default=1, help="Target prediction rank to explain. Defaults to 1.")
    args = parser.parse_args()

    main(args.race_id, args.model_type.upper(), args.rank)