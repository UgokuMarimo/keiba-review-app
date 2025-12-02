# C:\KeibaAI\explanation_templates.py (SHAP値そのまま入力 & 数値→文字変換)

"""
AIによる予測解説文のテンプレートと、特徴量のグループ分けを定義するファイル。
このファイルを編集することで、解説の口調や内容を調整できます。
数値データを元の文字データに戻すためのマッピングもここで管理します。
"""

import pandas as pd

# --- ヘルパー関数: 数値データを人間に分かりやすい文字列に変換 ---

# preprocess_and_clean 関数で使用されているマッピングを複製し、逆引きできるようにする
CLASS_MAPPING_INT_TO_STR = {
    8: "G1", 7: "G2", 6: "G3", 5: "オープンクラス",
    4: "3勝クラス", 3: "2勝クラス", 2: "1勝クラス",
    1: "未勝利戦", 0: "新馬戦", -99: "障害レース" # 障害レースは予測対象外だが、データとして存在しうる
}

CATEGORY_MAPPINGS_INT_TO_STR = {
    '性': {0: '牡', 1: '牝', 2: 'セ'},
    '芝・ダート': {0: '芝', 1: 'ダ', 2: '障'},
    '回り': {0: '右', 1: '左', 2: '芝直線'}, # '芝'と'直'は同じ2にマッピングされているので'芝直線'で統一
    '天気': {0: '晴', 1: '曇', 2: '小雨', 3: '雨', 4: '雪'},
    '馬場': {0: '良', 1: '稍重', 2: '重', 3: '不良'}
}

GATE_GROUP_MAPPING_INT_TO_STR = {
    '内': '内枠',
    '中': '中枠',
    '外': '外枠',
    '少頭数': '少頭数', # 変わらないが念のため
    '不明': '不明' # 変わらないが念のため
}


def get_original_value_display(feature_name: str, numeric_value: float) -> str:
    """
    特徴量名と数値データを受け取り、可能な場合は元の文字データを返す。
    そうでない場合は、元の数値データを文字列として返す。
    """
    if pd.isna(numeric_value):
        return "N/A"

    # クラス系の特徴量 (例: クラス1, クラス)
    if "クラス" in feature_name:
        return CLASS_MAPPING_INT_TO_STR.get(int(numeric_value), f"不明なクラス({int(numeric_value)})")
    
    # カテゴリマッピングに対応する特徴量
    for cat_feature, mapping in CATEGORY_MAPPINGS_INT_TO_STR.items():
        if cat_feature in feature_name: # '性1', '芝・ダート' などに対応
            return mapping.get(int(numeric_value), f"不明なカテゴリ({int(numeric_value)})")
    
    # 馬番グループ
    if "馬番グループ" in feature_name:
        # 馬番グループは文字列が値として入ってくる可能性があるので、そのまま渡す
        return str(numeric_value)
    
    # 長期休養明けフラグ
    if "長期休養明けフラグ" == feature_name:
        return "長期休養明け" if numeric_value == 1 else "長期休養ではない"

    # デフォルト: 数値をそのまま返す
    return f"{numeric_value:.2f}" if isinstance(numeric_value, float) and numeric_value != int(numeric_value) else str(int(numeric_value))

# --- 特徴量を、人間が解釈する際の「テーマ」ごとにグループ分けする ---
# 今回はLLMに自由な解釈をさせるため、グループ分けは残しつつ、具体的なテンプレートは削除します。
# LLMへのプロンプト生成時にグループ名を使用する想定です。
FEATURE_GROUPS = {
    "近走実績": {
        "positive": [
            "クラス1", "着順1", "オッズ1", "上がり1", "走破時間1", "通過順_平均1", "通過順_変動1",
            "クラス2", "着順2", "オッズ2", "上がり2", "走破時間2", "通過順_平均2", "通過順_変動2", # 追加
            "クラス3", "着順3", "オッズ3", "上がり3", "走破時間3", "通過順_平均3", "通過順_変動3", # 追加
            "クラス4", "着順4", "オッズ4", "上がり4", "走破時間4", "通過順_平均4", "通過順_変動4", # 追加
            "クラス5", "着順5", "オッズ5", "上がり5", "走破時間5", "通過順_平均5", "通過順_変動5", # 追加
            "過去5走_着順_平均", "過去5走_着順_最小", "過去5走_上がり_平均", "過去5走_上がり_最小",
            "過去5走_斤量_平均", "過去5走_体重_平均", "過去5走_体重変化_平均", # 追加
            "過去5走_通過順_平均_平均", "過去5走_通過順_変動_平均", # 追加
            "過去5走_着順_平均_race_dev", "過去5走_着順_平均_race_min_diff", # 相対的な特徴量をここに追加
            "過去5走_上がり_平均_race_dev", "過去5走_上がり_平均_race_min_diff", # 相対的な特徴量をここに追加
            # その他、過去走のポジティブな統計量
        ],
        "negative": [
            "過去5走_着順_平均", "過去5走_着順_最大", "過去5走_上がり_最大", # 追加
            "過去5走_斤量_最大", "過去5走_体重_最小", "過去5走_体重変化_最小", # 追加
            "過去5走_通過順_平均_最大", "過去5走_通過順_変動_最大", # 追加
            "過去5走_着順_平均_race_mean", "過去5走_着順_平均_race_max", # 相対的な特徴量をここに追加
            "過去5走_上がり_平均_race_mean", "過去5走_上がり_平均_race_max", # 相対的な特徴量をここに追加
            # その他、過去走のネガティブな統計量
        ]
    },
    "騎手の能力と相性": {
        "positive": [
            "jockey_id", "騎手勝率", "騎手競馬場勝率", "騎手年間騎乗数",
            "騎手勝率_race_dev", "騎手勝率_race_max_diff",
            "騎手競馬場勝率_race_dev", "騎手競馬場勝率_race_max_diff",
            "騎手年間騎乗数_race_dev", "騎手年間騎乗数_race_max_diff", # 追加
            "騎手馬場適性" # 追加
        ],
        "negative": [
            "騎手勝率_race_min_diff", "騎手競馬場勝率_race_min_diff",
            "騎手年間騎乗数_race_min_diff", # 追加
            "騎手勝率_race_mean", "騎手競馬場勝率_race_mean", # 追加
            "騎手馬場適性" # ネガティブな貢献度の場合
        ]
    },
    "馬の適性": {
        "positive": [
            "距離", "芝・ダート", "回り", "is_niigata_1000m", "馬場脚質適性", "脚質" # 脚質も追加
        ],
        "negative": [
            "距離差1", "距離差2", "距離差3", "距離差4", "距離差5", # 全ての距離差を追加
            "過去5走_距離差_平均", "過去5走_距離差_最大", "過去5走_距離差_最小" # 距離差の統計量も追加
            # その他、適性に関するネガティブ要因
        ]
    },
    "当日の条件とローテーション": {
        "positive": [
            "馬場", "天気", "開催バイアス_馬番グループ", "前日バイアス_馬番グループ", # 追加
            "馬番グループ", "出走頭数", # 追加
            "開催バイアス_馬番グループ_race_dev", "開催バイアス_馬番グループ_race_min_diff", # 相対的な特徴量をここに追加
            # その他、レース当日の好条件
        ],
        "negative": [
            "長期休養明けフラグ", "日付差1", "日付差2", "日付差3", "日付差4", "日付差5", # 全ての日付差を追加
            "開催バイアス_馬番グループ_race_mean", "開催バイアス_馬番グループ_race_max" # 相対的な特徴量をここに追加
            # その他、ローテーションや当日の悪条件
        ]
    },
    "馬体重とコンディション": {
        "positive": [
            "体重変化", "体重増減適性", "体重_前走比_標準化",
            "weight_diff_mean" # 追加
        ],
        "negative": [
            "体重変化", "体重増減適性", "体重_前走比_標準化",
            "weight_diff_std" # 追加（標準偏差が大きい場合は不安定要素になりうる）
        ]
    },
    "その他": {
        "positive": [], # 必要に応じてここに特徴量を追加
        "negative": ["クラス", "tansho_ninki"] # 現状未分類の可能性が高いもの、または汎用的なもの
    }
}

# 特徴量名から所属するグループを取得するためのヘルパー関数
def get_group_for_feature(feature_name: str, shap_value: float) -> str:
    for group_name, features_dict in FEATURE_GROUPS.items():
        if shap_value >= 0 and feature_name in features_dict["positive"]:
            return group_name
        if shap_value < 0 and feature_name in features_dict["negative"]:
            return group_name
    return "その他"


# LLMにSHAP出力をそのまま渡すため、具体的なEXPLANATION_TEMPLATESは今回は削除またはデフォルトのみとします。
# LLMが自由な文章を生成するために、詳細なルールベースは入れません。
EXPLANATION_TEMPLATES = {
    "default_positive": lambda feature, value_display, shap_value: f"「{feature}」（値: {value_display}）がポジティブな影響 ({shap_value:+.3f}) を与えています。",
    "default_negative": lambda feature, value_display, shap_value: f"「{feature}」（値: {value_display}）がネガティブな影響 ({shap_value:+.3f}) を与えています。"
}