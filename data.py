import pandas as pd

# === INPUT & OUTPUT FILES ===
SRC = "binary_features.csv"                 # 你上传并标注好的文件
DST = "database/upfile_data_labeled.csv"           # 最终数据库文件保存位置

# === 你要求保留的特征 ===
FEATURE_COLS = [
    "avg_comment_scraped",
    "avg_danmaku",
    "avg_length",
    "avg_play",
    "comment_repetition",
    "danmaku_missing_rate",
    "med_danmaku",
    "med_play",
    "std_length",
    "upload_freq"
]

# === 必须保留的基本字段 ===
BASE_COLS = ["uid", "up_name", "followers", "label_binary"]

# === 加载你手动标注过的文件 ===
df = pd.read_csv(SRC)

# === 检查是否缺少必备字段 ===
missing = [c for c in BASE_COLS + FEATURE_COLS if c not in df.columns]
if missing:
    print("❌ 你的 CSV 缺少字段：", missing)
else:
    print("所有字段均存在 ✔")

# === 只保留需要的字段 ===
df_clean = df[BASE_COLS + FEATURE_COLS].copy()

# === 保存最终数据库文件 ===
df_clean.to_csv(DST, index=False, encoding="utf-8-sig")

print("🎉 已生成干净数据库：", DST)
print("最终形状：", df_clean.shape)
print(df_clean.head())