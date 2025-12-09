import pandas as pd
import joblib

CSV_PATH = "final_up_profile_v2.csv"
MODEL_PATH = "up_classifier_specified.pkl"
OUTPUT_PATH = "upfile_label.csv"

# 模型特征
feature_cols = [
    "avg_length", "std_length", "upload_freq",
    "comment_repetition", "danmaku_missing_rate",
    "content_cohesion", "comment_topic_diversity",
    "danmaku_topic_diversity", "avg_play",
    "avg_danmaku", "med_play", "avg_comment_scraped",
    "med_danmaku"
]

# 1. 读取 CSV
df = pd.read_csv(CSV_PATH)

# 2. 加载模型
clf = joblib.load(MODEL_PATH)

# 3. 取出特征
X = df[feature_cols]

# 4. 预测 label（0 = 优质, 1 = 低质）
df["label_binary"] = clf.predict(X)

# 5. 保存新的 CSV
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("✔ 新的带标签 CSV 已生成：", OUTPUT_PATH)
print(df["label_binary"].value_counts())