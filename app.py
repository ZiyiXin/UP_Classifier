# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# ==== 路径按你实际放置调整 ====
CSV_PATH = "database/upfile_data_labeled.csv"       # 你最新写入的总表
MODEL_PATH = "classifier/up_classifier_10dim.pkl"          # 10 维模型

# 和训练时一致的 10 维特征
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
    "upload_freq",
]

# ================== 初始化：加载数据 & 模型 & 预计算分数 ==================
print("[INIT] Loading CSV and model...")
df = pd.read_csv(CSV_PATH)
clf = joblib.load(MODEL_PATH)

# 确保 uid 是 int（方便匹配）
if df["uid"].dtype != np.int64 and df["uid"].dtype != np.int32:
    df["uid"] = df["uid"].astype(int)

# 用模型对全体样本跑一遍，得到：
# - high 概率
# - 模型预测的 label
X_all = df[FEATURE_COLS].values
proba_high = clf.predict_proba(X_all)[:, 1]  # 类别 1 = 高商业价值
pred_labels = clf.predict(X_all)            # 0=低, 1=高

df["model_prob_high"] = proba_high
df["model_pred_label"] = pred_labels

# 置信度：模型自己这次预测的那一类的概率
df["confidence"] = np.where(
    df["model_pred_label"] == 1,
    df["model_prob_high"],
    1.0 - df["model_prob_high"],
)

# 商业价值评分：把置信度映射到 0~100 区间
df["value_score"] = df["confidence"] * 100.0

# 在全体 UP 中的评分百分位（0~100）
df["score_percentile"] = df["value_score"].rank(pct=True) * 100.0

# 区间文本：Top 20% / Middle 60% / Bottom 20%
def bucket_from_percentile(p):
    if p >= 80:
        return "Top 20%"
    elif p <= 20:
        return "Bottom 20%"
    else:
        return "Middle 60%"

df["score_bucket"] = df["score_percentile"].apply(bucket_from_percentile)

print("[INIT] Ready.")


# ================== 页面路由 ==================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    uid = request.args.get("uid", "").strip()
    return render_template("dashboard.html", uid=uid)


# ================== API: 单个 UP 信息 + 预测 ==================
@app.route("/api/predict/<uid>")
def api_predict(uid):
    uid = uid.strip()
    if not uid.isdigit():
        return jsonify({"success": False, "message": "Invalid UID"}), 400

    uid_int = int(uid)
    row_df = df[df["uid"] == uid_int]
    if row_df.empty:
        return jsonify({"success": False, "message": "UID not found"}), 404

    row = row_df.iloc[0]

    # 模型预测 label（用预计算好的）
    pred_label = int(row["model_pred_label"])
    prob_high = float(row["model_prob_high"])
    confidence = float(row["confidence"])
    value_score = float(row["value_score"])
    score_percentile = float(row["score_percentile"])
    score_bucket = row["score_bucket"]

    label_name = "高商业价值" if pred_label == 1 else "低商业价值"

    return jsonify({
        "success": True,
        "uid": uid_int,
        "up_name": row.get("up_name", ""),
        "followers": int(row.get("followers", -1)),
        "prediction": {
            "label_binary": pred_label,          # 0 低 1 高
            "label_name": label_name,
            "prob_high": prob_high,              # 模型认为是高价值的概率
            "confidence": confidence,            # 当前预测的置信度（0~1）
            "value_score": value_score,          # 0~100
            "score_percentile": score_percentile,
            "score_bucket": score_bucket,
        },
        "features": {c: float(row[c]) for c in FEATURE_COLS}
    })


# ================== API: 优质 UP（label_binary=1）统计 ==================
@app.route("/api/stats/good")
def good_stats():
    # 这里用你手动标注的 label_binary 作为“优质”定义
    good_df = df[df["label_binary"] == 1]

    median_vals = good_df[FEATURE_COLS].median().to_dict()
    min_vals = good_df[FEATURE_COLS].min().to_dict()

    return jsonify({
        "median": median_vals,
        "min": min_vals
    })


if __name__ == "__main__":
    # 方便本地调试
    app.run(host="0.0.0.0", port=5001, debug=True)