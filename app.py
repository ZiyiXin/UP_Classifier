from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

CSV_PATH = "upfile_label.csv"
MODEL_PATH = "up_classifier_specified.pkl"

FEATURE_COLS = [
    "avg_comment_scraped",
    "avg_danmaku",
    "avg_length",
    "avg_play",
    "comment_repetition",
    "comment_topic_diversity",
    "content_cohesion",
    "danmaku_missing_rate",
    "danmaku_topic_diversity",
    "med_danmaku",
    "med_play",
    "std_length",
    "upload_freq"
]

# --- Load Data & Model ---
print("[INIT] Loading CSV and model...")
df = pd.read_csv(CSV_PATH)
clf = joblib.load(MODEL_PATH)
print("[INIT] Ready.")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    uid = request.args.get("uid", "")
    return render_template("dashboard.html", uid=uid)

# ----- API: Get prediction + profile -----
@app.route("/api/predict/<uid>")
def api_predict(uid):
    try:
        uid = int(uid)
    except:
        return jsonify({"success": False, "message": "Invalid UID"}), 400

    row = df[df["uid"] == uid]
    if row.empty:
        return jsonify({"success": False, "message": "UID not found"}), 404

    row = row.iloc[0]

    # 取所有模型特征
    x = np.array([[row[c] for c in FEATURE_COLS]])
    pred = clf.predict(x)[0]
    prob = float(clf.predict_proba(x)[0][pred])

    return jsonify({
        "success": True,
        "uid": uid,
        "up_name": row.get("up_name", ""),
        "followers": int(row.get("followers", -1)),   # ⭐ 新增
        "label_binary": int(pred),
        "label_name": "good_value" if pred == 0 else "low_value",
        "confidence": prob,                           # ⭐ 置信度字段
        "features": {c: float(row[c]) for c in FEATURE_COLS}
    })

# ----- API: stats: median + min -----
@app.route("/api/stats/good")
def good_stats():
    good_df = df[df["label_binary"] == 0]  # 0 = good_value

    median_vals = good_df[FEATURE_COLS].median().to_dict()
    min_vals = good_df[FEATURE_COLS].min().to_dict()

    return jsonify({
        "median": median_vals,
        "min": min_vals
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)