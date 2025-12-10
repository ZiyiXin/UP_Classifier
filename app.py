# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
from treeinterpreter import treeinterpreter as ti
from sklearn.pipeline import Pipeline

# 特征列配置（和训练保持一致）
from analysis import FEATURE_COLS

app = Flask(__name__)

# ================== 路径 ==================
CSV_PATH = "database/upfile_data_labeled.csv"
MODEL_PATH = "classifier/up_classifier_10dim.pkl"

# ================== 帮助函数 ==================
def bucket_from_percentile(p: float) -> str:
    """根据百分位给区间标签"""
    if p >= 80:
        return "Top 20%"
    elif p <= 20:
        return "Bottom 20%"
    return "Middle 60%"


def get_model_and_X_for_ti(clf, X_raw: np.ndarray):
    """
    为 treeinterpreter 提供:
      - 纯模型基 learner (RandomForest / DecisionTree 等)
      - 预处理后的特征矩阵 X_for_ti

    支持 sklearn Pipeline；若 clf 不是 Pipeline，则直接返回 clf 和 X_raw。
    """
    if isinstance(clf, Pipeline):
        if len(clf.steps) > 1:
            # 前面所有步骤视为预处理
            preproc = clf[:-1]
            model = clf.steps[-1][1]
            X_for_ti = preproc.transform(X_raw)
        else:
            model = clf.steps[-1][1]
            X_for_ti = X_raw
    else:
        model = clf
        X_for_ti = X_raw

    return model, X_for_ti


# ================== 初始化：加载数据 & 模型 & 预计算分数 ==================
print("[INIT] Loading CSV and model...")
df = pd.read_csv(CSV_PATH)
clf = joblib.load(MODEL_PATH)

# uid 转为 int
df["uid"] = df["uid"].astype(int)

# ---- 1. 基础特征矩阵 ----
X_all = df[FEATURE_COLS].values

# ---- 2. 模型预测（pipeline 直接用）----
proba_all = clf.predict_proba(X_all)          # shape (n_samples, n_classes)
pred_labels_all = clf.predict(X_all)          # shape (n_samples,)

df["model_prob_high"] = proba_all[:, 1]       # 类别 1 = 高商业价值
df["model_pred_label"] = pred_labels_all      # 0 = 低, 1 = 高

# ---- 3. 置信度：模型对自己预测标签的概率 ----
df["confidence"] = np.where(
    df["model_pred_label"] == 1,
    df["model_prob_high"],       # 预测为高价值 → 用 P(high)
    1.0 - df["model_prob_high"]  # 预测为低价值 → 用 P(low) = 1 - P(high)
)

# ---- 4. 全局 SHAP 范围（用 treeinterpreter）----
print("[INIT] Computing global SHAP contributions...")
model_for_ti, X_for_ti = get_model_and_X_for_ti(clf, X_all)

prediction_all, bias_all, contrib_all = ti.predict(model_for_ti, X_for_ti)
# contrib_all 形状通常是 (n_samples, n_features, n_classes) 或 (n_samples, n_features)

if contrib_all.ndim == 3:
    # 取“高价值”这一类（假定索引 1），如果只有一个类则取 0
    class_idx = 1 if contrib_all.shape[2] > 1 else 0
    contrib_class = contrib_all[:, :, class_idx]   # (n_samples, n_features)
elif contrib_all.ndim == 2:
    contrib_class = contrib_all                    # (n_samples, n_features)
else:
    raise ValueError(f"Unexpected contrib_all ndim: {contrib_all.ndim}")

# 每个样本的 SHAP 总和
shap_sums = contrib_class.sum(axis=1)              # (n_samples,)

SHAP_MIN = float(shap_sums.min())
SHAP_MAX = float(shap_sums.max())
print(f"[INIT] SHAP range: min={SHAP_MIN:.4f}, max={SHAP_MAX:.4f}")

df["shap_sum"] = shap_sums

if SHAP_MAX > SHAP_MIN:
    df["shap_norm"] = (df["shap_sum"] - SHAP_MIN) / (SHAP_MAX - SHAP_MIN)
else:
    # 极端情况：所有样本 SHAP 完全一样
    df["shap_norm"] = 0.5

# 限制在 [0,1]
df["shap_norm"] = df["shap_norm"].clip(0.0, 1.0)

# ---- 5. 综合商业价值评分：方法 C ----
# Score = 100 * (0.5 * confidence + 0.5 * shap_norm)
df["value_score"] = 100.0 * (0.5 * df["confidence"] + 0.5 * df["shap_norm"])

# ---- 6. 在全体 UP 中的评分百分位 + 区间 ----
df["score_percentile"] = df["value_score"].rank(pct=True) * 100.0
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


# ================== API: 单个 UP 信息（预测 + 综合评分） ==================
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

    pred_label = int(row["model_pred_label"])
    prob_high = float(row["model_prob_high"])
    confidence = float(row["confidence"])
    value_score = float(row["value_score"])
    score_percentile = float(row["score_percentile"])
    score_bucket = row["score_bucket"]
    shap_sum = float(row["shap_sum"])
    shap_norm = float(row["shap_norm"])

    label_name = "高商业价值" if pred_label == 1 else "低商业价值"

    return jsonify({
        "success": True,
        "uid": uid_int,
        "up_name": row.get("up_name", ""),
        "followers": int(row.get("followers", -1)),
        "prediction": {
            "label_binary": pred_label,
            "label_name": label_name,
            "prob_high": prob_high,
            "confidence": confidence,
            "value_score": value_score,      # ★ 已是综合评分
            "score_percentile": score_percentile,
            "score_bucket": score_bucket,
            "shap_sum": shap_sum,
            "shap_norm": shap_norm,
        },
        "features": {c: float(row[c]) for c in FEATURE_COLS}
    })


# ================== API: 优质 UP 统计（中位数 & 最小值） ==================
@app.route("/api/stats/good")
def good_stats():
    # 用你手动标注的 label_binary 作为“优质”UP 依据
    good_df = df[df["label_binary"] == 1]

    median_vals = good_df[FEATURE_COLS].median().to_dict()
    min_vals = good_df[FEATURE_COLS].min().to_dict()

    return jsonify({
        "median": median_vals,
        "min": min_vals
    })


# ================== API: 商业价值处方解释 ==================
@app.route("/api/prescription/<uid>")
def api_prescription(uid):
    uid = uid.strip()
    if not uid.isdigit():
        return jsonify({"success": False, "message": "Invalid UID"}), 400

    uid_int = int(uid)
    row_df = df[df["uid"] == uid_int]
    if row_df.empty:
        return jsonify({"success": False, "message": "UID not found"}), 404

    row = row_df.iloc[0]

    # 构造输入
    x_raw = np.array([[row[c] for c in FEATURE_COLS]])

    # 取得适用于 treeinterpreter 的模型 & 特征
    model_for_ti, x_for_ti = get_model_and_X_for_ti(clf, x_raw)

    try:
        prediction, bias, contributions = ti.predict(model_for_ti, x_for_ti)
        contrib_arr = contributions[0]   # 针对当前这一个样本

        if contrib_arr.ndim == 2:
            class_idx = 1 if contrib_arr.shape[1] > 1 else 0
            contrib_arr = contrib_arr[:, class_idx]  # (n_features,)
        # 如果是 (n_features,) 则保持不变

    except Exception as e:
        return jsonify({
            "success": False,
            "uid": uid_int,
            "message": f"Failed to compute contributions: {e}"
        }), 500

    contrib_dict = {
        FEATURE_COLS[i]: float(contrib_arr[i])
        for i in range(len(FEATURE_COLS))
    }

    # 当前样本的 SHAP 总和 + 归一化
    shap_sum = float(contrib_arr.sum())
    if SHAP_MAX > SHAP_MIN:
        shap_norm = (shap_sum - SHAP_MIN) / (SHAP_MAX - SHAP_MIN)
    else:
        shap_norm = 0.5
    shap_norm = float(np.clip(shap_norm, 0.0, 1.0))

    # -----------------------------
    # 🍀 自动生成自然语言提升建议
    # -----------------------------
    suggestions = []
    for feat, contrib in contrib_dict.items():

        if contrib < -0.02:  # 明显负向
            suggestions.append(
                f"【{feat}】 对商业价值造成负向影响（{contrib:.3f}）。建议重点优化。"
            )
        elif contrib > 0.02:  # 明显正向
            suggestions.append(
                f"【{feat}】 当前表现较好（贡献 {contrib:.3f}）。建议保持。"
            )
        else:
            suggestions.append(
                f"【{feat}】 影响较弱（{contrib:.3f}），可根据业务策略灵活调整。"
            )

    return jsonify({
        "success": True,
        "uid": uid_int,
        "shap_sum": shap_sum,
        "shap_norm": shap_norm,
        "contributions": contrib_dict,
        "suggestions": suggestions
    })


# ================== 启动 ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)