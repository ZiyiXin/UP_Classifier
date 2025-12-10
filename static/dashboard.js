// dashboard.js
// 单个 UP 仪表盘：调用 /api/predict/<uid> 和 /api/stats/good 画四个分组图

const FEATURE_GROUPS = {
    interaction_primary: ["avg_comment_scraped", "avg_danmaku"],
    play_stats: ["avg_play", "med_play"],
    interaction_behavior: ["danmaku_missing_rate", "comment_repetition", "upload_freq"],
    video_length: ["avg_length", "std_length"],
};

async function loadDashboard() {
    // UID 从模板注入（优先）或 URL 查询参数获取
    const urlParams = new URLSearchParams(window.location.search);
    const uidFromQuery = urlParams.get("uid");
    const uid = (typeof UID_FROM_SERVER !== "undefined" && UID_FROM_SERVER) || uidFromQuery;

    if (!uid) {
        console.error("No UID provided");
        return;
    }

    // 请求预测信息、优质 UP 统计 + 解释
    const [predResp, statsResp, presResp] = await Promise.all([
        fetch(`/api/predict/${uid}`),
        fetch("/api/stats/good"),
        fetch(`/api/prescription/${uid}`),
    ]);

    if (!predResp.ok) {
        console.error("Predict API error:", predResp.status);
        return;
    }
    if (!statsResp.ok) {
        console.error("Stats API error:", statsResp.status);
        return;
    }

    const pred = await predResp.json();
    const stats = await statsResp.json();
    let pres = null;
    if (presResp.ok) {
        pres = await presResp.json();
    } else {
        console.warn("Prescription API error:", presResp.status);
    }

    if (!pred.success) {
        console.error("Predict API returned error:", pred);
        return;
    }

    fillInfoPanel(pred);
    drawAllCharts(pred, stats);
    if (pres) {
        fillExplanation(pres);
    }
}

function fillInfoPanel(pred) {
    const p = pred.prediction;

    document.getElementById("up_name").innerText = pred.up_name || "-";
    document.getElementById("followers").innerText = pred.followers ?? "-";
    document.getElementById("label_name").innerText = p.label_name || "-";
    document.getElementById("confidence").innerText = p.confidence.toFixed(3);
    document.getElementById("value_score").innerText = p.value_score.toFixed(1);
    document.getElementById("score_bucket").innerText = p.score_bucket;
    const percentileSpan = document.getElementById("score_percentile_text");
    if (percentileSpan) {
        percentileSpan.innerText = `（${p.score_percentile.toFixed(1)}%）`;
    }

    const confSpan = document.getElementById("score_confidence_text");
    if (confSpan) {
        confSpan.innerText = `${(p.confidence * 100).toFixed(1)}%`;
    }

    const shapSpan = document.getElementById("score_shap_text");
    if (shapSpan && typeof p.shap_norm === "number") {
        shapSpan.innerText = `${(p.shap_norm * 100).toFixed(1)}%`;
    }
}

function fillExplanation(pres) {
    if (!pres || !pres.success) {
        console.warn("No prescription data:", pres);
        return;
    }

    const contributionList = document.getElementById("contribution-list");
    const suggestionList = document.getElementById("suggestion-list");

    contributionList.innerHTML = "";
    suggestionList.innerHTML = "";

    // 贡献值列表
    Object.entries(pres.contributions).forEach(([feat, value]) => {
        const li = document.createElement("li");
        li.innerHTML = `<b>${feat}</b>: ${value.toFixed(3)}`;
        li.style.color = value >= 0 ? "green" : "red";
        contributionList.appendChild(li);
    });

    // 建议列表
    (pres.suggestions || []).forEach((s) => {
        const li = document.createElement("li");
        li.textContent = s;
        suggestionList.appendChild(li);
    });
}

function drawAllCharts(pred, stats) {
    const feats = pred.features;
    const med = stats.median || {};
    const min = stats.min || {};

    drawGroupChart(
        "chart_interaction_primary",
        FEATURE_GROUPS.interaction_primary,
        feats,
        med,
        min
    );

    drawGroupChart(
        "chart_play_stats",
        FEATURE_GROUPS.play_stats,
        feats,
        med,
        min
    );

    drawGroupChart(
        "chart_interaction_behavior",
        FEATURE_GROUPS.interaction_behavior,
        feats,
        med,
        min
    );

    drawGroupChart(
        "chart_video_length",
        FEATURE_GROUPS.video_length,
        feats,
        med,
        min
    );
}

function drawGroupChart(canvasId, cols, featValues, medianValues, minValues) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn("Canvas not found:", canvasId);
        return;
    }
    const ctx = canvas.getContext("2d");

    const labels = cols;
    const upVals = cols.map((c) => featValues[c]);
    const medVals = cols.map((c) => medianValues[c]);
    const minVals = cols.map((c) => minValues[c]);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "当前UP",
                    data: upVals,
                    backgroundColor: "rgba(54,162,235,0.7)",
                },
                {
                    label: "优质UP 中位数",
                    data: medVals,
                    backgroundColor: "rgba(255,159,64,0.7)",
                },
                {
                    label: "优质UP 最小值",
                    data: minVals,
                    backgroundColor: "rgba(75,192,192,0.7)",
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true },
            },
        },
    });
}

// 初始化
loadDashboard();
