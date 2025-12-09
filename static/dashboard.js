// static/dashboard.js

// 取 URL 中的 uid（或模板传入的 UID_FROM_SERVER）
function getUID() {
    if (typeof UID_FROM_SERVER !== "undefined" && UID_FROM_SERVER) {
        return String(UID_FROM_SERVER).trim();
    }
    const params = new URLSearchParams(window.location.search);
    return (params.get("uid") || "").trim();
}

const uid = getUID();
if (!uid) {
    alert("URL 缺少 uid 参数，例如：/dashboard?uid=383433896");
}

// Chart 实例
let chartInteractionPrimary = null;
let chartPlayStats = null;
let chartInteractionBehavior = null;
let chartVideoLength = null;

// 构造一个通用的数据集
function buildDatasets(keys, predFeatures, median, minVals) {
    const upVals = [];
    const medianVals = [];
    const minValues = [];

    keys.forEach((k) => {
        upVals.push(predFeatures[k]);
        medianVals.push(median[k]);
        minValues.push(minVals[k]);
    });

    return [
        {
            label: "当前UP",
            data: upVals,
            backgroundColor: "rgba(54, 162, 235, 0.7)",
        },
        {
            label: "优质UP 中位数",
            data: medianVals,
            backgroundColor: "rgba(255, 159, 64, 0.7)",
        },
        {
            label: "优质UP 最小值",
            data: minValues,
            backgroundColor: "rgba(99, 194, 111, 0.7)",
        },
    ];
}

// 生成一个柱状图
function createBarChart(ctx, labels, datasets) {
    return new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45,
                    },
                },
                y: {
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: {
                    display: true,
                },
            },
        },
    });
}

async function loadDashboard() {
    // 1. 请求预测结果
    const predRes = await fetch(`/api/predict/${uid}`).then((r) => r.json());
    if (!predRes.success) {
        alert(predRes.message || "查询失败");
        return;
    }

    // 2. 请求优质UP统计
    const stats = await fetch(`/api/stats/good`).then((r) => r.json());

    // ==== 左侧信息填充 ====
    document.getElementById("uid_text").innerText = predRes.uid;
    document.getElementById("up_name").innerText = predRes.up_name || "-";
    const followers = predRes.followers;
    document.getElementById("followers").innerText =
        followers && followers > 0 ? followers.toLocaleString() : "未知";

    const pred = predRes.prediction;
    const labelSpan = document.getElementById("label_name");
    labelSpan.innerText = pred.label_name || "-";
    labelSpan.classList.remove("low");
    if (pred.label_binary === 0) {
        labelSpan.classList.add("low");
    }

    document.getElementById("confidence").innerText =
        pred.confidence != null ? pred.confidence.toFixed(3) : "-";

    document.getElementById("value_score").innerText =
        pred.value_score != null ? pred.value_score.toFixed(2) : "-";

    document.getElementById("score_percentile").innerHTML =
        '评分区间：<span id="score_bucket">' + (pred.score_bucket || "-") + "</span>";

    // ==== 特征组 ====
    const features = predRes.features;
    const median = stats.median;
    const minVals = stats.min;

    // 分组定义（与你的方案一致）
    const interactionPrimaryKeys = ["avg_comment_scraped", "avg_danmaku"];
    const playStatKeys = ["avg_play", "med_play"];
    const interactionBehaviorKeys = [
        "danmaku_missing_rate",
        "comment_repetition",
        "upload_freq",
    ];
    const videoLengthKeys = ["avg_length", "std_length"];

    // interaction_primary
    {
        const ctx = document
            .getElementById("chart_interaction_primary")
            .getContext("2d");
        if (chartInteractionPrimary) chartInteractionPrimary.destroy();
        chartInteractionPrimary = createBarChart(
            ctx,
            interactionPrimaryKeys,
            buildDatasets(interactionPrimaryKeys, features, median, minVals)
        );
    }

    // play_stats
    {
        const ctx = document.getElementById("chart_play_stats").getContext("2d");
        if (chartPlayStats) chartPlayStats.destroy();
        chartPlayStats = createBarChart(
            ctx,
            playStatKeys,
            buildDatasets(playStatKeys, features, median, minVals)
        );
    }

    // interaction_behavior
    {
        const ctx = document
            .getElementById("chart_interaction_behavior")
            .getContext("2d");
        if (chartInteractionBehavior) chartInteractionBehavior.destroy();
        chartInteractionBehavior = createBarChart(
            ctx,
            interactionBehaviorKeys,
            buildDatasets(interactionBehaviorKeys, features, median, minVals)
        );
    }

    // video_length
    {
        const ctx = document
            .getElementById("chart_video_length")
            .getContext("2d");
        if (chartVideoLength) chartVideoLength.destroy();
        chartVideoLength = createBarChart(
            ctx,
            videoLengthKeys,
            buildDatasets(videoLengthKeys, features, median, minVals)
        );
    }
}

loadDashboard();