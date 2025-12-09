function getUID() {
    const params = new URLSearchParams(window.location.search);
    return params.get("uid");
}
const uid = getUID();

let chartPlay, chartInteract, chartContent;

async function loadDashboard() {

    // ---------- 1. 获取预测结果 ----------
    const predRes = await fetch(`/api/predict/${uid}`).then(r => r.json());
    const stats = await fetch(`/api/stats/good`).then(r => r.json());

    document.getElementById("uid").innerText = uid;
    document.getElementById("up_name").innerText = predRes.up_name;
    document.getElementById("followers").innerText = predRes.followers;

    const labelElem = document.getElementById("label");
    labelElem.innerText = predRes.label_name;
    labelElem.classList.add(predRes.label_binary === 0 ? "good" : "low");

    document.getElementById("confidence").innerText = predRes.confidence.toFixed(3);

    const f = predRes.features;
    const med = stats.median;
    const minVals = stats.min;


    // ---------- 公共函数：构建图表 ----------
    function buildChart(canvasId, labels, upVals, medVals, minValsArr, storedChart) {
        const ctx = document.getElementById(canvasId);

        if (storedChart) storedChart.destroy();

        return new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    { label: "该UP", data: upVals, backgroundColor: "rgba(54,162,235,0.7)" },
                    { label: "优质UP中位数", data: medVals, backgroundColor: "rgba(255,159,64,0.7)" },
                    { label: "优质UP最小值", data: minValsArr, backgroundColor: "rgba(120,120,120,0.5)" }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "top" } },
                scales: { y: { beginAtZero: true } }
            }
        });
    }


    // ---------- 图1：播放表现 ----------
    const playLabels = ["avg_play", "med_play"];
    chartPlay = buildChart(
        "chart_play",
        playLabels,
        playLabels.map(k => f[k]),
        playLabels.map(k => med[k]),
        playLabels.map(k => minVals[k]),
        chartPlay
    );


    // ---------- 图2：互动指标 ----------
    const interactLabels = ["avg_comment_scraped", "avg_danmaku", "med_danmaku"];
    chartInteract = buildChart(
        "chart_interact",
        interactLabels,
        interactLabels.map(k => f[k]),
        interactLabels.map(k => med[k]),
        interactLabels.map(k => minVals[k]),
        chartInteract
    );


    // ---------- 图3：内容结构 ----------
    const contentLabels = ["danmaku_missing_rate", "upload_freq", "avg_length", "std_length"];
    chartContent = buildChart(
        "chart_content",
        contentLabels,
        contentLabels.map(k => f[k]),
        contentLabels.map(k => med[k]),
        contentLabels.map(k => minVals[k]),
        chartContent
    );
}

loadDashboard();