// dashboard.js
// 单个 UP 仪表盘

// ========= 多语言配置 =========
let currentLang = localStorage.getItem("lang") || "zh";

const MESSAGES = {
    zh: {
        back: "返回",
        backToHome: "返回首页",
        cancel: "终止分析",
        loading: "正在分析，请稍候…",
        analysisCanceled: "分析已终止。",
        analysisFailed: "分析失败，请稍后重试。",
        noVideoTip: "该 UP 当前没有可用视频数据，无法计算商业价值。",
        noSuggestion: "暂无具体建议",
        prescriptionFailedPrefix: "获取处方失败：",
        noUidTip: "当前页面缺少 UID，无法加载仪表盘。",

        currentUp: "当前UP",
        goodMedian: "优质UP 中位数",
        goodMin: "优质UP 最小值",

        dashboardEyebrow: "Creator Intelligence",
        dashboardTitle: "UP 商业价值仪表盘",
        insightEyebrow: "Diagnosis",
        insightColumnTitle: "解释与行动建议",
        chartEyebrow: "Benchmark",
        chartColumnTitle: "核心特征对比",
        basicInfoTitle: "UP 基本信息",
        modelSectionTitle: "模型判断",
        scoreSectionTitle: "商业价值评分",
        aiExplainTitle: "商业价值解释",
        suggestionsTitle: "提升建议",
        suggestionsSummaryTitle: "总结结论",
        peersTitle: "同类 UP 对标",
        viewDetails: "查看详情",
        close: "关闭",
        insightsTitle: "AI 诊断详情",
        strengthsTitle: "优势 Top 3",
        weaknessesTitle: "短板 Top 3",
        chartInteractionTitle: "互动规模",
        chartPlayTitle: "播放表现",
        chartBehaviorTitle: "互动质量",
        chartLengthTitle: "内容节奏",

        // 左侧 labels
        uidLabel: "UID：",
        upNameLabel: "UP主：",
        followersLabel: "粉丝数：",
        bizClassLabel: "商业分类：",
        confidenceLabel: "置信度：",
        scoreRangeLabel: "评分区间：",
        modelConfidenceLabel: "模型置信度",
        shapContributionLabel: "行为贡献",

        langToggleLabel: "中 / EN",
    },
    en: {
        back: "Back",
        backToHome: "Back to Home",
        cancel: "Cancel",
        loading: "Analyzing, please wait…",
        analysisCanceled: "Analysis was canceled.",
        analysisFailed: "Analysis failed. Please try again later.",
        noVideoTip: "This creator currently has no valid videos, so we cannot compute the business value.",
        noSuggestion: "No specific suggestions for now.",
        prescriptionFailedPrefix: "Failed to get prescription: ",
        noUidTip: "UID is missing, dashboard cannot be loaded.",

        currentUp: "Current UP",
        goodMedian: "Good creators median",
        goodMin: "Good creators minimum",

        dashboardEyebrow: "Creator Intelligence",
        dashboardTitle: "Creator Value Dashboard",
        insightEyebrow: "Diagnosis",
        insightColumnTitle: "Insights and actions",
        chartEyebrow: "Benchmark",
        chartColumnTitle: "Core feature benchmark",
        basicInfoTitle: "Basic Info",
        modelSectionTitle: "Model Decision",
        scoreSectionTitle: "Business Value Score",
        aiExplainTitle: "Value Explanation",
        suggestionsTitle: "Suggestions",
        suggestionsSummaryTitle: "Summary",
        peersTitle: "Similar creators",
        viewDetails: "View details",
        close: "Close",
        insightsTitle: "AI insights (details)",
        strengthsTitle: "Strengths Top 3",
        weaknessesTitle: "Weaknesses Top 3",
        chartInteractionTitle: "Interaction scale",
        chartPlayTitle: "Play performance",
        chartBehaviorTitle: "Interaction quality",
        chartLengthTitle: "Content rhythm",

        uidLabel: "UID:",
        upNameLabel: "Creator:",
        followersLabel: "Followers:",
        bizClassLabel: "Business class:",
        confidenceLabel: "Confidence:",
        scoreRangeLabel: "Score range:",
        modelConfidenceLabel: "Model confidence",
        shapContributionLabel: "Behavior contribution",

        langToggleLabel: "中 / EN",
    },
};

function t(key) {
    const table = MESSAGES[currentLang] || MESSAGES.zh;
    return table[key] || key;
}

function formatNumber(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n < 0) return "-";
    return n.toLocaleString();
}

function getCreatorInitial(name) {
    const s = String(name || "").trim();
    if (!s) return "UP";
    const first = Array.from(s)[0] || "U";
    return first.toUpperCase();
}

function sanitizeAiText(text) {
    if (!text) return "";
    let s = String(text);
    // Remove common lightweight markdown markers that look like "乱码" in plain text rendering
    s = s.replace(/\*\*(.+?)\*\*/g, "$1");
    s = s.replace(/__(.+?)__/g, "$1");
    s = s.replace(/`(.+?)`/g, "$1");
    return s;
}

function applyI18n() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
        const key = el.getAttribute("data-i18n");
        if (!key) return;
        el.textContent = t(key);
    });
}

// 更新已有图表的 legend 文案（中英文切换时调用）
function updateChartLegends() {
    Object.values(chartInstances).forEach((chart) => {
        if (!chart || !chart.data || !chart.data.datasets) return;
        const ds = chart.data.datasets;
        if (ds[0]) ds[0].label = t("currentUp");
        if (ds[1]) ds[1].label = t("goodMedian");
        if (ds[2]) ds[2].label = t("goodMin");
        chart.update();
    });

    if (!window.Chart || !Chart.instances) return;

    const instances = Chart.instances;

    // Chart.js v3/v4: Map，有 forEach
    if (typeof instances.forEach === "function") {
        instances.forEach((chart) => {
            if (!chart || !chart.data || !chart.data.datasets) return;
            const ds = chart.data.datasets;
            if (ds[0]) ds[0].label = t("currentUp");
            if (ds[1]) ds[1].label = t("goodMedian");
            if (ds[2]) ds[2].label = t("goodMin");
            chart.update();
        });
    } else {
        // Chart.js v2: 普通对象
        Object.values(instances).forEach((chart) => {
            if (!chart || !chart.data || !chart.data.datasets) return;
            const ds = chart.data.datasets;
            if (ds[0]) ds[0].label = t("currentUp");
            if (ds[1]) ds[1].label = t("goodMedian");
            if (ds[2]) ds[2].label = t("goodMin");
            chart.update();
        });
    }
}

// ========= Loading & 终止 =========
let currentController = null;

function setLoading(show) {
    const overlay = document.getElementById("loading-overlay");
    if (!overlay) return;
    overlay.style.display = show ? "flex" : "none";

    if (show) {
        setProgress(5);
        const text = document.getElementById("loading-text");
        if (text) text.textContent = t("loading");
    }
}

function setProgress(pct) {
    const bar = document.getElementById("loading-progress");
    if (!bar) return;
    bar.style.width = pct + "%";
}

function showGlobalMessage(msg) {
    const box = document.getElementById("global-message");
    if (!box) {
        alert(msg);
        return;
    }
    box.textContent = msg;
    box.style.display = "block";
}

function hideGlobalMessage() {
    const box = document.getElementById("global-message");
    if (!box) return;
    box.style.display = "none";
    box.textContent = "";
}

// ========= 特征分组 =========
const FEATURE_GROUPS = {
    interaction_primary: ["avg_comment_scraped", "avg_danmaku"],
    play_stats: ["avg_play", "med_play"],
    interaction_behavior: ["danmaku_missing_rate", "comment_repetition", "upload_freq"],
    video_length: ["avg_length", "std_length"],
};

const chartInstances = {};

// ========= 特征中英文解释（前端展示用；后端也会返回 feature_meta，这里作为兜底） =========
const LOCAL_FEATURE_META = {
    avg_comment_scraped: {
        zh: { name: "平均评论数（抓取）", desc: "每条视频抓取到的评论条数的平均值（最多抓取一定数量）。" },
        en: { name: "Avg comments (scraped)", desc: "Average number of comments scraped per video (capped by crawler)." },
    },
    avg_danmaku: {
        zh: { name: "平均弹幕数", desc: "每条视频弹幕条数的平均值。" },
        en: { name: "Avg danmaku", desc: "Average danmaku count per video." },
    },
    avg_length: {
        zh: { name: "平均视频时长（秒）", desc: "近若干条视频的平均时长（秒）。" },
        en: { name: "Avg length (sec)", desc: "Average video length in seconds." },
    },
    avg_play: {
        zh: { name: "平均播放量", desc: "近若干条视频播放量的平均值。" },
        en: { name: "Avg plays", desc: "Average play count across recent videos." },
    },
    comment_repetition: {
        zh: { name: "评论重复度", desc: "评论分词后 Top 词占比（越高表示重复/灌水越多）。" },
        en: { name: "Comment repetition", desc: "Top-token share after tokenization (higher means more repetitive/spammy)." },
    },
    danmaku_missing_rate: {
        zh: { name: "弹幕缺失率", desc: "弹幕为 0 的视频占比（越高代表互动缺失越多）。" },
        en: { name: "Danmaku missing rate", desc: "Share of videos with zero danmaku (higher means more missing engagement)." },
    },
    med_danmaku: {
        zh: { name: "弹幕中位数", desc: "近若干条视频弹幕条数的中位数。" },
        en: { name: "Median danmaku", desc: "Median danmaku count across recent videos." },
    },
    med_play: {
        zh: { name: "播放量中位数", desc: "近若干条视频播放量的中位数。" },
        en: { name: "Median plays", desc: "Median play count across recent videos." },
    },
    std_length: {
        zh: { name: "时长波动（标准差）", desc: "视频时长的标准差（越高代表内容结构/节奏不稳定）。" },
        en: { name: "Length volatility (std)", desc: "Standard deviation of video length (higher means less consistent)." },
    },
    upload_freq: {
        zh: { name: "更新频率（视频/天）", desc: "近若干条视频覆盖时间窗内的日均更新量。" },
        en: { name: "Upload frequency (videos/day)", desc: "Average uploads per day over the observed window." },
    },
};

let featureMetaFromServer = null;

function getFeatureMeta(key) {
    if (featureMetaFromServer && featureMetaFromServer[key]) return featureMetaFromServer[key];
    const local = LOCAL_FEATURE_META[key];
    if (!local) return { name: key, desc: "" };
    const langPack = local[currentLang] || local.zh;
    return { name: langPack.name || key, desc: langPack.desc || "" };
}

function featureLabel(key) {
    return (getFeatureMeta(key).name || key);
}

// ========= Dashboard 主流程 =========
async function loadDashboard() {
    // UID 从模板注入或 URL 获取
    const urlParams = new URLSearchParams(window.location.search);
    const uidFromQuery = urlParams.get("uid");
    const uid = (typeof UID_FROM_SERVER !== "undefined" && UID_FROM_SERVER) || uidFromQuery;

    if (!uid) {
        console.error("No UID provided");
        showGlobalMessage(t("noUidTip"));
        return;
    }

    // 开始一次新分析前，清掉上一次全局提示
    hideGlobalMessage();

    // 取消前一次请求
    if (currentController) {
        currentController.abort();
    }
    const controller = new AbortController();
    currentController = controller;

    setLoading(true);
    setProgress(10);

    try {
        // 1. 并发拉预测 & 统计
        const [predResp, statsResp] = await Promise.all([
            fetch(`/api/predict/${uid}?lang=${currentLang}`, { signal: controller.signal }),
            fetch("/api/stats/good", { signal: controller.signal }),
        ]);

        setProgress(50);

        // 处理预测接口 HTTP 层错误
        if (!predResp.ok) {
            console.error("Predict API error:", predResp.status);

            let msg = "";
            try {
                const errData = await predResp.json();
                msg = errData && errData.message ? String(errData.message) : "";
            } catch (e) {
                // body 不是 json 就忽略
            }

            if (msg.includes("未获取到任何视频") || msg.toLowerCase().includes("no video")) {
                showGlobalMessage(t("noVideoTip"));   // 业务提示：没作品
            } else if (msg) {
                showGlobalMessage(msg); // 显示后端具体错误（例如 UID 不在数据集）
            } else {
                showGlobalMessage(t("analysisFailed")); // 真错误
            }
            return;
        }

        if (!statsResp.ok) {
            console.error("Stats API error:", statsResp.status);
            showGlobalMessage(t("analysisFailed"));
            return;
        }

        const pred = await predResp.json();
        const stats = await statsResp.json();
        window.__lastStatsData = stats;

        // 后端返回 success=false 的情况
        if (!pred.success) {
            handlePredictError(pred);
            return;
        }

        // 后端 feature_meta（用于把接口字段转换成用户可读中文/英文）
        featureMetaFromServer = pred.feature_meta || null;

        // 2. 渲染基本信息 + 图表
        fillInfoPanel(pred);
        drawAllCharts(pred, stats);

        setProgress(75);

        // 3. 并发拉：处方 + 同类UP对标
        try {
            const [presResp, peerResp] = await Promise.all([
                fetch(`/api/prescription/${uid}?lang=${currentLang}`, { signal: controller.signal }),
                fetch(`/api/peers/${uid}?lang=${currentLang}&k=3`, { signal: controller.signal }),
            ]);

            if (presResp.ok) {
                const pres = await presResp.json();
                if (pres.success) fillExplanation(pres);
                else showPrescriptionError(pres.message);
            } else {
                showPrescriptionError(`HTTP ${presResp.status}`);
            }

            if (peerResp.ok) {
                const peers = await peerResp.json();
                if (peers && peers.success) fillPeers(peers);
            }
        } catch (error) {
            if (error.name === "AbortError") {
                console.warn("post-fetch aborted");
            } else {
                console.error("post-fetch error:", error);
            }
        }

        setProgress(100);
    } catch (error) {
        if (error.name === "AbortError") {
            console.warn("Dashboard loading aborted");
            showGlobalMessage(t("analysisCanceled"));
        } else {
            console.error("loadDashboard error:", error);
            showGlobalMessage(t("analysisFailed"));
        }
    } finally {
        setLoading(false);
        currentController = null;
    }
}

// ========= 错误处理（无作品等） =========
function handlePredictError(pred) {
    const msg = pred && pred.message ? String(pred.message) : "";
    if (msg.includes("未获取到任何视频") || msg.toLowerCase().includes("no video")) {
        showGlobalMessage(t("noVideoTip"));
    } else {
        showGlobalMessage(msg || t("analysisFailed"));
    }
}

// ========= UI 填充 =========
function fillInfoPanel(pred) {
    const p = pred.prediction;
    window.__lastPredData = pred;

    const upName = pred.up_name || "-";
    document.getElementById("up_name").textContent = upName;
    document.getElementById("followers").textContent = formatNumber(pred.followers);

    const initialEl = document.getElementById("creator_initial");
    if (initialEl) initialEl.textContent = getCreatorInitial(upName);

    // 商业分类：根据语言映射
    let labelText = p.label_name || "-";
    if (currentLang === "en") {
        if (labelText.includes("高商业价值")) {
            labelText = "High business value";
        } else if (labelText.includes("低商业价值")) {
            labelText = "Low business value";
        }
    }
    const labelEl = document.getElementById("label_name");
    labelEl.textContent = labelText;
    labelEl.classList.toggle("low", Number(p.label_binary) !== 1);

    document.getElementById("confidence").textContent = p.confidence.toFixed(3);
    document.getElementById("value_score").textContent = p.value_score.toFixed(1);
    document.getElementById("score_bucket").textContent = p.score_bucket;

    const meter = document.getElementById("score-meter-fill");
    if (meter) {
        const score = Math.max(0, Math.min(100, Number(p.value_score) || 0));
        meter.style.width = `${score}%`;
    }

    const percentileSpan = document.getElementById("score_percentile_text");
    if (percentileSpan) {
        const percentText = `${p.score_percentile.toFixed(1)}%`;
        if (currentLang === "en") {
            percentileSpan.textContent = ` (${percentText})`;
        } else {
            percentileSpan.textContent = `（${percentText}）`;
        }
    }

    const confSpan = document.getElementById("score_confidence_text");
    if (confSpan) {
        confSpan.textContent = `${(p.confidence * 100).toFixed(1)}%`;
    }

    const shapSpan = document.getElementById("score_shap_text");
    if (shapSpan && typeof p.shap_norm === "number") {
        shapSpan.textContent = `${(p.shap_norm * 100).toFixed(1)}%`;
    }
}

// ========= 建议翻译（把后端中文模板翻成英文） =========
function translateSuggestionToEn(s) {
    if (currentLang !== "en" || !s) return s;

    // 提取特征名和贡献值
    const featMatch = s.match(/【(.+?)】/);
    const feat = featMatch ? featMatch[1] : "";
    const numMatch = s.match(/([-+]?\d+\.\d+)/);
    const contrib = numMatch ? numMatch[1] : "";

    if (s.includes("对商业价值造成负向影响")) {
        return `[${feat}] has a negative impact on business value (contribution ${contrib}). Please prioritize improving this metric.`;
    }
    if (s.includes("当前表现较好")) {
        return `[${feat}] is currently performing well (contribution ${contrib}). Keep up this strength.`;
    }
    if (s.includes("影响较弱")) {
        return `[${feat}] has only a weak impact on business value (contribution ${contrib}). You can adjust it flexibly according to your strategy.`;
    }

    // 兜底：不认识的文案直接原样返回
    return s;
}

function fillExplanation(pres) {
    console.log("[DEBUG] 处方API返回的数据:", pres);

    if (!pres || !pres.success) {
        console.warn("No prescription data:", pres);
        return;
    }

    // prescription 的 feature_meta 可能更完整，优先覆盖
    if (pres.feature_meta) {
        featureMetaFromServer = pres.feature_meta;
    }

    const contributionList = document.getElementById("contribution-list");
    const suggestionList = document.getElementById("suggestion-list");
    const summaryEl = document.getElementById("suggestion-summary");
    const modalSummaryEl = document.getElementById("modal-summary");
    const strengthList = document.getElementById("strength-list");
    const weaknessList = document.getElementById("weakness-list");

    if (contributionList) contributionList.innerHTML = "";
    if (suggestionList) suggestionList.innerHTML = "";
    if (summaryEl) summaryEl.innerHTML = "";
    if (modalSummaryEl) modalSummaryEl.textContent = "";
    if (strengthList) strengthList.innerHTML = "";
    if (weaknessList) weaknessList.innerHTML = "";

    // 贡献值列表
    if (contributionList && pres.contributions) {
        const itemsByKey = {};
        if (Array.isArray(pres.items)) {
            pres.items.forEach((it) => {
                if (it && it.feature) itemsByKey[it.feature] = it;
            });
        }

        const entries = Object.entries(pres.contributions)
            .map(([feat, value]) => [feat, Number(value), itemsByKey[feat] || null])
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

        entries.forEach(([feat, value, it]) => {
            const meta = getFeatureMeta(feat);
            const li = document.createElement("li");
            li.title = meta.desc || "";
            const sign = value >= 0 ? "+" : "";

            let compare = "";
            if (it && typeof it.value === "number" && typeof it.good_median === "number") {
                const v = it.value;
                const m = it.good_median;
                if (currentLang === "en") {
                    compare = ` (current ${v.toFixed(3)} vs median ${m.toFixed(3)})`;
                } else {
                    compare = `（当前 ${v.toFixed(3)} vs 优质中位数 ${m.toFixed(3)}）`;
                }
            }

            li.innerHTML = `<b>${meta.name}</b>: ${sign}${value.toFixed(3)}${compare}`;
            li.style.color = value >= 0 ? "green" : "red";
            contributionList.appendChild(li);
        });
    }

    // 建议列表
    if (suggestionList) {
        // 优先使用结构化 suggestions_struct，便于按语言渲染
        if (pres.suggestions_struct && pres.suggestions_struct.length > 0) {
            pres.suggestions_struct.forEach((it) => {
                const meta = getFeatureMeta(it.feature);
                const li = document.createElement("li");
                const c = Number(it.contribution || 0);
                const sign = c >= 0 ? "+" : "";

                if (currentLang === "en") {
                    if (it.impact === "negative") {
                        li.textContent = `${meta.name} is dragging your score (${sign}${c.toFixed(3)}). Focus on improving this metric.`;
                    } else {
                        li.textContent = `${meta.name} is a current strength (${sign}${c.toFixed(3)}). Keep it stable.`;
                    }
                } else {
                    if (it.impact === "negative") {
                        li.textContent = `「${meta.name}」在拉低商业价值（贡献 ${sign}${c.toFixed(3)}），建议优先优化。`;
                    } else {
                        li.textContent = `「${meta.name}」是当前优势（贡献 ${sign}${c.toFixed(3)}），建议保持。`;
                    }
                }
                suggestionList.appendChild(li);
            });
        } else if (pres.suggestions && pres.suggestions.length > 0) {
            pres.suggestions.forEach((s) => {
                const li = document.createElement("li");
                li.textContent = translateSuggestionToEn(s);
                suggestionList.appendChild(li);
            });
        } else {
            const li = document.createElement("li");
            li.textContent = t("noSuggestion");
            suggestionList.appendChild(li);
        }
    }

    // DeepSeek 整理的总结（后端返回 llm_summary）
    if (summaryEl) {
        if (pres.llm_summary) {
            summaryEl.textContent = sanitizeAiText(pres.llm_summary);
            summaryEl.style.display = "block";
        } else {
            summaryEl.style.display = "none";
        }
    }

    if (modalSummaryEl) {
        modalSummaryEl.textContent = sanitizeAiText(pres.llm_summary || "");
    }

    // Strengths / Weaknesses（从 items 贡献值中取 Top 3）
    if ((strengthList || weaknessList) && Array.isArray(pres.items)) {
        const items = pres.items
            .filter((x) => x && typeof x.contribution === "number" && x.feature)
            .map((x) => ({
                feature: x.feature,
                contribution: Number(x.contribution),
                value: typeof x.value === "number" ? x.value : null,
                good_median: typeof x.good_median === "number" ? x.good_median : null,
            }));

        const topPos = [...items]
            .filter((x) => x.contribution > 0)
            .sort((a, b) => b.contribution - a.contribution)
            .slice(0, 3);

        const topNeg = [...items]
            .filter((x) => x.contribution < 0)
            .sort((a, b) => a.contribution - b.contribution)
            .slice(0, 3);

        function renderKeyList(container, list, color) {
            if (!container) return;
            container.innerHTML = "";
            if (!list.length) {
                const li = document.createElement("li");
                li.textContent = currentLang === "en" ? "N/A" : "暂无";
                li.style.color = "#6b7280";
                container.appendChild(li);
                return;
            }
            list.forEach((x) => {
                const meta = getFeatureMeta(x.feature);
                const li = document.createElement("li");
                const sign = x.contribution >= 0 ? "+" : "";
                let compare = "";
                if (typeof x.value === "number" && typeof x.good_median === "number") {
                    compare = ` · ${x.value.toFixed(2)} vs ${x.good_median.toFixed(2)}`;
                }
                li.textContent = `${meta.name}: ${sign}${x.contribution.toFixed(3)}${compare}`;
                li.style.color = color;
                container.appendChild(li);
            });
        }

        renderKeyList(strengthList, topPos, "#065f46");
        renderKeyList(weaknessList, topNeg, "#991b1b");
    }
}

function fillPeers(peersResp) {
    const list = document.getElementById("peer-list");
    if (!list) return;
    list.innerHTML = "";

    const items = (peersResp && Array.isArray(peersResp.results)) ? peersResp.results : [];
    if (!items.length) {
        const li = document.createElement("li");
        li.textContent = peersResp && peersResp.message ? String(peersResp.message) : "-";
        li.style.color = "#999";
        list.appendChild(li);
        return;
    }

    items.forEach((p) => {
        const li = document.createElement("li");
        const name = p.up_name ? String(p.up_name) : `UID ${p.uid}`;
        const sim = typeof p.similarity === "number" ? p.similarity : null;
        const simText = sim === null ? "" : (currentLang === "en" ? ` (similarity ${sim.toFixed(3)})` : `（相似度 ${sim.toFixed(3)}）`);
        li.textContent = `${name} · UID ${p.uid}${simText}`;
        li.style.cursor = "pointer";
        li.addEventListener("click", () => {
            window.location.href = `/dashboard?uid=${encodeURIComponent(p.uid)}`;
        });
        list.appendChild(li);
    });
}

function showPrescriptionError(message) {
    const suggestionList = document.getElementById("suggestion-list");
    const summaryEl = document.getElementById("suggestion-summary");
    const modalSummaryEl = document.getElementById("modal-summary");
    if (suggestionList) {
        suggestionList.innerHTML = "";
        const li = document.createElement("li");
        li.textContent = t("prescriptionFailedPrefix") + message;
        li.style.color = "orange";
        suggestionList.appendChild(li);
    }

    const msgText = t("prescriptionFailedPrefix") + message;
    if (summaryEl) {
        summaryEl.textContent = msgText;
        summaryEl.style.display = "block";
    }
    if (modalSummaryEl) {
        modalSummaryEl.textContent = msgText;
    }
}

// ========= 图表 =========
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

    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }

    const labels = cols.map(featureLabel);
    const upVals = cols.map((c) => featValues[c]);
    const medVals = cols.map((c) => medianValues[c]);
    const minVals = cols.map((c) => minValues[c]);

    chartInstances[canvasId] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: t("currentUp"),
                    data: upVals,
                    backgroundColor: "rgba(37, 88, 168, 0.78)",
                    borderColor: "rgba(37, 88, 168, 1)",
                    borderWidth: 1,
                    borderRadius: 6,
                    maxBarThickness: 28,
                },
                {
                    label: t("goodMedian"),
                    data: medVals,
                    backgroundColor: "rgba(15, 118, 110, 0.62)",
                    borderColor: "rgba(15, 118, 110, 1)",
                    borderWidth: 1,
                    borderRadius: 6,
                    maxBarThickness: 28,
                },
                {
                    label: t("goodMin"),
                    data: minVals,
                    backgroundColor: "rgba(180, 83, 9, 0.5)",
                    borderColor: "rgba(180, 83, 9, 0.88)",
                    borderWidth: 1,
                    borderRadius: 6,
                    maxBarThickness: 28,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            layout: {
                padding: 0,
            },
            interaction: {
                mode: "index",
                intersect: false,
            },
            plugins: {
                legend: {
                    position: "top",
                    align: "end",
                    labels: {
                        boxWidth: 8,
                        boxHeight: 8,
                        usePointStyle: true,
                        color: "#667085",
                        font: {
                            size: 11,
                            weight: 700,
                        },
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(17, 24, 39, 0.92)",
                    padding: 10,
                    titleFont: {
                        size: 13,
                        weight: 800,
                    },
                    bodyFont: {
                        size: 12,
                    },
                    callbacks: {
                        label(context) {
                            const raw = Number(context.raw);
                            const value = Number.isFinite(raw) ? raw.toLocaleString(undefined, { maximumFractionDigits: 3 }) : context.raw;
                            return `${context.dataset.label}: ${value}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: {
                        display: false,
                    },
                    ticks: {
                        maxRotation: 0,
                        minRotation: 0,
                        color: "#667085",
                        font: {
                            size: 11,
                            weight: 700,
                        },
                    },
                },
                y: {
                    beginAtZero: true,
                    border: {
                        display: false,
                    },
                    grid: {
                        color: "rgba(102, 112, 133, 0.14)",
                    },
                    ticks: {
                        color: "#667085",
                        font: {
                            size: 11,
                        },
                        maxTicksLimit: 4,
                        callback(value) {
                            return Number(value).toLocaleString();
                        },
                    },
                },
            },
        },
    });
}

// ========= 初始化（返回按钮 / 语言切换 / 终止按钮） =========
function initDashboard() {
    // 语言应用
    applyI18n();

    // 返回按钮
    const backBtn = document.getElementById("btn-back");
    if (backBtn) {
        backBtn.addEventListener("click", () => {
            if (window.history.length > 1) {
                window.history.back();
            } else {
                window.location.href = "/";
            }
        });
    }

    // 语言切换按钮（不重新跑分析，只改文案 + 图表 legend）
    const langBtn = document.getElementById("btn-lang-toggle");
    if (langBtn) {
        langBtn.textContent = t("langToggleLabel");
        langBtn.addEventListener("click", () => {
            currentLang = currentLang === "zh" ? "en" : "zh";
            localStorage.setItem("lang", currentLang);

            // 更新带 data-i18n 的文本
            applyI18n();
            // 更新图表 legend 文案
            updateChartLegends();
            // 更新“评分区间”括号样式等
            const scoreData = window.__lastPredData;
            if (scoreData) {
                featureMetaFromServer = null;
                fillInfoPanel(scoreData);
                if (window.__lastStatsData) {
                    drawAllCharts(scoreData, window.__lastStatsData);
                }
            }
        });
    }

    // 终止按钮
    const cancelBtn = document.getElementById("btn-cancel");
    if (cancelBtn) {
        cancelBtn.addEventListener("click", () => {
            if (currentController) {
                currentController.abort();
            }
        });
    }

    // Insights modal
    const modal = document.getElementById("insights-modal");
    const openBtn = document.getElementById("btn-open-insights");
    const closeBtn = document.getElementById("btn-close-insights");

    function openModal() {
        if (!modal) return;
        modal.style.display = "flex";
    }

    function closeModal() {
        if (!modal) return;
        modal.style.display = "none";
    }

    if (openBtn) openBtn.addEventListener("click", openModal);
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeModal();
        });
    }

    // 实际加载
    loadDashboard();
}

document.addEventListener("DOMContentLoaded", initDashboard);
