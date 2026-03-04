# 创作者价值评分与排序系统（工程师简历写法梳理）

> 目标：把你已经写出来的代码与产物，整理成一段可直接放进工程师简历/项目经历的描述（不夸大、可落地、细节具体）。

## 1. 简历里怎么写（推荐结构）

### 1) 项目标题（1 行）
- **创作者价值评分与排序系统（B 站 UP 主）**｜数据采集 + 特征工程 + 机器学习建模 + 可解释性 + 可视化仪表盘

### 2) 项目一句话（2-3 行）
- 面向“仅用粉丝数筛选创作者导致价值失真”的问题，基于抓取到的 UP 主内容与互动数据，构建 10 维行为特征与二分类模型，并在 Web 端输出可解释的价值评分、分位区间与运营建议，辅助排序与推荐决策。

### 3) 角色与范围（不要夸大但要清晰）
- 我负责：数据抓取与特征计算、离线数据处理与聚类/建模、线上推断服务与评分逻辑、解释与可视化展示的端到端打通。

### 4) 技术栈（用你实际用到的）
- Python / Pandas / NumPy
- Flask（API + 页面渲染）
- scikit-learn（StandardScaler、KMeans、RandomForest、Pipeline）
- requests（B 站接口）、xml.etree.ElementTree（弹幕 XML 解析）
- jieba（中文分词，用于重复率指标）
- transformers + SentenceTransformers（离线情感/语义特征探索与构建）
- Chart.js（前端图表）
- treeinterpreter（树模型特征贡献解释；不是 shap 库）

## 2. 可直接放简历的项目经历（示例）

> 下面是“简历 bullet 写法”，你可按需要删减；每一条都对应到仓库里能找到的代码/产物。

**创作者价值评分与排序系统（B 站 UP 主）**｜Python / Flask / scikit-learn / Chart.js  
- 设计并实现数据采集链路：对接 B 站公开接口与 WBI 签名机制，在服务端按 UID 抓取 UP 基础信息、粉丝数、投稿列表，并进一步抓取视频评论与弹幕；弹幕使用 XML 解析落地为文本集合（`app_1.py`）。  
- 实现线上可复用的 10 维特征计算：基于最近投稿的播放/时长/评论/弹幕统计（均值、中位数、标准差），并计算更新频率、弹幕缺失率；评论重复率用 `jieba` 分词 + TopK 词占比度量（`app_1.py`, `analysis.py`）。  
- 在离线 Notebook 完成数据清洗与特征工程：从 `video_features.csv`、`all_comments.csv`、`all_danmaku.csv` 等表加载并按 UID 聚合，构建包含“语义一致性/主题多样性/情感”等扩展特征的 UP 画像表，并合并输出 `final_up_profile_v2.csv`（`data.ipynb`）。  
- 使用 PCA + KMeans 对 UP 画像做聚类探索：对特征标准化后降维可视化，计算 silhouette score 辅助选择 k，并导出带 `cluster` 的结果表用于后续标注/映射（`data.ipynb`）。  
- 训练并固化线上可用的二分类模型：在 `classifier.ipynb` 中基于 `binary_features.csv` 的 `label_binary` 标签训练 `StandardScaler + RandomForestClassifier` Pipeline，使用 stratify 的训练/测试切分、输出分类报告与混淆矩阵，并用 joblib 导出模型文件（`classifier.ipynb` → `classifier/up_classifier_10dim.pkl`）。  
- 实现“可解释评分”推断服务：Flask 启动时加载 CSV 与模型，批量计算 `predict_proba`、置信度、特征贡献（treeinterpreter），并将贡献汇总归一化后与置信度融合为 `value_score`（0-100），同时计算全表百分位与 Top/Middle/Bottom 分桶（`app_1.py`）。  
- 构建可视化仪表盘：前端以 Chart.js 展示 4 组特征对比（当前 UP vs“优质 UP”中位数/最小值），提供中英切换、加载进度与取消请求（AbortController）等交互（`templates/`, `static/`）。  
- 增加可选的 AI 文案总结能力：后端根据特征贡献与规则化建议拼接提示词，若环境变量存在 `DEEPSEEK_API_KEY` 则调用 DeepSeek 生成摘要，否则自动降级为空（`app_1.py`）。  

## 3. 工程细节展开（面试时可讲的“怎么做”）

### 3.1 线上系统架构（你现在的实现）
- 入口：Flask 服务（`app_1.py`），首页输入 UID（`templates/home.html`）→ 仪表盘（`templates/dashboard.html`）。  
- 预测链路：前端请求 `/api/predict/<uid>`；后端在该请求内完成抓取、特征计算、写入/更新 CSV、全表重算评分，并返回该 UID 的预测与特征。  
- 解释链路：前端随后请求 `/api/prescription/<uid>`；后端基于 treeinterpreter 输出每个特征贡献值，生成建议列表与（可选）LLM 总结。  
- 对照基线：`/api/stats/good` 从 `label_binary == 1` 的子集计算各特征 median/min，供前端图表对比使用。  

### 3.2 爬虫/数据采集怎么做（`app_1.py`）
- WBI 签名：通过 `/x/web-interface/nav` 获取 `img_key/sub_key`，按固定表 `_MIXIN_KEY_ENC_TAB` 混淆生成 mixin key，并对排序后的 query 做 md5 得到 `w_rid`；用于访问 `x/space/wbi/*` 接口。  
- UP 基础信息与粉丝：分别通过 `x/space/wbi/acc/info` 与 `x/relation/stat` 获取；粉丝字段做了容错回退。  
- 投稿列表：`x/space/wbi/arc/search` 分页抓取，按 `order=pubdate` 获取最近投稿；当前实现默认每个 UP 抓 `VIDEOS_PER_UP = 20` 条。  
- 评论抓取：`x/v2/reply` 取热评/主评论分页抓取，默认每个视频最多 `MAX_COMMENTS_PER_VIDEO = 100` 条文本。  
- 弹幕抓取：先用 `x/web-interface/view` 获取分 P 信息与 `cid`，再拉 `x/v1/dm/list.so`，使用 `xml.etree.ElementTree` 提取 `<d>` 节点文本。  
- 节流：在分页/逐视频抓取中加入 `time.sleep`，降低被限流概率（这是你现在的“简单可用”策略）。  

### 3.3 10 维线上特征如何定义（`analysis.py` + `app_1.py`）
线上推断只依赖下列 10 维（`FEATURE_COLS`）：  
- `avg_comment_scraped`：每个视频抓取到的评论条数均值（注意：这里是“抓到的数量”，不是全站真实评论数）。  
- `avg_danmaku` / `med_danmaku`：每个视频弹幕条数的均值/中位数。  
- `avg_length` / `std_length`：视频时长（秒）的均值/总体标准差（`statistics.pstdev`）。  
- `avg_play` / `med_play`：播放量的均值/中位数。  
- `upload_freq`：抓取到的视频发布时间跨度内的投稿频率（视频数/天）。  
- `danmaku_missing_rate`：弹幕为 0 的视频占比。  
- `comment_repetition`：把所有评论做 `jieba.lcut` 分词，统计 top_k（默认 30）词频总和占全部 token 的比例。  

### 3.4 离线特征工程与聚类（`data.ipynb`）
你在 `data.ipynb` 里做的事情可以在面试里按“原始数据 → 结构化特征 → 聚类探索”讲清楚：  
- 多表加载与聚合：读取 `video_features.csv`、`all_comments.csv`、`all_danmaku.csv`，并按 UID 聚合产出 UP 级别特征表。  
- 重复率特征：生成 `comment_repetition.csv` / `danmaku_repetition.csv`（jieba 分词 + topK 占比）。  
- 语义一致性（cohesion）特征：用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 编码文本/标题，计算“与中心向量的平均余弦相似度”作为同质度指标（生成 `title_cohesion.csv`、`comment_cohesion.csv`、`danmaku_cohesion.csv`）。  
- 主题多样性特征：对评论/弹幕做 TF-IDF（自定义 jieba tokenizer）后用 `MiniBatchKMeans` 聚类，计算聚类分布熵（entropy）衡量话题多样性，输出 `comment_topic.csv`、`danmaku_topic.csv`。  
- 情感特征：用 `transformers` 加载中文二分类情感模型 `uer/roberta-base-finetuned-jd-binary-chinese`，对全量评论/弹幕生成 sentiment 分数并按 UID 聚合（对应 `comment_sentiment.csv`、`danmaku_sentiment.csv`）。  
- 特征汇总：把上述多张特征表与 `up_profile.csv` 合并，输出 `final_up_profile_v2.csv`，作为聚类与后续标注/训练的基础。  
- 聚类探索：对选定特征做 StandardScaler 标准化、PCA 2D 可视化、KMeans 聚类，并用 silhouette score 辅助选择 k；导出带 cluster 的结果表（例如 `final_cluster_d15_k4.csv` 等），并提供按 cluster 打印 UP 列表的辅助代码便于人工审阅。  

### 3.5 标签构造与分类器训练（`classifier.ipynb`）
你目前的训练方式可以如实描述为“聚类辅助 + 人工规则映射 → 二分类训练”：  
- 标签构造：从聚类结果 CSV（如 `final_cluster_k4.csv`）读取 `cluster`，在 Notebook 中用规则把部分 cluster 映射为 `label_binary`（高/低价值），并导出 `less_features_binary.csv`（`classifier.ipynb`）。  
- 训练数据：读取 `binary_features.csv`，选择 10 维特征训练二分类器（Notebook 输出显示训练集约 121 行）。  
- 模型：`Pipeline(StandardScaler + RandomForestClassifier)`，参数包括 `n_estimators=300`、`class_weight='balanced'`、`random_state=42`。  
- 评估与导出：`train_test_split(test_size=0.25, stratify=y)`；输出 `classification_report` 与 `confusion_matrix`；`joblib.dump` 固化为 `up_classifier_10dim.pkl`，并在服务端加载使用。  

### 3.6 线上评分与解释（`app_1.py`）
- 推断：`predict_proba` 得到正类概率；`predict` 得到类别。  
- 置信度：若预测为高价值则取 `prob_high`，否则取 `1 - prob_high`。  
- 解释：用 treeinterpreter `ti.predict` 拿到每个特征对预测的贡献（contributions）。项目里把贡献汇总字段命名为 `shap_sum/shap_norm`，但实现上不是 shap 库，而是 treeinterpreter 的贡献分解。  
- 综合评分：把“置信度”和“贡献汇总归一化”各占 0.5 融合为 `value_score = 100 * (0.5 * confidence + 0.5 * shap_norm)`，并基于全表计算百分位与区间（Top 20% / Middle 60% / Bottom 20%）。  
- 数据更新：对新 UID 抓取完特征后 upsert 进内存 df 并写回 `database/upfile_data_labeled_10.csv`，随后对全表重算预测/评分（实现简单直观，但对数据量大时会有性能压力）。  

## 4. 你可以主动交代的“边界与不足”（加分但要克制）
- 抓取依赖 Cookie/WBI：`app_1.py` 中当前写死了 `BILI_COOKIE`，且 B 站接口可能限流/变更；这是可运行 demo 的选择，工程化可改为配置化与更完善的容错/重试/缓存。  
- 线上特征与离线特征池不完全一致：离线构建了更多 NLP/语义特征用于探索与聚类，但线上服务目前只计算 10 维可稳定抓取的统计特征，以控制复杂度与耗时。  
- 重算策略偏重：每次插入新 UID 会全表重算评分（`recompute_scores()`）；如果数据规模扩大，可改为增量计算/异步任务队列。  

## 5. 简历落款小技巧（可选）
- 如果你投“算法工程/数据工程/后端”不同岗位，可以做三种版本：  
  - 算法工程：强调聚类→标签→RandomForest→解释→线上化闭环。  
  - 数据工程：强调多源采集、清洗、聚合、特征表产出与版本管理。  
  - 后端/全栈：强调 Flask API、数据更新、可视化仪表盘与交互体验（加载、取消、多语言）。  

