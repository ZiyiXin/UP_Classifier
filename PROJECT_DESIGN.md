# UP Classifier Agent 项目设计说明

## 1. 项目定位

本项目是一个面向 B 站 UP 主的商业价值分析系统。它把“数据抓取、特征构建、模型推断、可解释评分、可视化仪表盘”串成一条完整链路，让用户输入 UID 或 UP 主名称后，可以看到该账号的商业价值判断、核心影响因素、对标数据和提升建议。

项目交付形态是一个 Flask Web 应用：

- 首页负责搜索 UID 或 UP 主名称。
- 仪表盘负责展示单个 UP 的画像、评分、模型解释、图表对比和同类账号。
- 后端 API 负责抓取 B 站数据、更新本地 CSV、调用本地模型并组织解释结果。

## 2. 总体架构

```text
用户浏览器
  |
  |  搜索 / 打开 dashboard / 请求分析
  v
Flask 应用 app_1.py
  |
  |-- B 站接口抓取层
  |     |-- UP 基础信息
  |     |-- 粉丝数据
  |     |-- 投稿列表
  |     |-- 评论
  |     `-- 弹幕 XML
  |
  |-- 特征工程层
  |     |-- 播放、弹幕、评论统计
  |     |-- 视频时长统计
  |     |-- 上传频率
  |     `-- 评论重复度
  |
  |-- 模型评分层
  |     |-- scikit-learn 分类器
  |     |-- treeinterpreter 特征贡献
  |     |-- 价值分数与分位区间
  |
  |-- 数据持久层
  |     |-- database/upfile_data_labeled_10.csv
  |
  `-- 前端响应
        |-- Chart.js 图表
        |-- 双语文案
        `-- 诊断建议
```

## 3. 主要模块设计

### 3.1 Flask 后端

核心文件是 `app_1.py`。它同时承担应用入口、接口路由、B 站数据抓取、模型推断和结果整理职责。

主要职责：

- 启动时加载 CSV 数据和本地模型。
- 初始化后全量计算模型预测、置信度、贡献值、价值分数和分位。
- 接收 UID 请求，实时抓取最新 B 站数据。
- 将新抓取的特征 upsert 到 `database/upfile_data_labeled_10.csv`。
- 返回前端需要的结构化 JSON。

### 3.2 特征定义

`analysis.py` 集中定义模型输入特征 `FEATURE_COLS`，当前共有 10 个：

```text
avg_comment_scraped
avg_danmaku
avg_length
avg_play
comment_repetition
danmaku_missing_rate
med_danmaku
med_play
std_length
upload_freq
```

这些特征覆盖四类行为：

- 内容热度：平均播放量、播放中位数。
- 互动强度：平均弹幕、弹幕中位数、平均抓取评论数。
- 内容稳定性：平均视频时长、时长标准差。
- 运营质量：上传频率、弹幕缺失率、评论重复度。

### 3.3 B 站数据抓取

项目使用 `requests` 调用 B 站公开接口和 WBI 接口，不是通过解析网页 HTML 实现。

主要抓取链路：

1. `_fetch_up_profile(uid)` 获取 UP 基础资料。
2. `_fetch_relation_stat(uid)` 获取粉丝等关系数据。
3. `_fetch_user_videos(uid)` 获取最近投稿列表，默认 `VIDEOS_PER_UP = 20`。
4. `_fetch_comments_for_video(aid)` 分页抓取评论，默认每个视频最多 100 条。
5. `_fetch_danmaku_first_page(bvid)` 先查 `cid`，再抓取首 P 弹幕 XML。
6. `fetch_features_for_uid(uid)` 汇总所有视频数据并产出一行模型特征。

WBI 接口通过 `_refresh_wbi_keys_sync()`、`_mixin_key()` 和 `_wbi_sign()` 完成签名。签名 key 会缓存 10 分钟，减少重复请求。

### 3.4 模型与评分

模型文件位于 `classifier/up_classifier_10dim.pkl`，通过 `joblib.load()` 加载。

评分流程：

1. 取 `FEATURE_COLS` 对应的 10 维输入。
2. 使用分类器输出二分类预测和 `predict_proba`。
3. 根据预测类别计算 `confidence`。
4. 使用 `treeinterpreter` 得到每个特征对高商业价值类别的贡献。
5. 汇总贡献值并归一化为 `shap_norm`。
6. 计算综合分数：

```text
value_score = 100 * (0.5 * confidence + 0.5 * shap_norm)
```

7. 按全量数据中的分位数生成：

```text
Top 20%
Middle 60%
Bottom 20%
```

### 3.5 可解释建议

后端维护了 `FEATURE_META`，为每个特征提供中文名、英文名、解释和“高更好/低更好”的方向信息。

`/api/prescription/<uid>` 会返回：

- 每个特征的贡献值。
- 当前 UP 与优质 UP 中位数/最小值的对比。
- 优势 Top 3。
- 短板 Top 3。
- 提升建议。
- 可选 DeepSeek 生成的自然语言总结。

如果没有 `DEEPSEEK_API_KEY`，系统会使用规则模板生成总结，不影响核心评分链路。

### 3.6 同类 UP 对标

`find_similar_ups(uid, k=3)` 基于 `FEATURE_COLS` 做相似账号检索：

1. 对全表特征做 z-score 标准化。
2. 对当前 UID 与其他 UID 计算 cosine similarity。
3. 返回最相似的 3 个 UP。

这部分用于仪表盘中的“同类 UP 对标”模块。

### 3.7 前端设计

前端由 `templates/` 和 `static/` 组成：

- `templates/home.html`：搜索入口，支持 UID 精确搜索和名称模糊搜索。
- `templates/dashboard.html`：仪表盘页面结构。
- `static/dashboard.js`：请求 API、渲染数据、图表、双语切换、弹窗和取消请求。
- `static/dashboard.css`：页面布局、卡片、图表网格、弹窗和 loading 样式。

仪表盘布局分为：

- 顶部基础信息区：UID、UP 名、粉丝、模型分类、置信度、综合评分。
- 左侧洞察区：总结、优势/短板、同类账号。
- 右侧图表区：四组 Chart.js 柱状图，对比当前 UP、优质 UP 中位数和优质 UP 最小值。

前端双语状态保存在 `localStorage("lang")`，首页和仪表盘共用。

## 4. 数据流

### 4.1 搜索流程

```text
用户输入 query
  -> /api/search?q=<query>
  -> 如果是数字，按 UID 匹配
  -> 否则按 up_name 模糊匹配
  -> 返回候选列表
  -> 点击候选进入 /dashboard?uid=<uid>
```

### 4.2 分析流程

```text
/dashboard?uid=<uid>
  -> 前端并发请求 /api/predict/<uid> 和 /api/stats/good
  -> /api/predict/<uid> 尝试抓取最新数据
  -> upsert CSV
  -> recompute_scores()
  -> 返回模型结果、特征值、分数、元信息
  -> 前端再请求 /api/prescription/<uid> 和 /api/peers/<uid>
  -> 渲染解释、建议、同类账号和图表
```

### 4.3 容错设计

- 抓取失败但本地 CSV 已存在该 UID 时，后端可回退使用历史数据。
- 无视频数据时，接口返回业务错误，前端展示“没有可用视频数据”。
- DeepSeek 调用失败时，不影响模型结果，返回规则总结。

## 5. 技术栈

- 后端：Flask
- 数据处理：pandas、NumPy
- 模型推断：scikit-learn、joblib
- 模型解释：treeinterpreter
- HTTP 抓取：requests
- 中文文本处理：jieba
- 前端：原生 HTML/CSS/JavaScript
- 图表：Chart.js
- 可选 LLM：DeepSeek Chat API

## 6. 运行与部署注意事项

- 运行入口是 `python app_1.py`。
- 依赖安装使用 `pip install -r requirements.txt`。
- B 站接口依赖 cookie、WBI 签名、接口稳定性和访问频率，线上部署需要考虑限流和缓存。
- 当前抓取和预测发生在一次 HTTP 请求中，遇到视频较多或接口慢时页面会等待较久。
- 新 UID 分析会写回 CSV，因此部署环境需要对 `database/upfile_data_labeled_10.csv` 有写权限。

## 7. 后续改进方向

- 将抓取任务异步化，避免长时间阻塞 HTTP 请求。
- 增加请求缓存和接口限流，降低 B 站接口风控风险。
- 把模型版本、特征版本和训练数据版本绑定管理。
- 将 B 站 cookie、API key 等敏感信息完全迁移到环境变量。
- 给 `/api/predict`、`/api/prescription`、`/api/search` 增加自动化测试。
- 拆分 `app_1.py`，把爬虫、模型、评分、路由、LLM 调用拆成独立模块。
