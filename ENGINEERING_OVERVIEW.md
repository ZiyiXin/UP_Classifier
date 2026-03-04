# 创作者价值评分与排序系统（工程视角说明）

> 这是基于当前项目代码与资源整理的工程说明，侧重“做了什么、怎么做、用了什么技术”。

## 1. 工程目标与交付能力
- 目标：构建一个“创作者价值评分与排序”系统，提供可解释的模型判断、分数区间与运营建议。
- 交付形式：一个可交互的 Web 仪表盘 + 后端 API 服务，支持输入 B 站 UID 触发数据抓取与评分。

## 2. 端到端流程（线上运行链路）
1) 用户输入 UID → 前端发起预测请求（`/api/predict/<uid>`）。
2) 后端抓取 B 站数据：UP 信息、粉丝数、视频列表、评论、弹幕。
3) 计算 10 维核心特征（平均播放、平均弹幕、评论重复率、更新频率等）。
4) 使用本地模型进行二分类与概率预测，计算综合“价值分数”。
5) 用 treeinterpreter 生成特征贡献（类 SHAP 解释），给出原因与建议。
6) 前端展示基础信息、评分、分位、4 组图表，以及 AI 总结建议。

## 3. 核心模块与职责
### 3.1 数据抓取与特征构建（`app_1.py`）
- B 站 API 接入：
  - 用户信息：`/x/space/wbi/acc/info`
  - 粉丝数据：`/x/relation/stat`
  - 视频列表：`/x/space/wbi/arc/search`
  - 评论：`/x/v2/reply`
  - 弹幕：`/x/v1/dm/list.so`
- 访问鉴权：实现 WBI 签名流程（`_refresh_wbi_keys_sync` + `_wbi_sign`）。
- 特征计算：
  - 播放/弹幕/评论统计：均值、中位数、标准差
  - 评论重复率：`jieba` 分词 + TopK 词占比
  - 更新频率：时间跨度内上传数/天
  - 弹幕缺失率

## 3.x B 站数据抓取逻辑（工程实现细节）
你当前实现的“爬虫”本质是 **调用 B 站的 HTTP 接口获取结构化数据（JSON / XML）**，而不是去抓网页 HTML 再解析 DOM。

### 抓取方式：公开接口 + 需要签名的 WBI 接口
- **请求库**：`requests`（`_bili_get_json()` 统一封装 GET + 超时 + code 检查）。  
- **鉴权/上下文**：在请求中带 `BILI_HEADERS`（UA/Referer/Accept）与 `BILI_COOKIE`（`SESSDATA`/`bili_jct`/`buvid3`）。  
- **两类接口**：  
  - **WBI 接口（需要签名）**：例如获取 UP 信息、投稿列表（`/x/space/wbi/...`）。  
  - **非 WBI 接口（不走签名或签名要求不同）**：例如粉丝统计、评论接口、视频信息接口等（`/x/relation/stat`、`/x/v2/reply`、`/x/web-interface/view`）。  

### WBI 签名怎么做（`_refresh_wbi_keys_sync()` / `_wbi_sign()` / `_mixin_key()`）
为了调用 `x/space/wbi/*` 系列接口，你实现了一套 WBI 签名流程：
1) **拿 WBI keys**：请求 `https://api.bilibili.com/x/web-interface/nav`，从返回的 `wbi_img.img_url` / `wbi_img.sub_url` 提取 `img_key`、`sub_key`。代码里做了 **10 分钟缓存**（`_WBI_KEYS_TS`，避免每次都刷新）。  
2) **生成 mixin key**：把 `img_key + sub_key` 拼接后按 `_MIXIN_KEY_ENC_TAB` 做重排截断，得到 32 位 `mixin_key`。  
3) **清洗参数 + 加时间戳**：把请求参数中的特殊字符 `! ' ( ) *` 去掉，并补上 `wts=当前秒级时间戳`。  
4) **计算 w_rid**：对排序后的 querystring 拼接 `mixin_key` 做 md5，得到 `w_rid`。  
5) **带签名请求**：最终把 `wts` / `w_rid` 加回 params 发起请求。  

### 单个 UID 的抓取链路（`fetch_features_for_uid()`）
这条链路由 `/api/predict/<uid>` 触发（后端在一次请求内完成抓取 + 计算 + 写回）：
1) **UP 基础资料**：`_fetch_up_profile(uid)` → `x/space/wbi/acc/info`（WBI 签名）。  
2) **粉丝/关注统计**：`_fetch_relation_stat(uid)` → `x/relation/stat`。  
3) **投稿列表**：`_fetch_user_videos(uid, limit=VIDEOS_PER_UP)` → `x/space/wbi/arc/search`（WBI 签名，按 `pubdate` 拉最近视频；默认 `VIDEOS_PER_UP=20`，分页循环直到够数或无更多数据）。  
4) **逐视频补充信息并汇总**：对每条投稿：
   - 时长解析：`_parse_length_to_seconds("mm:ss"/"hh:mm:ss")` → 秒。  
   - 评论抓取：`_fetch_comments_for_video(aid, max_comments=MAX_COMMENTS_PER_VIDEO)` → `x/v2/reply` 分页取评论文本（默认最多 100 条/视频）。  
   - 弹幕抓取：`_fetch_danmaku_first_page(bvid)`：  
     - 先调用 `x/web-interface/view` 获取 `cid`（首 P）。  
     - 再请求 `x/v1/dm/list.so?oid=<cid>`，拿到 **XML**，用 `xml.etree.ElementTree` 解析 `<d>` 文本列表。  
5) **节流**：在分页/逐视频抓取中用 `time.sleep(0.5~1s)` 做简单限速，降低风控/限流概率。  
6) **输出为特征行**：聚合成 10 维统计特征 + 基础字段（uid、up_name、followers），供模型推断使用。  

### 抓取失败时的行为（线上可用性）
- `_bili_get_json()` 会检查返回 `code`，非 0 会打印告警；网络/HTTP 错误会 `raise_for_status()` 抛异常。  
- `/api/predict/<uid>` 的策略是：**先尝试抓取并 upsert**；如果抓取失败但 CSV 里已有该 UID 的旧记录，则 **回退使用旧记录** 做展示（保证页面可用）。  

### 关键结论（你可以对外怎么说）
- 你的实现是“**基于接口的抓取**”（JSON + XML），不是爬网页 HTML。  
- WBI 接口需要签名，你在代码里实现了签名与 key 刷新缓存；评论/弹幕走不同接口，弹幕需要额外 XML 解析。  

### 3.2 模型推断与评分（`app_1.py`, `analysis.py`）
- 模型加载：`classifier/up_classifier_10dim.pkl`（scikit-learn 模型）
- 输入特征：`analysis.py` 中定义的 `FEATURE_COLS`（10 维）
- 评分规则（工程实现）：
  - 预测概率 → 置信度（高类概率或低类反向）
  - treeinterpreter 贡献值归一化（`shap_norm`）
  - 综合得分：`value_score = 100 * (0.5 * confidence + 0.5 * shap_norm)`
  - 排名区间：按百分位划分 Top/Middle/Bottom

### 3.3 解释与建议生成（`app_1.py` + `static/dashboard.js`）
- 解释：使用 treeinterpreter 输出每个特征的贡献值。
- 建议：根据贡献值阈值生成“保持/优化/灵活调整”模板。
- 可选 AI 总结：调用 DeepSeek（`DEEPSEEK_API_KEY`）生成一句话结论与建议摘要。

### 3.4 Web 可视化与交互（`templates/`, `static/`）
- 前端页面：
  - 首页输入 UID（`templates/home.html`）
  - 评分仪表盘（`templates/dashboard.html`）
- 图表展示：Chart.js 生成 4 个特征组对比图（当前 UP vs 优质 UP 中位数/最小值）。
- 双语支持：前端本地切换中文/英文，后端处方接口支持 `lang` 参数。

## 4. 离线分析与建模资产
- `data.ipynb` / `archive/data.ipynb`：用于探索、特征工程、聚类/建模实验（包含 KMeans 相关内容）。
- `archive/data.py`：用于从标注数据中整理最终训练集字段。
- `database/`：保存标注数据与线上使用的特征表（CSV）。

## 5. 技术栈
- 后端：Flask、Pandas、NumPy、scikit-learn、joblib
- 解释：treeinterpreter（树模型贡献度）
- 爬虫/接口：requests、WBI 签名
- 文本处理：jieba
- 前端：HTML/CSS/JS、Chart.js
- 可选 LLM：DeepSeek Chat API

## 6. 运行入口与关键文件
- 服务入口：`app_1.py`
- 特征定义：`analysis.py`
- 模型文件：`classifier/up_classifier_10dim.pkl`
- 数据库 CSV：`database/upfile_data_labeled_10.csv`
- 前端页面：`templates/home.html`, `templates/dashboard.html`
- 前端脚本与样式：`static/dashboard.js`, `static/dashboard.css`

## 7. 当前实现中的工程特性
- 特征与模型完全本地化：线上推断不依赖外部服务（除可选 LLM）。
- 支持数据更新：新 UID 会触发抓取并写回 CSV，再全量重算评分。
- 可解释性：为每个特征输出贡献值并与评分联动。
- 容错：若抓取失败但 CSV 中已有记录，会回退到历史数据。

## 8. 可能的后续工程化方向（可选）
- API 限流与缓存策略（B 站接口频率与稳定性）。
- 模型版本化与可回滚（模型与特征版本绑定）。
- 任务队列化（爬虫与推断分离，避免请求阻塞）。
- 统一日志与监控（抓取失败、超时、模型输入异常）。
