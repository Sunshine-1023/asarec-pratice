# FashionRec 数据、标签与排序升级实施计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 将当前基于交易行的 `SASRecF + 规则召回 + Weighted RRF` 升级为以购物篮、point-in-time 特征、多路候选诊断和学习排序为核心的可复现推荐实验流水线。

**Architecture:** 保留现有 `src/fashionrec` 分层和 run-scoped 产物结构。先把 raw 交易转换为不会制造同日伪序列的事件/购物篮数据，再用多窗口回测冻结基线；随后加入用户、商品、用户商品交叉特征和新品/长尾召回，最后在固定候选上训练 LightGBM LambdaRank。SASRecF/TiSASRec、多兴趣和双塔模型只在前面各层已经证明瓶颈后进入实验，不作为第一步重写。

**Tech Stack:** Python 3.10+、pandas/Polars 或 DuckDB、PyArrow Parquet、NumPy、PyYAML、RecBole/PyTorch、LightGBM（必要时 CatBoostRanker 对照）、pytest。

---

## 1. 真实数据基线与边界

本计划基于 2026-08-19 对 raw 数据的扫描结果：

- `transactions_train.csv`：31,788,324 行，2018-09-20 至 2020-09-22，136.2 万交易用户，10.45 万交易 SKU。
- `customers.csv`：1,371,980 行；`FN` 缺失约 65.2%，`Active` 缺失约 66.2%，年龄缺失约 1.16%。
- `articles.csv`：105,542 个 SKU，47,224 个 `product_code` 款式，24 个商品字段，`detail_desc` 缺失约 0.39%。
- 最近 6 周中，64.8% 的用户-日期包含多个不同商品；9.26% 的交易行是同一用户-日期-SKU 的额外数量。
- 当前 4 周历史对最终测试周造成约 54.9% 的用户冷启动；改为 26 周约 17.4%，52 周约 11.1%。
- Top 30,000 SKU 约覆盖 87.7% 全量交易，但会丢掉大量新品和长尾商品。

本计划不承诺固定绝对 MAP 提升。每一项改动必须在相同回测窗口下报告平均值、标准差、分层结果和候选覆盖率，避免把单周偶然提升当成真实收益。

## 2. 总体验收标准

主指标：

- 多窗口 `MAP@12`。

必须同时报告：

- `Candidate Recall@50/100/300/500`。
- `Recall@12`、`NDCG@12`、`Hit@12`。
- warm/cold、low/medium/high activity 分层指标。
- repeat、new-to-user、new-item、long-tail 分层指标。
- 每路召回的覆盖率、独占命中率、通道 Jaccard。
- 候选数量、CPU/内存、训练和推理耗时。

阶段放行规则：

1. 新方案至少在 3 个回测窗口中的 2 个窗口不低于基线，才可成为候选默认方案。
2. 平均 `MAP@12` 未提升但 `Candidate Recall@300` 提升时，只保留为后续排序实验候选，不宣称最终推荐提升。
3. 任一主要用户分层平均 MAP 下降超过 5%，必须增加分层保护或撤回改动。
4. test 窗口只做最终一次评估，不能参与权重、特征、checkpoint 或模型选择。

---

## 阶段 0：冻结基线与数据版本

### Task 0.1：补充实验配置

**Objective:** 把数据窗口、basket 语义、候选规模、标签和回测参数写入统一配置。

**Files:**

- Modify: `configs/experiment.yaml`
- Modify: `src/fashionrec/experiment/config.py`
- Test: `tests/test_experiment_config.py`

**Add fields:**

```yaml
data:
  history_weeks: 26
  valid_weeks: 1
  test_weeks: 1
  backtest_windows: 3
  snapshot_frequency: weekly
  max_user_history: 200
  keep_full_item_universe: true
  deduplicate_user_day_item: true

label:
  horizon_days: 7
  target_mode: next_basket
  include_repeat_label: true
  include_new_to_user_label: true

candidate:
  per_channel_top_k: 200
  union_top_k: 500
  final_top_k: 12

ranking:
  enabled: false
  library: lightgbm
  objective: lambdarank
  top_k_for_training: 500
```

**Steps:**

1. 先为新字段写配置加载和默认值测试。
2. 运行 `PYTHONPATH=src python -m pytest tests/test_experiment_config.py -q`，确认旧配置兼容。
3. 实现类型校验和互斥校验，例如 `horizon_days >= 1`、`union_top_k >= final_top_k`。
4. 冻结配置快照到每次 `RunArtifacts.resolved_config`。

**Acceptance:** 任意数据准备、回测和排序命令都能从同一配置读取协议；旧配置缺少新字段时有明确默认值。

### Task 0.2：建立 raw 数据 profile 命令

**Objective:** 每次实验前自动输出原始表 schema、行数、唯一键、缺失率、时间范围和关联覆盖率。

**Files:**

- Create: `src/fashionrec/data/profile.py`
- Modify: `src/fashionrec/data/command.py`
- Modify: `src/fashionrec/cli.py`
- Test: `tests/test_data_profile.py`

**Profile checks:**

- 交易用户是否都存在于 customers。
- 交易 SKU 是否都存在于 articles。
- `article_id`、`customer_id` 是否始终按字符串读取。
- 价格是否为空、非正、异常极值。
- 交易日期是否有空值或逆序。
- `article_id -> product_code` 是否一对多合法。

**Commands:**

```bash
PYTHONPATH=src python -m fashionrec profile-data \
  --transactions data/raw/transactions_train.csv \
  --customers data/raw/customers.csv \
  --articles data/raw/articles.csv \
  --output outputs/data_profile.json
```

**Acceptance:** 大文件使用 chunks/流式统计；profile JSON 可复现且不需要把 3GB CSV 一次性装入内存。

### Task 0.3：重新生成 baseline 数据并记录 manifest

**Objective:** 明确区分 raw、filtered、processed，避免沿用旧 filtered 产物。

**Files:**

- Modify: `src/fashionrec/data/preprocess.py`
- Modify: `src/fashionrec/data/manifest.py`
- Modify: `src/fashionrec/data/command.py`
- Test: `tests/test_data_manifest.py`, `tests/test_preprocess_causality.py`

**Steps:**

1. 新 run 默认使用 raw；`--with-filter` 只能使用同一次运行刚生成的 filtered 文件。
2. manifest 记录 raw 输入哈希、处理参数、窗口边界、用户数、SKU 数和 schema 版本。
3. 新产物放入 `outputs/runs/<run_id>/data/` 或带版本的 `data/processed/<dataset_version>/`，不覆盖旧产物。
4. 先保留旧行级数据作为 baseline 对照，不立即删除。

**Acceptance:** 同一 raw + 同一配置的 manifest（忽略 generated_at）完全一致；旧产物不会被静默复用。

---

## 阶段 1：修正交易语义，构建事件与购物篮

### Task 1.1：实现同日同 SKU 聚合

**Objective:** 把重复交易行变为一个 user-day-item 事件，同时保留数量和价格信息。

**Files:**

- Create: `src/fashionrec/data/build_events.py`
- Test: `tests/test_build_events.py`
- Modify: `src/fashionrec/data/command.py`

**Output schema:**

```text
user_id
item_id
date
quantity
mean_price
min_price
max_price
sales_channel_mode
channel_count
```

**Rules:**

- 交易行同一 `customer_id, t_dat, article_id` 聚合为 `quantity`。
- 标签计算使用去重 item 集合，不能让购买数量直接重复计入 AP。
- 数量、价格、渠道只作为特征或辅助标签。
- 输出 Parquet 分区按月份或日期组织。

**Tests:**

- 重复两行变成 quantity=2。
- 不同渠道同日同 SKU 的 `channel_count=2`。
- 同日不同 SKU 保留为多个事件。
- ID 前导零不丢失。

### Task 1.2：实现按天购物篮和无伪顺序序列

**Objective:** 禁止使用 `article_id` 给同日商品制造先后顺序。

**Files:**

- Create: `src/fashionrec/data/build_baskets.py`
- Modify: `src/fashionrec/data/build_sequences.py`
- Modify: `src/fashionrec/data/preprocess.py`
- Test: `tests/test_build_sequences.py`, `tests/test_baskets.py`

**Recommended semantics:**

```text
history_baskets_before_day_D -> target basket on day_D
```

训练先实现低风险版本：同一用户同一天的所有目标商品共享同一个 `history_before_day`；同日目标之间互不泄漏。历史窗口内可保留最近 N 个购物日，而不是最近 N 行。

**Do not:**

- 不按 `article_id` 排同日事件。
- 不在 valid/test 内逐商品推进历史。
- 不把同日其它目标商品放进当前目标历史。

**Acceptance:** fixture 中同日 A/B/C 生成三个目标时，三者历史完全相同；下一个日期才可以看到 A/B/C。

### Task 1.3：构建 next-basket 标签和样本索引

**Objective:** 将购买推荐目标从“下一行商品”改为“未来 horizon 内购买商品集合”。

**Files:**

- Create: `src/fashionrec/data/labels.py`
- Create: `src/fashionrec/data/snapshots.py`
- Test: `tests/test_labels.py`, `tests/test_snapshots.py`

**Labels per snapshot:**

- `label_purchase`：未来 7 天购买。
- `label_repeat`：此前买过同 SKU 后再次购买。
- `label_new_to_user`：用户首次购买该 SKU。
- `label_same_style_new_color`：购买同 `product_code` 的新 SKU。
- `label_quantity`：未来数量，可作为辅助任务，不直接替代二值相关性。

**Acceptance:** 所有特征带 `as_of_date`；标签窗口严格在 `as_of_date` 之后；同一 user-item 在标签集合中只出现一次。

### Task 1.4：实现多窗口时间回测

**Objective:** 真正使用配置中的 `backtest_windows`，防止单个 valid 周过拟合。

**Files:**

- Create: `src/fashionrec/data/backtest.py`
- Modify: `src/fashionrec/data/split.py`
- Modify: `src/fashionrec/pipeline/orchestrator.py`
- Test: `tests/test_backtest.py`, `tests/test_no_leakage.py`

**Protocol:**

```text
window_i:
  history: [cutoff, valid_start)
  valid:   [valid_start, test_start)
  test:    [test_start, test_end)
```

每个窗口单独生成历史、标签、候选和排序数据；valid 只能选参数，test 只做最终报告。

**Acceptance:** 3 个窗口能由同一命令生成；每个窗口 manifest 写明边界；人为把 future 行传给历史特征时测试失败。

---

## 阶段 2：用户、商品和 point-in-time 特征

### Task 2.1：商品静态特征与款式层级

**Objective:** 同时支持 SKU 级和 `product_code` 款式级建模。

**Files:**

- Modify: `src/fashionrec/data/build_item_features.py`
- Create: `src/fashionrec/data/item_features.py`
- Test: `tests/test_item_features.py`

**Features:**

- 全量 24 个 articles 字段，而不是只保留 8 个类别。
- `product_code`、product_type_no、department_no 等 ID 同时保留 numeric/token 版本。
- `prod_name`、`detail_desc` 清洗后的文本长度、token 数、缺失标记。
- 颜色、图案、明暗、材质等组合字段。
- 变体数量：同 `product_code` 下 SKU 数量。

**Rules:**

- 训练商品 universe 不再默认截断到 Top 30k。
- 仅在资源不足的快速实验中显式启用 Top-item sampling。
- 未见商品 metadata 必须保留 unknown 行，不静默删除。

### Task 2.2：用户静态特征

**Objective:** 把 customers 表转成可用于冷启动和精排的静态画像。

**Files:**

- Create: `src/fashionrec/data/customer_features.py`
- Test: `tests/test_customer_features.py`

**Features:**

- age、age_bucket、age_missing。
- `club_member_status`、`fashion_news_frequency`、字段存在标记。
- `FN`、`Active` 映射为三态：1、0/空缺、missing indicator，不能均值填充。
- postal_code 只做频次/哈希分桶，禁止当作连续距离。

### Task 2.3：as-of 用户行为统计

**Objective:** 对每个 snapshot 只使用历史行为生成可审计用户特征。

**Files:**

- Create: `src/fashionrec/data/user_features.py`
- Test: `tests/test_user_features.py`, `tests/test_no_leakage.py`

**Windows:** 1d/7d/28d/84d/182d/365d。

**Features:**

- purchase count、active days、basket size。
- recency、frequency、monetary、平均购物间隔。
- 渠道 1/2 比例。
- 品类、颜色、部门、价格带分布。
- 类别熵、颜色熵、款式多样性。
- repeat rate、same-style rate、new-item rate。

**Acceptance:** 对同一用户构造两份含未来标签的输入，as-of 特征完全一致。

### Task 2.4：用户-商品交叉特征

**Objective:** 为候选排序提供个性化匹配信号。

**Files:**

- Create: `src/fashionrec/data/cross_features.py`
- Test: `tests/test_cross_features.py`

**Features:**

- 用户购买该 SKU、款式、品类、颜色的次数。
- 距离上次购买该层级的天数。
- 用户价格均值与候选价格差。
- 用户类别/颜色偏好与候选属性的匹配比例。
- 候选是否为同款新颜色、用户是否买过同款。
- 候选在用户所属人群、渠道和最近窗口中的热度。

**Acceptance:** 输出一行一候选商品，包含 `user_id,item_id,as_of_date` 主键和 feature_version。

---

## 阶段 3：召回升级与候选诊断

### Task 3.1：实现候选诊断报告

**Objective:** 在优化排序前确定 Recall 瓶颈和各通道互补性。

**Files:**

- Create: `src/fashionrec/evaluation/candidate_diagnostics.py`
- Create: `src/fashionrec/evaluation/coverage_metrics.py`
- Modify: `src/fashionrec/evaluation/experiment_report.py`
- Test: `tests/test_candidate_diagnostics.py`, `tests/test_coverage_metrics.py`

**Metrics:**

- each channel Recall/Hit@50/100/300。
- union Recall/Hit@100/300/500。
- user coverage、mean/percentile candidate count。
- channel pair Jaccard、exclusive hit rate。
- warm/cold、repeat/new-item、activity tier 分层。

**Acceptance:** 每个候选实验自动生成 JSON + CSV；没有候选覆盖报告不得进入排序比较。

### Task 3.2：升级多窗口热门召回

**Objective:** 同时覆盖短期趋势、稳定热门和冷启动用户。

**Files:**

- Modify: `src/fashionrec/recall/popular.py`
- Modify: `src/fashionrec/recall/category_popular.py`
- Test: `tests/test_popular.py`, `tests/test_category_popular.py`

**First configuration:** 1w/2w/4w/12w windows，分数先按窗口内 rank 或 z-score 归一化，再加权。

对 cold-start 增加按 `index_group_name/product_group_name/age_bucket/channel` 的人群热门，但所有聚合必须使用 as-of 历史。

### Task 3.3：升级 Item2Item 相似度

**Objective:** 降低 raw co-occurrence 的热门偏差。

**Files:**

- Modify: `src/fashionrec/recall/item2item.py`
- Modify: `src/fashionrec/recall/registry.py`
- Test: `tests/test_item2item.py`

**Experiment variants:**

1. raw co-occurrence baseline。
2. cosine/IUF normalized ItemCF。
3. time-decayed ItemCF。
4. directed sequential transition。
5. optional Swing-style score。

固定同一历史窗口和同一候选 K，单独比较 Recall@300、独占命中率和长尾覆盖。

### Task 3.4：新增复购、款式和内容召回

**Objective:** 覆盖复购、新颜色、长尾和新品。

**Files:**

- Create: `src/fashionrec/recall/repurchase.py`
- Create: `src/fashionrec/recall/style.py`
- Create: `src/fashionrec/recall/content.py`
- Modify: `src/fashionrec/recall/registry.py`
- Test: `tests/test_repurchase.py`, `tests/test_style_recall.py`, `tests/test_content_recall.py`

**First implementations:**

- Repurchase：用户-SKU 次数 + recency decay。
- Style：用户最近购买的 `product_code` 下其它 SKU/颜色。
- Content：TF-IDF/hashed text + categorical overlap；有图片后再接 CLIP/ANN。

**Acceptance:** 新品/长尾召回必须单独报告，且不能读取未来销量；无历史用户必须能走人群热门或内容冷启动 fallback。

### Task 3.5：扩大候选并集并保留证据

**Objective:** 避免 Top-300 阶段过早丢失多路一致候选。

**Files:**

- Modify: `src/fashionrec/candidates/union.py`
- Modify: `configs/experiment.yaml`
- Test: `tests/test_candidate_pipeline.py`

**Rules:**

- 每路先取 100～300。
- union 默认 500，必要时实验 1000。
- 保存 channel present、rank、score、source timestamp 和 feature version。
- 先按多路覆盖和召回证据保留，再交给排序模型，不只按某一路最小 rank。

---

## 阶段 4：学习排序

### Task 4.1：建立候选排序训练表

**Objective:** 将固定候选、point-in-time 特征和未来标签合成 LambdaRank 输入。

**Files:**

- Create: `src/fashionrec/ranking/dataset.py`
- Modify: `src/fashionrec/ranking/features.py`
- Test: `tests/test_ranking_dataset.py`, `tests/test_ranking.py`

**Schema:**

```text
user_id, item_id, snapshot_date, group_id,
label, relevance,
channel evidence,
user features,
item features,
cross features
```

正例必须来自候选集中的未来购买商品；候选集外的未来商品不能被当成排序正例，否则会把召回错误混入排序训练。

首版 relevance：purchase=1，非购买候选=0；后续有点击/加购日志再改为分级 relevance。

### Task 4.2：训练 LightGBM LambdaRank

**Objective:** 用学习排序替代手工 Weighted RRF，保留 RRF 作为安全基线。

**Files:**

- Create: `src/fashionrec/ranking/train.py`
- Create: `src/fashionrec/ranking/predict.py`
- Modify: `src/fashionrec/cli.py`
- Modify: `src/fashionrec/pipeline/orchestrator.py`
- Test: `tests/test_ranker_training.py`, `tests/test_ranker_predict.py`

**Initial settings:**

- objective `lambdarank`。
- group 为 user-snapshot。
- 训练只使用训练窗口，超参只在 valid 窗口选择。
- 先使用 100～500 棵树和 early stopping，避免大规模过拟合。
- 同一候选特征表同时保存 parquet，便于复盘 feature importance。

**Feature groups:**

- channel rank/score/present/count。
- user recency/frequency/price/channel/category preference。
- item popularity/trend/age/price/long-tail/new-item。
- user-item SKU/style/category/color/price match。

**Acceptance:** 推理端只依赖保存的 model artifact + feature schema；缺特征时有明确默认值并记录缺失率。

### Task 4.3：排序与 RRF 对照评估

**Objective:** 判断 LambdaRank 是否真的优于当前融合，而不是只拟合 valid。

**Files:**

- Modify: `src/fashionrec/evaluation/offline_eval.py`
- Modify: `src/fashionrec/evaluation/experiment_report.py`
- Test: `tests/test_experiment_report.py`

**Compare:**

1. RRF fixed weights。
2. valid-only coordinate descent RRF。
3. LambdaRank。
4. LambdaRank + business rerank（阶段 5）。

**Gate:** LambdaRank 平均 MAP@12、Recall@12 或 NDCG@12 至少一项稳定提升，且没有重大分层退化，才替换默认排序器。

---

## 阶段 5：业务重排和指标完善

### Task 5.1：实现去重、多样性和策略重排

**Objective:** 让最终 Top-12 更接近真实展示，而非只按相关性分数截断。

**Files:**

- Create: `src/fashionrec/ranking/rerank.py`
- Create: `src/fashionrec/ranking/constraints.py`
- Test: `tests/test_rerank.py`

**First policies:**

- 库存/下架过滤接口预留，当前无库存时使用 no-op policy。
- 同 `product_code` 最多 N 件。
- MMR 或类别配额控制同类重复。
- 新品、长尾、复购候选设置可配置 soft quota，不硬塞无相关商品。
- 候选不足时按人群热门、全局热门降级。

**Metrics:**

- intra-list diversity。
- category coverage。
- new-item exposure。
- long-tail exposure。
- repeat-item ratio。
- fallback rate。

### Task 5.2：增加分层和可解释报告

**Objective:** 让每个实验能回答“谁变好了、谁变差了、为什么”。

**Files:**

- Modify: `src/fashionrec/evaluation/experiment_report.py`
- Modify: `src/fashionrec/evaluation/metrics.py`
- Test: `tests/test_experiment_report.py`

**Report dimensions:**

- activity tier。
- history window。
- repeat vs new-to-user。
- warm vs new-item。
- channel 1 vs channel 2。
- age bucket / membership status（存在且样本足够时）。

---

## 阶段 6：可选模型升级（只有前面诊断证明需要时）

### Task 6.1：时间间隔序列模型

**Objective:** 判断同日修正和长历史之后，时间信息是否仍是主要瓶颈。

**Candidates:**

- TiSASRec：加入行为间隔 embedding。
- BERT4Rec：掩码序列对照，但推理需明确因果输入。
- FMLP-Rec：长序列效率对照。

**Files:**

- Create/Modify: `configs/sasrec_time.yaml`
- Create: `src/fashionrec/training/sequence_ablation.py`
- Test: `tests/test_sequence_protocol.py`

**Gate:** 只在同一候选协议和同一窗口下与 SASRecF 对比；不以 RecBole full-ranking 数字直接和 offline fusion 比较。

### Task 6.2：多兴趣/双塔召回

**Objective:** 解决用户兴趣多样和全量商品 ANN 检索问题。

**Candidates:**

- MIND/ComiRec 多兴趣用户向量。
- Two-Tower user/item encoder。
- Faiss ANN index。

**Files:**

- Create: `src/fashionrec/recall/two_tower.py`
- Create: `src/fashionrec/recall/ann_index.py`
- Create: `tests/test_ann_recall.py`

**Gate:** 先证明 `Recall@100/300` 和新品覆盖率提升，再考虑替换 SASRecF；同时记录 ANN 构建时间、索引大小、查询延迟。

---

## 阶段 7：工业化工程收尾

### Task 7.1：特征和模型版本管理

**Files:**

- Modify: `src/fashionrec/experiment/artifacts.py`
- Modify: `src/fashionrec/data/manifest.py`
- Create: `src/fashionrec/experiment/schema.py`
- Test: `tests/test_artifact_versions.py`

每个 run 记录：

- data version / raw hash。
- feature schema version。
- recall index version。
- ranker model version。
- config hash。
- selected checkpoint。

### Task 7.2：离线推荐 API 和降级路径

**Files:**

- Create: `src/fashionrec/serving/app.py`
- Create: `src/fashionrec/serving/service.py`
- Test: `tests/test_serving.py`

**Initial contract:**

```text
GET /recommendations/{user_id}?k=12
```

必须支持：

- 模型正常时多路召回 + rank + rerank。
- 用户未知时人群热门/全局热门 fallback。
- 候选为空时稳定返回非空结果或明确空响应原因。
- response 带 model/index/feature version。

这一阶段不作为离线性能改动的前置条件，可在排序模型稳定后实现。

### Task 7.3：运行监控与数据质量检查

**Files:**

- Create: `src/fashionrec/monitoring/quality.py`
- Create: `src/fashionrec/monitoring/metrics.py`
- Test: `tests/test_quality_checks.py`

最低监控：

- feature missing rate。
- candidate coverage / fallback rate。
- per-channel contribution。
- latency / error count。
- new-item/long-tail exposure。
- user/item distribution drift。

---

## 3. 建议的实验矩阵

每次只改变一个主要因素，所有实验都使用同一 3 窗口协议：

| 实验 | 变化 | 主要判断 |
|---|---|---|
| B0 | 当前 raw 行级 + RRF | 复现当前 baseline |
| B1 | 同日购物篮 + 26 周历史 | 数据语义/冷启动收益 |
| B2 | B1 + 多窗口热门 | cold-start 召回 |
| B3 | B2 + normalized/time-aware ItemCF | 个性化召回 |
| B4 | B3 + repurchase/style/content | 复购、新品、长尾 |
| B5 | B4 + 500/1000 union | 候选上限是否瓶颈 |
| B6 | B5 + LambdaRank | 学习排序收益 |
| B7 | B6 + rerank constraints | 多样性和业务约束 |
| B8 | B7 + TiSASRec/BERT4Rec | 时间序列模型收益 |
| B9 | B7 + Two-Tower/ANN | 大规模召回收益 |

每个实验都保存：

```text
resolved_config.json
manifest.json
candidate_diagnostics.json
metrics_by_window.json
metrics_by_tier.csv
feature_schema.json
model/index artifacts
```

## 4. 执行顺序与预计产物

### Milestone A：数据语义正确

完成 Task 0.1～1.4 后，应拥有：

- raw profile。
- 事件/购物篮 Parquet。
- next-basket labels。
- 3 个滚动回测窗口。
- 同日不泄漏测试。
- 当前 RRF 和行级 baseline 对照。

### Milestone B：特征与召回可诊断

完成 Task 2.1～3.5 后，应拥有：

- point-in-time user/item/cross features。
- 多窗口热门、normalized ItemCF、复购、款式、内容召回。
- 每路和 union 的 Recall@K 诊断。

### Milestone C：排序质量提升

完成 Task 4.1～5.2 后，应拥有：

- LambdaRank model artifact。
- RRF vs ranker 的多窗口对照。
- 多样性、长尾、新品和 fallback 报告。

### Milestone D：工业接口

完成 Task 6～7 后，应拥有：

- 可选时间序列/双塔/ANN 实验。
- 推荐 API、降级策略、版本追踪和质量监控。

## 5. 验证命令

每个阶段先跑局部测试，再跑全量测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q
```

数据 profile 和小 fixture smoke：

```bash
PYTHONPATH=src python -m fashionrec profile-data --help
PYTHONPATH=src python -m pytest tests/test_build_events.py tests/test_baskets.py tests/test_labels.py -q
```

回测 smoke：

```bash
PYTHONPATH=src python -m fashionrec backtest \
  --experiment-config configs/experiment.yaml \
  --output-root outputs/runs \
  --windows 1
```

正式实验必须通过：

```bash
make check
make pipeline RUN_ID=<new-run-id> STRICT=1
```

## 6. 风险、取舍与回退

- **同日购物篮 vs 行级序列：** 购物篮语义更真实，但可能降低 next-item 指标；保留 B0 作为基线，用 next-basket 指标判断，不能只看旧 RecBole 指标。
- **26/52 周历史：** 冷启动下降，但计算和序列长度上升；先用聚合长期特征，不盲目把所有历史塞入 Transformer。
- **全量商品：** 更接近新品/长尾工业场景，但训练和 full-ranking 更贵；候选阶段使用 ANN/规则，模型训练可显式采样，但不能静默删除。
- **价格特征：** 交易 price 是销售价格，不一定等于原价；命名为 sale_price，并避免声称得到真实 discount，除非有原价字段。
- **客户属性：** `postal_code` 已匿名化，不能推导真实地理关系；只做频次/哈希编码。
- **内容 embedding：** 当前仓库没有图片资产，先做文本和类别特征；图片到位后作为独立 ablation，不能把缺失图片当成全零语义。
- **LambdaRank：** 只在候选集内学习，不能修复召回缺失；候选 Recall 不足时先回到阶段 3。
- **深度模型：** 只有当特征/召回/排序基线稳定后才增加；每次仅改变一个主要因素。
- **生产化：** API、ANN、监控不应掩盖离线协议问题；先有可复现离线产物再部署。

## 7. 开始执行时的第一批动作

1. 完成 Task 0.1 配置扩展和 Task 0.2 profile 命令。
2. 运行 raw profile，保存 `outputs/data_profile.json`，再次确认本计划中的数据统计。
3. 实现 Task 1.1/1.2 的最小 fixture，不接训练、不跑全量。
4. 通过同日不泄漏测试后，再用 raw 数据生成第一版事件/购物篮 Parquet。
5. 只在新数据语义通过测试后，启动 3 窗口基线训练。

本计划不包含自动 Git commit、push 或发布；执行时每个 milestone 由用户确认后再继续。
