# FashionRec-Transformer 系统优化实施计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 将现有“SASRecF + 四路手工加权融合”升级为可复现、可诊断、可回测的“两阶段推荐系统：多路候选召回 + 学习排序”，以稳定提升跨时间窗口的 Offline MAP@12 为目标。

**Architecture:** 保留当前 `hm` 时间切分、SASRecF、Popular、Category Popular、Item2Item 和离线评估主干；先补齐实验协议、指标诊断与测试，再扩展复购、多窗口趋势和改进 Item2Item 等候选通道，最后使用 CatBoostRanker 学习融合。SASRecF 的进一步调优放在召回与排序稳定之后，避免在无法定位瓶颈时盲目增加模型复杂度。

**Tech Stack:** Python 3.11、pandas、NumPy、PyTorch、RecBole、SASRec/SASRecF、CatBoost、pytest、YAML/JSON。

---

## 1. 当前基线与计划边界

### 1.1 当前主链路

```text
filter / preprocess
  -> 4 周 train + 1 周 valid + 1 周 test
  -> hm_seq 序列样本 + item 类别特征
  -> SASRecF Top-100
  -> Popular / Category Popular / Item2Item
  -> 按用户活跃度加权倒数排名融合
  -> Top-12
  -> MAP@12 / Recall@12 / NDCG@12 / Hit@12
```

### 1.2 已知历史结果

- SASRec test MAP@12：约 `0.0154`。
- SASRecF test MAP@12：约 `0.0156`。
- 四路融合搜权后 test MAP@12：约 `0.0205`。
- 历史结果说明：当前阶段的主要增益来自候选召回和融合，而不是 SASRecF 相对 SASRec 的模型升级。

这些数字只作为历史参考，不作为当前数据版本的验收基线。实施阶段必须重新生成同一数据协议下的基线。

### 1.3 本计划不做的事情

- 不在第一阶段更换为大型自定义 Transformer。
- 不使用最终 test 调权重、选特征或选择模型。
- 不把 RecBole full-ranking 指标与 Offline candidate-ranking 指标直接横向比较。
- 不优先建设在线服务、FastAPI 或部署链路。
- 不承诺固定的绝对 MAP 提升；所有结论以多窗口回测为准。
- 不在未获得用户授权时执行 Git commit、push 或其他 Git 修改。

### 1.4 优化成功的统一标准

主指标：

- 多时间窗口平均 `MAP@12`。

辅助指标：

- `Candidate Recall@50/100/300`。
- `Candidate HitRate@50/100/300`。
- `Recall@12`、`NDCG@12`、`Hit@12`。
- SASRecF 召回用户覆盖率。
- 各活跃度层级的用户数、MAP@12 和候选 Recall。
- 每个召回通道独立指标及通道交集率。
- 推理耗时、峰值内存、候选数量。

建议的阶段放行标准：

- 新方案至少在 3 个回测窗口中的 2 个窗口提升 MAP@12。
- 平均 MAP@12 相对基线提升至少 2%，才进入下一阶段或成为默认方案。
- 任一主要用户层级 MAP@12 下降超过 5% 时，必须解释原因或增加保护策略。
- 候选通道若不提升 Recall@300、覆盖率或互补性，不进入主流水线。

---

## 2. 目标目录结构

计划完成后，推荐形成以下结构：

```text
configs/
  experiment.yaml
  sasrec.yaml
  sasrecf.yaml
  sasrecf_sweeps/

src/
  data/
    backtest.py
    manifest.py
    preprocess.py
    split.py
  recall/
    popular.py
    repurchase.py
    category_popular.py
    category_transition.py
    item2item.py
    registry.py
  ranking/
    __init__.py
    candidate_dataset.py
    features.py
    train.py
    predict.py
  evaluate/
    metrics.py
    candidate_diagnostics.py
    experiment_report.py
    offline_eval.py
    weight_search.py

tests/
  fixtures/
  test_item_id.py
  test_time_split.py
  test_metrics.py
  test_fusion.py
  test_recall.py
  test_ranking_features.py
  test_no_leakage.py

run_backtest.py
run_candidate_diagnostics.py
run_ranker.py
run_build_submission.py
```

---

# 阶段 0：冻结实验协议与基线

## Task 1：建立统一实验配置

**Objective:** 把散落在代码中的时间窗口、Top-K、用户分层和随机种子收敛为一份实验配置，保证不同实验只改变明确参数。

**Files:**

- Create: `configs/experiment.yaml`
- Create: `src/experiment/__init__.py`
- Create: `src/experiment/config.py`
- Test: `tests/test_experiment_config.py`
- Modify: `run_pipeline.py`

**建议配置字段：**

```yaml
experiment:
  name: fashionrec_v3
  seed: 2026

data:
  history_weeks: 4
  valid_weeks: 1
  test_weeks: 1
  backtest_windows: 3
  max_user_history: 100
  min_user_purchases: 5

candidate:
  per_channel_top_k: 100
  union_top_k: 300
  final_top_k: 12

evaluation:
  primary_metric: MAP@12
  activity_tiers:
    cold_start: [0, 0]
    low: [1, 2]
    medium: [3, 9]
    high: [10, null]
```

**Steps:**

1. 写测试，验证配置加载后字段类型、必填项和默认值正确。
2. 运行 `pytest tests/test_experiment_config.py -v`，确认测试先失败。
3. 实现只读配置加载器，不在这一任务中重构全部旧模块。
4. 再次运行测试，预期全部通过。
5. 在 `run_pipeline.py` 增加 `--experiment-config`，只负责传递配置路径。
6. 运行 `python run_pipeline.py --help`，确认现有参数保持兼容。

**Acceptance:** 同一份 YAML 能明确描述数据、候选和评估协议；旧命令仍可使用。

---

## Task 2：生成数据与实验清单

**Objective:** 每次实验保存可核对的数据版本、时间范围和代码参数，解决“报告数字与当前数据不一致”的问题。

**Files:**

- Create: `src/data/manifest.py`
- Create: `tests/test_data_manifest.py`
- Modify: `run_data_prep.py`
- Output: `data/processed/manifest.json`

**Manifest 至少包含：**

- 原始/过滤数据路径及 SHA256。
- 每个文件的行数、用户数、商品数。
- 最小/最大日期。
- train/valid/test 的时间边界。
- 预处理参数。
- 生成时间和项目 Git SHA；Git SHA 只读获取，不执行 Git 修改。

**Steps:**

1. 使用极小 TSV fixture 写 manifest 测试。
2. 验证时间范围、行数和哈希稳定。
3. 实现流式统计，避免为了算 manifest 一次性加载 3GB 文件。
4. 在数据准备成功结束后生成 manifest。
5. 再次运行相同输入，确认除生成时间外的内容一致。

**Acceptance:** 任意实验报告都能指回唯一数据快照。

---

## Task 3：拆出统一指标模块并建立单元测试

**Objective:** 为 MAP@K 等核心指标建立独立、可验证的唯一实现。

**Files:**

- Create: `src/evaluate/metrics.py`
- Create: `tests/test_metrics.py`
- Modify: `src/evaluate/offline_eval.py`

**关键测试样例：**

```python
def test_map_at_12_respects_rank_and_unique_targets():
    actual = {"1", "2"}
    pred = ["1", "9", "2"]
    assert map_at_k(actual, pred, 12) == pytest.approx((1.0 + 2 / 3) / 2)


def test_map_at_k_empty_target_is_zero():
    assert map_at_k(set(), ["1"], 12) == 0.0
```

还需覆盖：

- 重复预测商品。
- 真实标签数量大于 K。
- 预测不足 K。
- 无命中。
- item ID 前导零规范化前后的一致性。

**Steps:**

1. 先为现有指标行为写测试并确认通过/暴露差异。
2. 将 `_map_at_k`、`_recall_at_k`、`_ndcg_at_k`、`_hit_at_k` 移入统一模块。
3. `offline_eval.py` 只导入统一实现。
4. 运行 `pytest tests/test_metrics.py -v`。

**Acceptance:** 项目内不存在第二套含义不同的 MAP@12 实现。

---

## Task 4：建立时间切分和防泄漏测试

**Objective:** 证明所有召回索引、特征和排序训练只使用预测时刻之前的数据。

**Files:**

- Create: `tests/test_time_split.py`
- Create: `tests/test_no_leakage.py`
- Modify: `src/data/split.py`

**必须验证：**

- `max(train.timestamp) < min(valid.timestamp)`。
- `max(valid.timestamp) < min(test.timestamp)`。
- valid 召回索引只使用 train。
- test 召回索引只使用 train + valid。
- 排序特征中的商品热度和用户偏好不读取标签周。
- 同一天相同时间戳的排序规则确定，不依赖 pandas 当前行顺序。

**Acceptance:** 测试中人为把未来记录混入历史时，测试必须失败。

---

## Task 5：重新建立当前代码基线

**Objective:** 在统一协议下生成之后所有优化要比较的基线报告。

**Files:**

- Create: `src/evaluate/experiment_report.py`
- Create: `run_baseline.py`
- Output: `outputs/experiments/<run_id>/manifest.json`
- Output: `outputs/experiments/<run_id>/metrics.json`
- Output: `outputs/experiments/<run_id>/per_tier_metrics.csv`

**必须记录的基线：**

1. Popular 单通道。
2. Category Popular 单通道。
3. Item2Item 单通道。
4. SASRecF 单通道。
5. 当前默认权重融合。
6. 当前 valid 权重搜索融合。

**Commands:**

```bash
pytest tests/test_metrics.py tests/test_time_split.py tests/test_no_leakage.py -v
python run_baseline.py --config configs/experiment.yaml --eval-split valid
python run_baseline.py --config configs/experiment.yaml --eval-split test --weights-json <valid_best_weights>
```

**Acceptance:** test 不参与权重搜索；报告能展示整体和四个活跃度层级的指标。

---

# 阶段 1：候选召回诊断与扩展

## Task 6：实现候选集诊断工具

**Objective:** 在改算法前先回答“真实商品有没有进入候选集、哪个通道贡献最大”。

**Files:**

- Create: `src/evaluate/candidate_diagnostics.py`
- Create: `run_candidate_diagnostics.py`
- Create: `tests/test_candidate_diagnostics.py`

**输出指标：**

- 每通道 Recall@50/100。
- 通道联合 Recall@100/300。
- 每用户平均候选数。
- 用户覆盖率。
- 通道两两 Jaccard 重合率。
- 独占命中率：只有该通道召回到真实商品的比例。
- 按 high/medium/low/cold_start 分层统计。

**Acceptance:** 能清楚判断新增通道是带来互补候选，还是仅重复热门商品。

---

## Task 7：增加复购召回通道

**Objective:** 显式利用 H&M 场景中的重复购买行为。

**Files:**

- Create: `src/recall/repurchase.py`
- Create: `tests/test_repurchase.py`
- Modify: `src/recall/__init__.py`
- Modify: `src/recall/rule_recall_export.py`
- Modify: `src/evaluate/offline_eval.py`

**初始评分公式：**

```text
repurchase_score(item)
  = log1p(user_purchase_count)
    * exp(-days_since_last_purchase / tau)
    * (1 + recent_item_popularity)
```

首轮只搜索 `tau ∈ {7, 14, 28}`，避免过度调参。

**Tests:**

- 最近购买且购买次数多的商品排名更高。
- 未来购买不能进入历史。
- 同商品多次购买只输出一个候选。
- 冷启动用户返回空列表，由 Popular 兜底。

**Experiment:**

- Repurchase 单通道。
- 原四路融合。
- 四路 + Repurchase。
- 分层观察 high/medium 用户是否获益。

**Gate:** 若平均 MAP 或候选独占命中没有提升，则保留为实验模块但不进入默认流水线。

---

## Task 8：增加多时间窗口趋势热门召回

**Objective:** 同时捕捉短期趋势和稳定热门，适应时尚商品快速变化。

**Files:**

- Modify: `src/recall/popular.py`
- Create: `tests/test_popular_windows.py`

**候选分量：**

- 最近 1 周热门。
- 最近 2 周热门。
- 最近 4 周热门。
- 全历史窗口热门。

**首轮融合：**

```text
score = 0.40 * pop_1w + 0.30 * pop_2w + 0.20 * pop_4w + 0.10 * pop_all
```

每个分量先按 rank 或最大值归一化，不能直接混合不同量纲的原始购买次数。

**Acceptance:** 在至少两个回测窗口提升 cold_start/low 的 Candidate Recall@100，且整体覆盖率不下降。

---

## Task 9：升级 Item2Item 相似度

**Objective:** 降低热门商品对共现次数的垄断，提高个性化和互补性。

**Files:**

- Modify: `src/recall/item2item.py`
- Create: `tests/test_item2item_similarity.py`

**需要比较的相似度：**

```text
raw_cooccurrence
cosine = cooccur(a,b) / sqrt(count(a) * count(b))
jaccard = cooccur(a,b) / (count(a) + count(b) - cooccur(a,b))
```

**额外权重：**

- 同一天购物篮共现权重更高。
- 共现时间越近权重越高。
- 用户历史中越近的 seed item 权重越高。
- 对超热门商品增加 `1 / log1p(item_count)` 惩罚。

**实验矩阵：**

| 方案 | 相似度 | 时间衰减 | 热门惩罚 |
|---|---|---|---|
| A | raw | 否 | 否 |
| B | cosine | 否 | 否 |
| C | cosine | 是 | 否 |
| D | cosine | 是 | 是 |

**Gate:** 选择跨窗口平均 Candidate Recall@100 和 MAP@12 最好的方案，而不是只看单个 valid。

---

## Task 10：增加类别转移召回

**Objective:** 捕捉“购买某类商品后，下一步更可能购买另一类商品”的互补关系。

**Files:**

- Create: `src/recall/category_transition.py`
- Create: `tests/test_category_transition.py`
- Modify: `src/recall/rule_recall_export.py`

**流程：**

1. 将用户历史按日期排序。
2. 统计相邻购买的 `category_a -> category_b` 转移。
3. 对转移概率做最小支持度过滤和平滑。
4. 根据用户最近 3～5 个类别预测后续类别。
5. 在预测类别内使用近期热门商品召回。

**防泄漏要求:** valid 的类别转移矩阵只允许使用 train，test 只允许使用 train + valid。

**Gate:** 新通道必须提供可观的独占命中，否则不进入排序模型。

---

## Task 11：建立召回通道注册表

**Objective:** 避免每增加一个通道就在 `offline_eval.py` 复制一段硬编码逻辑。

**Files:**

- Create: `src/recall/registry.py`
- Create: `tests/test_recall_registry.py`
- Modify: `src/evaluate/offline_eval.py`
- Modify: `src/recall/rule_recall_export.py`

**接口约定：**

```python
class RecallChannel(Protocol):
    name: str

    def build(self, history_paths: list[Path]) -> None: ...

    def recall(self, user_id: str, history: list[str], top_k: int) -> list[tuple[str, float]]: ...
```

**Acceptance:** 增删通道只需修改配置或注册表，不需要复制整段评估循环。

---

# 阶段 2：多窗口回测

## Task 12：实现滚动时间窗口生成器

**Objective:** 用多个预测周判断优化是否稳定，而不是只在单个 valid 周有效。

**Files:**

- Create: `src/data/backtest.py`
- Create: `tests/test_backtest_windows.py`
- Create: `run_backtest.py`

**推荐协议：**

若总数据至少覆盖 8 周：

```text
Window 1: history weeks 1-4 -> label week 5
Window 2: history weeks 2-5 -> label week 6
Window 3: history weeks 3-6 -> label week 7
Final test: later fixed week，仅在方案冻结后运行
```

每个窗口都必须重新构建：

- Popular 索引。
- 类别热门索引。
- Item2Item 索引。
- 类别转移矩阵。
- 用户偏好特征。

不能复用包含未来数据的全局索引。

**Output:**

- `metrics_by_window.csv`
- `metrics_mean_std.json`
- `per_tier_by_window.csv`

**Acceptance:** 同一方案能够输出 3 个窗口的均值、标准差和分层结果。

---

## Task 13：建立实验放行表

**Objective:** 用统一表格决定哪些召回通道进入下一阶段。

**Files:**

- Create: `docs/optimization_experiment_log.md`
- Create: `outputs/experiments/summary.csv`（运行时生成）

**表格字段：**

```text
run_id
data_manifest_sha
change
window
MAP@12
CandidateRecall@100
CandidateRecall@300
coverage
high_MAP
medium_MAP
low_MAP
cold_start_MAP
runtime_seconds
decision
```

**Decision:** `promote`、`reject` 或 `needs_more_evidence`。

---

# 阶段 3：学习排序替代手工权重

## Task 14：生成排序训练候选集

**Objective:** 将所有召回通道的候选合并成用户—商品训练样本，并严格按时间生成标签。

**Files:**

- Create: `src/ranking/__init__.py`
- Create: `src/ranking/candidate_dataset.py`
- Create: `tests/test_candidate_dataset.py`

**数据结构：**

```text
user_id
item_id
group_id
label
prediction_date
channel ranks/scores
candidate source flags
```

**标签：**

- 预测周实际购买：`label=1`。
- 候选集中未购买：`label=0`。

**采样策略：**

- 所有正样本保留。
- 每用户最多保留 200～300 个负样本。
- 优先保留高排名和多通道命中的 hard negatives。
- 不从全商品库随机生成大量无意义负样本。

**Tests:**

- 每个正样本必须来自标签周。
- 候选特征只来自标签周之前。
- 同一用户商品对去重。
- 一个用户的所有候选拥有同一 `group_id`。

---

## Task 15：实现无泄漏排序特征

**Objective:** 构造能表达通道质量、用户偏好、复购和趋势的特征。

**Files:**

- Create: `src/ranking/features.py`
- Create: `tests/test_ranking_features.py`

**第一版只实现下列特征：**

通道特征：

- 每个通道的 rank。
- 每个通道的归一化 score。
- 是否被该通道召回。
- 命中通道数量。
- 最佳 rank、平均 rank。

用户特征：

- 历史长度。
- 活跃度层级。
- 最近一次购买距预测日天数。
- 最近 7/14/28 天购买次数。
- 用户历史去重商品数。

商品特征：

- 最近 1/2/4 周热度。
- 商品类别、颜色、department、garment group。
- 商品首次和最近出现时间。

用户—商品交叉特征：

- 是否购买过该商品。
- 历史购买次数。
- 距上次购买天数。
- 用户对该商品类别的购买占比。
- 用户对颜色、department、garment group 的偏好比例。
- 商品与最近购买商品的最大 Item2Item 相似度。

**明确不在第一版加入：**

- 文本 embedding。
- 图片 embedding。
- 大规模深度交叉网络。

**Acceptance:** 所有时间相关特征都有 `as_of_date` 参数，并由防泄漏测试覆盖。

---

## Task 16：训练 CatBoostRanker

**Objective:** 学习替代固定分层权重的候选排序函数。

**Files:**

- Modify: `requirements.txt`，增加 `catboost`
- Create: `src/ranking/train.py`
- Create: `src/ranking/predict.py`
- Create: `run_ranker.py`
- Create: `tests/test_ranker_smoke.py`

**初始模型：**

```python
CatBoostRanker(
    loss_function="YetiRankPairwise",
    eval_metric="NDCG:top=12",
    iterations=500,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=5,
    random_seed=2026,
    verbose=50,
)
```

参数仅作为首轮起点，不在最终 test 上搜索。

**训练协议：**

- Window 1、Window 2 生成排序训练数据。
- Window 3 用于 early stopping、特征选择和模型比较。
- 最终 test 在模型方案冻结后运行一次。
- `group_id=user_id`，保证同一用户候选在同一排序组。

**对照实验：**

1. 当前默认手工权重。
2. valid 搜索后的手工权重。
3. Logistic Regression 简单融合基线。
4. CatBoostRanker 全特征。

**Acceptance:** CatBoost 必须在多窗口平均 MAP@12 上超过搜权融合，而不是只提高训练指标。

---

## Task 17：做排序特征消融

**Objective:** 证明排序提升来自哪些信息，避免模型不可解释地堆特征。

**Files:**

- Modify: `run_ranker.py`
- Output: `outputs/experiments/<run_id>/feature_ablation.csv`

**消融顺序：**

| 实验 | 特征组 |
|---|---|
| R0 | 仅通道 rank/score |
| R1 | R0 + 用户活跃度 |
| R2 | R1 + 商品趋势热度 |
| R3 | R2 + 复购特征 |
| R4 | R3 + 类别偏好 |
| R5 | R4 + Item2Item 相似度 |

**Gate:** 不能稳定贡献的特征组从默认模型移除，保持模型简洁。

---

# 阶段 4：SASRecF 定向优化

## Task 18：检查同日购物篮和序列确定性

**Objective:** 解决 H&M 同一天多件商品缺少真实先后顺序导致的序列噪声。

**Files:**

- Modify: `run_sasrec.py`
- Modify: `src/data/preprocess.py`
- Create: `tests/test_sequence_samples.py`

**需要比较：**

1. 当前逐商品序列。
2. 同日内按稳定商品 ID 排序。
3. 同日重复商品去重。
4. 仅保留每天一个代表顺序或购物篮聚合实验。

**Acceptance:** 无论原始 CSV 行顺序如何，相同输入集合生成的序列样本必须确定。

---

## Task 19：进行小规模 SASRecF 参数矩阵

**Objective:** 在保持数据和评估口径一致的条件下判断模型容量、序列长度和损失函数影响。

**Files:**

- Create: `configs/sasrecf_sweeps/`
- Create: `run_sasrecf_sweep.py`
- Output: `outputs/experiments/sasrecf_sweep_summary.csv`

**第一轮矩阵：**

| 参数 | 候选值 |
|---|---|
| `MAX_ITEM_LIST_LENGTH` | 30, 50, 100 |
| `hidden_size` | 64, 128 |
| `n_layers` | 1, 2 |
| `loss_type` | CE, BPR |

采用分阶段搜索，不做 3×2×2×2 全笛卡尔积：

1. 固定模型容量，先选序列长度。
2. 固定序列长度，比较 hidden size 和层数。
3. 最后比较 CE/BPR。

每个候选先单种子筛选，进入最终候选后再跑 3 个种子。

**Acceptance:** 报告均值、标准差、训练耗时和 checkpoint 大小；单次最高分不能直接成为最终结论。

---

## Task 20：评估商品特征有效性

**Objective:** 判断 SASRecF 的 8 个商品类别特征哪些真正有用。

**Files:**

- Modify: `configs/sasrecf.yaml`
- Extend: `run_sasrecf_sweep.py`

**消融：**

1. 无商品特征，即 SASRec。
2. 仅 product/garment 类别。
3. 加颜色特征。
4. 加 section/department/index 特征。
5. 全部 8 个特征。

**Gate:** 若 SASRecF 多窗口、多种子平均表现不能稳定超过 SASRec，则不再优先投入更复杂的 SASRecF 特征融合。

---

## Task 21：仅在前述实验支持后加入时间特征

**Objective:** 让序列模型区分连续购买、长期间隔和短期兴趣。

**Files:**

- Potential Create: `src/models/time_aware_sasrec.py`
- Potential Create: `tests/test_time_aware_sasrec.py`
- Modify: `configs/sasrecf.yaml`

**进入条件：**

- 候选 Recall 已不再是主要瓶颈。
- 学习排序已经稳定。
- 同日序列问题已经处理。
- SASRecF 消融证明序列模型通道仍有稳定贡献。

否则跳过该任务，避免过早自定义 RecBole 模型。

---

# 阶段 5：冷启动与 Kaggle 输出完整性

## Task 22：建立完整用户冷启动策略

**Objective:** 不只评估标签周出现的用户，而是能为完整 customer 集生成 12 个商品。

**Files:**

- Create: `src/recall/cold_start.py`
- Create: `tests/test_cold_start.py`

**回退优先级：**

1. 用户年龄段 × 销售渠道热门。
2. 用户可用属性分群热门。
3. 最近 1/2 周全局热门。
4. 全局稳定热门。

**Acceptance:** 任意用户都能得到 12 个唯一、格式正确的商品 ID。

---

## Task 23：实现提交文件生成与校验

**Objective:** 将离线项目补全为能够生成 H&M Kaggle submission 的端到端系统。

**Files:**

- Create: `run_build_submission.py`
- Create: `src/service/submission.py`
- Create: `tests/test_submission.py`

**验证项：**

- customer 数量与 `sample_submission.csv` 完全一致。
- 每个 customer 恰好 12 个商品。
- 商品 ID 保留 10 位前导零。
- 商品之间以单个空格分隔。
- 不存在空预测、重复商品或未知商品。
- 用户顺序与 sample submission 一致。

**Command:**

```bash
python run_build_submission.py \
  --sample-submission data/raw/sample_submission.csv \
  --output outputs/submissions/submission.csv
```

---

# 阶段 6：最终验收与文档

## Task 24：运行最终对照实验

**Objective:** 在冻结的 test 上比较基线与最终方案。

**最终对照：**

1. SASRecF 单通道。
2. 原始四路默认权重。
3. 原始四路 valid 搜权。
4. 改进召回 + 手工融合。
5. 改进召回 + CatBoostRanker。

**必须输出：**

- 整体与分层 MAP@12。
- Candidate Recall@100/300。
- 通道覆盖和独占命中。
- 三窗口平均值和标准差。
- 最终 test 单次结果。
- 运行时间和资源消耗。

**Acceptance:** 报告明确区分回测均值、valid 选择结果和最终 test，禁止混用。

---

## Task 25：更新项目文档和架构图

**Objective:** 让 README、项目指南、代码和实验结果保持一致。

**Files:**

- Modify: `README.md`
- Modify: `docs/PROJECT_GUIDE.md`
- Create: `docs/v3_optimization_report.md`
- Create: `outputs/figures/README.md` 或移除无效链接

**文档内容：**

- 数据 manifest 标识。
- 最终数据时间边界。
- 新旧系统结构对照。
- 候选召回与排序特征说明。
- 防泄漏协议。
- 多窗口回测结果。
- 消融实验。
- 完整复现命令。
- 已知限制和下一步研究方向。

---

## 3. 推荐执行顺序与里程碑

### Milestone A：可信基线

包含 Task 1～6。

交付物：

- 实验配置。
- 数据 manifest。
- 核心指标测试。
- 防泄漏测试。
- 当前基线和候选诊断报告。

只有 Milestone A 完成后，才开始宣称算法提升。

### Milestone B：高质量候选集

包含 Task 7～13。

交付物：

- 复购召回。
- 多窗口热门。
- 改进 Item2Item。
- 类别转移召回。
- 三窗口回测报告。

目标是优先提升 Candidate Recall@300 和通道互补性。

### Milestone C：学习排序

包含 Task 14～17。

交付物：

- 无泄漏排序数据集。
- CatBoostRanker。
- 特征消融结果。
- 与手工权重的严格对照。

这是预期最有价值的系统升级。

### Milestone D：模型精调

包含 Task 18～21。

交付物：

- 确定性序列生成。
- SASRecF 参数和商品特征消融。
- 是否值得开发时间感知 SASRec 的证据。

### Milestone E：完整交付

包含 Task 22～25。

交付物：

- 全用户冷启动。
- Kaggle submission builder。
- 最终 test 报告。
- v3 项目文档。

---

## 4. 测试与验证总清单

每个实现任务完成后至少运行对应单测；阶段合并时运行：

```bash
pytest tests -v
```

阶段 0 验证：

```bash
pytest tests/test_experiment_config.py \
       tests/test_data_manifest.py \
       tests/test_metrics.py \
       tests/test_time_split.py \
       tests/test_no_leakage.py -v
```

阶段 1 验证：

```bash
pytest tests/test_repurchase.py \
       tests/test_popular_windows.py \
       tests/test_item2item_similarity.py \
       tests/test_category_transition.py \
       tests/test_recall_registry.py -v
```

阶段 3 验证：

```bash
pytest tests/test_candidate_dataset.py \
       tests/test_ranking_features.py \
       tests/test_ranker_smoke.py \
       tests/test_no_leakage.py -v
```

所有实验验收必须同时检查：

- 数据 manifest 是否相同。
- 时间边界是否相同。
- 用户评估集合是否相同。
- 候选 Top-K 是否相同。
- item ID 规范是否相同。
- 随机种子是否记录。
- test 是否未参与调参。

---

## 5. 主要风险与处理方式

### 风险 1：valid 搜权过拟合

**表现：** valid 上升，但 test 或其他时间窗口下降。

**处理：** 用滚动窗口均值选方案；最终 test 冻结；减少权重和特征搜索空间。

### 风险 2：候选集没有真实商品

**表现：** Ranker 训练指标不错，但最终 MAP 上限低。

**处理：** 排序前先以 Candidate Recall@300 作为阶段门槛；优先增加互补召回。

### 风险 3：热门商品支配全部通道

**表现：** 各通道 Jaccard 很高，独占命中很低。

**处理：** Item2Item 使用 cosine/热门惩罚；增加复购和类别转移；Ranker 加入来源标记。

### 风险 4：同日购买顺序是伪序列

**表现：** SASRecF 对输入行顺序敏感，多次预处理结果不一致。

**处理：** 建立确定性同日排序；比较去重和购物篮处理；保留消融结果。

### 风险 5：排序特征泄漏

**表现：** valid 指标异常高，跨窗口明显崩溃。

**处理：** 所有特征函数强制接收 `as_of_date`；专门防泄漏测试；每个窗口重新构建索引。

### 风险 6：工程复杂度增长过快

**表现：** 新通道大量硬编码到 `offline_eval.py`，难以维护。

**处理：** 在增加多个通道前完成 registry；每阶段只保留通过放行标准的方案。

### 风险 7：指标口径混淆

**表现：** RecBole full ranking 与 Offline candidate ranking 数字被直接比较。

**处理：** 报告中明确写 `protocol`；不同协议分别建表；不计算跨协议的直接相对提升结论。

---

## 6. 实施优先级总结

如果资源有限，只做最值得的部分，顺序应为：

1. Task 1～6：实验协议、测试和候选诊断。
2. Task 7～9：复购、多窗口热门、改进 Item2Item。
3. Task 12：三窗口回测。
4. Task 14～17：CatBoostRanker 和特征消融。
5. Task 18～20：SASRecF 定向消融。
6. Task 22～25：冷启动、提交文件和最终文档。

最核心的决策原则是：

```text
先证明候选集变好
  -> 再证明排序变好
  -> 最后判断是否需要更复杂的序列模型
```

这样可以把每一次 MAP@12 变化追溯到明确原因，避免项目变成无法复现的模型和规则堆叠。
