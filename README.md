# FashionRec-Transformer

> 📖 **完整项目指南（推荐先读）：** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)  
> 📊 **成果图表（答辩/PPT 用）：** [outputs/figures/README.md](outputs/figures/README.md)
> 从零讲清数据 → 训练 → 四路召回 → 融合 → 评估 → 权重搜索全流程。

基于 [H&M 交易数据](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 的时尚推荐项目。当前**主实验线**为 **SASRecF**（带商品类别特征）+ **四路离线召回融合**（Popular / Category Popular / Item2Item / SASRecF），按用户活跃度自适应加权，最终以 **Offline MAP@12** 作为主评估口径。

---

## 项目结构

```
FashionRec-Transformer/
├── configs/
│   ├── sasrecf.yaml                # SASRecF 训练配置（主实验）
│   └── sasrec.yaml                 # SASRec 训练配置（v1 对照）
├── data/                           # 原始与处理后数据（大文件见 .gitignore）
├── docs/                           # 实验报告与入门指南
├── outputs/                        # checkpoint / 召回 CSV / 评估 JSON
├── src/
│   ├── domain/                     # 商品 ID 与 Candidate 数据契约
│   ├── data/                       # 数据过滤、切分、序列样本、商品特征
│   ├── recall/                     # 通道接口、注册表与统一候选生成
│   ├── candidates/                 # 候选并集、去重与物化
│   ├── ranking/                    # Weighted RRF 与 LightGBM 特征边界
│   ├── evaluate/                   # 指标、权重搜索与评估报告
│   ├── experiment/                 # 统一配置、运行上下文与产物路径
│   ├── pipeline/                   # 配置驱动的流水线编排
│   ├── fusion/                     # 旧融合 API（兼容层）
│   └── pytorch_compat.py           # PyTorch / NumPy 兼容补丁
├── run_pipeline.py                 # 一键按顺序跑全流程
├── run_data_prep.py                # ① 数据准备
├── run_sasrecf.py                  # ② 训练 SASRecF
├── run_sasrecf_recall.py           # ③ 导出 SASRecF 召回
├── run_rule_recall.py              # ④ 规则召回与四路候选物化
├── run_fusion_weight_search.py     # ⑤ valid 权重搜索
├── run_offline_eval.py             # ⑥ 融合 + MAP@12 评估
├── run_sasrec.py                   # v1 对照：训练 SASRec
├── requirements.txt
└── README.md
```

---

## 脚本执行顺序（主实验线）

| 步骤 | 脚本 | 说明 | 必需 |
|------|------|------|------|
| ① | `run_data_prep.py` | filter（可选）→ preprocess → split → hm_seq → item 特征 | ✅ |
| ② | `run_sasrecf.py` | 训练 SASRecF | ✅ |
| ③ | `run_sasrecf_recall.py` | 导出 `sasrecf_valid.csv` / `sasrecf_test.csv` | ✅ |
| ④ | `run_rule_recall.py` | 规则召回 + 四路候选并集物化 | ✅ |
| ⑤ | `run_fusion_weight_search.py` | valid 上网格搜融合权重 | ✅ |
| ⑥ | `run_offline_eval.py` | 四路融合 + MAP@12（先 valid，再 test） | ✅ |

### 一键跑全流程

```bash
conda activate dl
python run_pipeline.py --with-filter
```

正式流水线会创建 `outputs/runs/<run_id>/`，把配置快照、checkpoint、召回、候选、排序和指标隔离保存。完整依赖方向与数据契约见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

可选参数：

- `--run-id <id>`：继续使用指定运行目录
- `--no-strict`：允许旧流程的兼容回退
- `--skip-data-prep` / `--skip-train` / `--skip-recall` / `--skip-candidates` / `--skip-weight-search`：跳过对应步骤

### 逐步手动执行

```bash
conda activate dl

python run_data_prep.py --with-filter
python run_sasrecf.py --skip-preprocess
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test
python run_fusion_weight_search.py
python run_offline_eval.py --eval-split valid
python run_offline_eval.py --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

---

## 代码注释规范

本项目 **所有 Python 源码**（`run_*.py` 与 `src/**/*.py`）均采用 **简体中文行尾注释**（`# ...`）：

- 每一行非空白代码均有注释，说明该行作用
- 模块 docstring 与函数说明同样使用简体中文
- 注释风格与 `run_sasrec.py`、`src/data/filter.py` 保持一致

阅读代码时可直接看行尾注释理解逻辑，无需单独对照文档。

---

## 核心流程

```
filter → preprocess → split → hm_seq → hm_seq.item
    → 训练 SASRecF → 导出 sasrecf_{valid,test}.csv
    → 四路 Candidate 物化 → Weighted RRF / LightGBM 排序边界
    → valid 权重搜索 → offline_eval（MAP@12）
```

四路召回：**SASRecF**、**Popular**、**Category Popular**、**Item2Item**。融合时按用户历史长度分档（high / medium / low / cold_start）；默认排序实现位于 `src/ranking/weighted_rrf.py`。

### 两套切分逻辑

| 用途 | 数据 | 说明 |
|------|------|------|
| RecBole 训练 | `hm_seq.{train,valid,test}.inter` | 含 `item_id_list` 序列列 |
| 离线评估 | `hm.{train,valid,test}.inter` | valid：历史=train；test：历史=train+valid |

### 关键默认参数

| 参数 | 值 | 位置 |
|------|-----|------|
| 数据窗口 | 6 周（4+1+1） | `filter.py` / `preprocess.py` / `split.py` |
| 每用户最长行为 | 100 | 各 data / fusion 模块 |
| 序列最大长度 | 100 | `configs/sasrecf.yaml` |
| 召回 Top-K | 100 → 融合 Top-12 | 召回 + `offline_eval.py` |
| 主指标 | MAP@12 | `offline_eval.py` |

---

## 环境准备

```bash
conda activate dl
pip install -r requirements.txt
```

将 H&M Kaggle 数据放入 `data/raw/`：`transactions_train.csv`、`articles.csv`、`customers.csv`。

---

## 各步骤说明

### ① 数据准备

```bash
python run_data_prep.py
python run_data_prep.py --with-filter   # 含 filter 采样
```

### ② 训练 SASRecF

```bash
python run_sasrecf.py --skip-preprocess
```

checkpoint：`outputs/checkpoints/sasrecf/`

### ③ SASRecF 召回

```bash
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test
```

输出：`outputs/recommendations/sasrecf_{valid,test}.csv`

### ④ 规则召回与候选物化

```bash
python run_rule_recall.py --eval-split both
```

旧命令输出 `popular_*.csv`、`category_popular_*.csv`、`item2item_*.csv` 和候选并集；正式流水线还会把 SASRecF 召回合入 `outputs/runs/<run_id>/candidates/{valid,test}.csv`，权重搜索与评估消费同一份固定候选。

### ⑤⑥ 权重搜索与融合评估

```bash
python run_fusion_weight_search.py
python run_offline_eval.py --eval-split valid
python run_offline_eval.py --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

输出：

- `outputs/evaluation/best_fusion_weights.json`
- `outputs/recommendations/fusion_{valid,test}.csv`
- `outputs/evaluation/fusion_{valid,test}_metrics.json`

---

## 源码模块索引

| 目录 / 文件 | 职责 |
|-------------|------|
| `src/data/filter.py` | 原始数据过滤（时间窗 / Top item / 活跃用户） |
| `src/data/preprocess.py` | CSV → `hm.inter` |
| `src/data/split.py` | 按周切分 train / valid / test |
| `src/data/build_item_features.py` | `articles.csv` → `hm_seq.item` |
| `src/data/build_sequences.py` | 时间切分 → 因果序列样本 |
| `src/domain/ids.py` | 用户与商品 ID 规范 |
| `src/domain/candidates.py` | 跨召回/排序 Candidate schema |
| `src/recall/popular.py` | 全局热门召回 |
| `src/recall/category_popular.py` | 类别热门召回 |
| `src/recall/item2item.py` | 商品共现召回 |
| `src/recall/sasrec_recall.py` | SASRecF Top-K 导出 |
| `src/recall/generator.py` | 统一候选生成与物化 |
| `src/candidates/union.py` | 候选去重与并集上限 |
| `src/ranking/weighted_rrf.py` | 按活跃度加权 RRF 排序 |
| `src/ranking/features.py` | LightGBM LambdaRank 特征表 |
| `src/evaluate/weight_search.py` | valid 集权重网格搜索 |
| `src/evaluate/offline_eval.py` | 四路融合 + 离线 MAP@12 |
| `src/pipeline/orchestrator.py` | 配置驱动的 run-scoped 阶段编排 |

---

## v1 对照

```bash
python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess
```

详见 [docs/v1_experiment_report.md](docs/v1_experiment_report.md)。

---

## 评估指标

- **MAP@12**（主指标）
- Recall@12、NDCG@12、Hit@12

---

## 实验文档

| 文档 | 内容 |
|------|------|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | 入门指南与 FAQ |
| [docs/v2_sasrecf_fusion_experiment_report_jul09.md](docs/v2_sasrecf_fusion_experiment_report_jul09.md) | SASRecF 四路融合实验报告 |
| [docs/two_experiments_chronicle.md](docs/two_experiments_chronicle.md) | 两次实验对照 |
| [docs/sasrec_recbole_comparison.md](docs/sasrec_recbole_comparison.md) | Offline vs RecBole 口径说明 |

---

## Git 与大文件

`.gitignore` 已忽略 `data/raw/**`、`data/processed/**` 大 CSV 与 `outputs/**` 产物。克隆后自行准备数据并按上述顺序运行即可。
