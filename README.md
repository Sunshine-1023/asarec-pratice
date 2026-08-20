# FashionRec-Transformer

> 📖 **完整项目指南（推荐先读）：** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)  
> 📊 **成果图表（答辩/PPT 用）：** [outputs/figures/README.md](outputs/figures/README.md)
> 从零讲清数据 → 训练 → 四路召回 → 融合 → 评估 → 权重搜索全流程。

基于 [H&M 交易数据](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 的时尚推荐项目。项目提供两条物理隔离的正式训练链：稳定的 **SASRecF + 四路 Weighted RRF baseline**，以及使用购物篮标签、PIT 特征、扩展召回和 **LightGBM LambdaRank** 的 industrial 实验链。

---

## 项目结构

```
FashionRec-Transformer/
├── Makefile                        # 统一训练、续跑、评估命令入口
├── pyproject.toml                  # 包元数据与 fashionrec 命令注册
├── configs/
│   ├── sasrecf.yaml                # SASRecF 训练配置（主实验）
│   └── sasrec.yaml                 # SASRec 训练配置（v1 对照）
├── data/                           # 原始与处理后数据（大文件见 .gitignore）
├── docs/                           # 实验报告与入门指南
├── outputs/                        # checkpoint / 召回 CSV / 评估 JSON
├── src/
│   └── fashionrec/
│       ├── cli.py                  # 唯一公开 Python 命令入口
│       ├── domain/                 # 商品 ID 与 Candidate 数据契约
│       ├── data/                   # 数据过滤、切分、序列样本、商品特征
│       ├── recall/                 # 通道接口、注册表与统一候选生成
│       ├── candidates/             # 候选并集、去重与物化
│       ├── ranking/                # Weighted RRF、融合与排序特征
│       ├── training/               # 模型训练与 checkpoint 选择
│       ├── evaluation/             # 指标、权重搜索与评估报告
│       ├── experiment/             # 统一配置、运行上下文与产物路径
│       └── pipeline/               # 配置驱动的流水线编排
├── tests/
├── requirements.txt
└── README.md
```

---

## Makefile 统一入口

日常训练与评估统一从 Makefile 启动。Makefile 不复制流水线逻辑，所有目标最终都进入 `fashionrec` CLI，再由 package 内的 pipeline 层根据统一配置和 run-scoped 产物目录编排。

两条链路使用独立配置和产物命名空间。它们只共享只读的 `data/raw/` 与源码，不共享处理数据、checkpoint、候选、排序模型、权重或指标：

```text
outputs/runs/baseline/<run-id>/
outputs/runs/industrial/<run-id>/
```

相同 run-id 可以同时用于两个 profile；同一 profile 内若配置哈希发生变化，续跑会直接失败。

```bash
conda activate dl

make help
make baseline WITH_FILTER=1
make industrial WITH_FILTER=1
```

完整运行不指定 `RUN_ID` 时会自动创建新实验目录。分阶段执行或续跑必须显式使用同一个 `RUN_ID`：

```bash
make data PROFILE=baseline RUN_ID=exp-001 WITH_FILTER=1
make train PROFILE=baseline RUN_ID=exp-001
make select-checkpoint PROFILE=baseline RUN_ID=exp-001
make recall PROFILE=baseline RUN_ID=exp-001
make candidates PROFILE=baseline RUN_ID=exp-001
make weights PROFILE=baseline RUN_ID=exp-001
make evaluate PROFILE=baseline RUN_ID=exp-001
```

模型已经训练完成时，可以一次运行全部下游步骤：

```bash
make downstream PROFILE=baseline RUN_ID=exp-001
```

工业链分阶段运行时将 `PROFILE` 改为 `industrial`；`make ranker PROFILE=industrial RUN_ID=<id>` 可在已有数据和候选上重建训练表、训练 LambdaRank、打分并评估。常用变量：`PYTHON`、`PROFILE`、`EXPERIMENT_CONFIG`、`OUTPUT_ROOT`、`RUN_ID`、`WITH_FILTER=0|1`、`STRICT=0|1`、`WEIGHTS_JSON`。

---

## 两条训练链

| Profile | 配置 | 数据/标签 | 候选 | 排序与评估 |
|---|---|---|---|---|
| `baseline` | `configs/baseline/experiment.yaml` | 行级时间切分、SASRecF 序列 | Popular / Category Popular / Item2Item / SASRecF | valid 搜索 Weighted RRF，行级购买集合 MAP@12 |
| `industrial` | `configs/industrial/experiment.yaml` | user-day-item、basket、next-basket、PIT 用户特征 | baseline + Repurchase / Style / Content | 因果快照训练表、LambdaRank、next-basket RRF 对照 |

## 脚本执行顺序

| 步骤 | Make 目标 / CLI | 说明 | 必需 |
|------|------|------|------|
| ① | `make data` / `fashionrec data` | filter（可选）→ preprocess → split → hm_seq → item 特征 | ✅ |
| ② | `make train` / 应用内 `train` | 使用各应用自己的 `models/sasrecf.yaml` 训练 SASRecF | ✅ |
| ③ | `make select-checkpoint` / `fashionrec select-checkpoint` | 完整 valid 用户周 MAP@12 选择 checkpoint | ✅ |
| ④ | `make recall` / `fashionrec recall` | 使用选定模型导出 valid/test 召回 | ✅ |
| ⑤ | `make candidates` / `fashionrec candidates` | baseline 四路或 industrial 扩展多路候选物化 | ✅ |
| ⑥ | `make weights` / `fashionrec weights` | valid 上坐标下降搜索融合权重 | ✅ |
| ⑦ | `make evaluate` / `fashionrec evaluate` | 四路融合 + MAP@12（test 只在最终阶段评估） | ✅ |

### 一键跑全流程

```bash
conda activate dl
make baseline WITH_FILTER=1
# 或
make industrial WITH_FILTER=1
```

正式流水线会创建 `outputs/runs/<profile>/<run_id>/`，把配置快照、checkpoint、召回、候选、排序和指标隔离保存。完整依赖方向与数据契约见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

两条链现在是两个 top-level 应用。Baseline 位于 `src/fashionrec/baseline/`，Industrial 位于 `src/fashionrec/industrial/`；它们分别拥有 data/models/recall/ranking/evaluation/pipeline/CLI 及正式算法实现。共享层只保留 ID、Candidate、Recall/Ranker 接口、中立 I/O、纯指标和运行器；旧顶层算法目录仅用于 import 兼容。

可选参数：

- `--run-id <id>`：继续使用指定运行目录
- `--no-strict`：允许旧流程的兼容回退
- `--skip-data-prep` / `--skip-train` / `--skip-checkpoint-selection` / `--skip-recall` / `--skip-candidates` / `--skip-weight-search`：跳过对应步骤

### 统一 Python CLI

```bash
PYTHONPATH=src python -m fashionrec --help
PYTHONPATH=src python -m fashionrec.baseline pipeline --experiment-config configs/baseline/experiment.yaml
PYTHONPATH=src python -m fashionrec.industrial pipeline --experiment-config configs/industrial/experiment.yaml
PYTHONPATH=src python -m fashionrec.baseline --help
PYTHONPATH=src python -m fashionrec.industrial --help
PYTHONPATH=src python -m fashionrec train --help
```

执行 `pip install -e .` 后可直接使用 `fashionrec <command>`，不需要设置 `PYTHONPATH`。

---

## 代码注释规范

本项目 Python 源码位于 `src/fashionrec/`，采用简体中文行尾注释：

- 每一行非空白代码均有注释，说明该行作用
- 模块 docstring 与函数说明同样使用简体中文
- 注释风格与 `src/fashionrec/data/filter.py` 保持一致

阅读代码时可直接看行尾注释理解逻辑，无需单独对照文档。

---

## 核心流程

```
filter（train 拟合商品集）→ preprocess → split → model_train → hm_seq → hm_seq.item
    → 训练 SASRecF → valid 用户周 MAP@12 选 checkpoint → 导出 sasrecf_{valid,test}.csv
    → 四路 Candidate 物化 → Weighted RRF / LightGBM 排序边界
    → valid 权重搜索 → offline_eval（MAP@12）
```

四路召回：**SASRecF**、**Popular**、**Category Popular**、**Item2Item**。融合时按用户历史长度分档（high / medium / low / cold_start）；默认排序实现位于 `src/fashionrec/ranking/weighted_rrf.py`。

### 两套切分逻辑

| 用途 | 数据 | 说明 |
|------|------|------|
| RecBole 训练 | `hm_seq.{train,valid,test}.inter` | 含 `item_id_list` 序列列 |
| 离线评估 | `hm.{train,valid,test}.inter` | valid：历史=train；test：历史=train+valid |

### 关键默认参数

| 参数 | 值 | 位置 |
|------|-----|------|
| 数据窗口 | 6 周（4+1+1） | `filter.py` / `preprocess.py` / `split.py` |
| 每用户最长行为 | 使用时截断至 100，不在切分前删行 | 序列 / 召回上下文 |
| checkpoint 粗筛 | 所有验证 epoch 中 RecBole-valid 指标 Top-5 | `configs/baseline/experiment.yaml` |
| Baseline 序列最大长度 | 100 | `configs/baseline/models/sasrecf.yaml` |
| Industrial 序列最大长度 | 100 | `configs/industrial/models/sasrecf.yaml` |
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
make data RUN_ID=exp-001                 # 始终使用原始 transactions
make data RUN_ID=exp-001 WITH_FILTER=1   # 生成并使用本次 train-fitted filtered transactions
```

数据准备不会因为 `data/raw/filtered/` 中存在历史文件而自动切换输入；实际使用路径会写入 `outputs/runs/<profile>/<run_id>/data/manifest.json`。

### ② 训练 SASRecF

```bash
make train RUN_ID=exp-001
```

checkpoint：`outputs/runs/<profile>/<run_id>/checkpoints/sasrecf/`

同一 run/seed 的 shortlist 目录非空时训练会安全失败，避免混入旧 checkpoint。需要复用旧结果时跳过训练；需要重新训练时使用新的 run ID 或 checkpoint 目录。

### ③ SASRecF 召回

```bash
make recall RUN_ID=exp-001
```

输出：`outputs/runs/<profile>/<run_id>/recall/sasrecf_{valid,test}.csv`

### ④ 规则召回与候选物化

```bash
make candidates RUN_ID=exp-001
```

该阶段输出各通道召回到 `outputs/runs/<profile>/<run_id>/recall/`，候选并集写入 `outputs/runs/<profile>/<run_id>/candidates/{valid,test}.csv`；权重搜索与评估消费同一份固定候选。

### ⑤⑥ 权重搜索与融合评估

```bash
make weights RUN_ID=exp-001
make evaluate RUN_ID=exp-001
```

输出：

- `outputs/runs/<profile>/<run_id>/ranking/best_fusion_weights.json`
- `outputs/runs/<profile>/<run_id>/ranking/fusion_{valid,test}.csv`
- `outputs/runs/<profile>/<run_id>/evaluation/fusion_{valid,test}_metrics.json`

---

## 源码模块索引

| 目录 / 文件 | 职责 |
|-------------|------|
| `src/fashionrec/baseline/` | 旧协议数据、SASRecF、四路召回、Weighted RRF、评估与独立 DAG |
| `src/fashionrec/industrial/` | 购物篮/PIT、扩展召回、SASRecF、LambdaRank、next-basket 评估与独立 DAG |
| `src/fashionrec/shared/` | domain、interfaces、io、metrics、runtime 中立内核 |
| `src/fashionrec/{data,recall,ranking,training,evaluation,candidates}/` | 旧 import 兼容 facade，不承载正式算法 |
| `src/fashionrec/domain/` | 旧领域契约 import 兼容 facade |
| `src/fashionrec/experiment/` | 配置、运行上下文与产物路径 |
| `src/fashionrec/pipeline/` | 旧 profile pipeline 兼容入口；正式 DAG 在两应用内部 |

---

## v1 对照

```bash
PYTHONPATH=src python -m fashionrec train --config configs/sasrec.yaml
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
