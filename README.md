# FashionRec-Transformer

> 📖 **完整项目指南（推荐先读）：** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)  
> 从零讲清数据 → 训练 → 四路召回 → 融合 → 评估 → 权重搜索全流程。

基于 [H&M 交易数据](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 的时尚推荐项目。当前**主实验线**为 **SASRecF**（带商品类别特征）+ **四路离线召回融合**（Popular / Category Popular / Item2Item / SASRecF），按用户活跃度自适应加权，最终以 **Offline MAP@12** 作为主评估口径。

仓库同时保留 **SASRec**（v1）训练与 **RecBole full ranking** 评估脚本，便于与早期实验对照。

---

## 项目结构

```
FashionRec-Transformer/
├── configs/
│   ├── sasrecf.yaml                # SASRecF 训练配置（主实验）
│   └── sasrec.yaml                 # SASRec 训练配置（v1 对照）
├── data/                           # 原始与处理后数据（大文件见 .gitignore）
│   ├── raw/
│   │   ├── transactions_train.csv
│   │   ├── articles.csv
│   │   ├── customers.csv
│   │   └── filtered/               # 可选：filter 采样输出
│   └── processed/
│       ├── hm/                     # 时序切分（离线评估主逻辑）
│       │   ├── hm.inter
│       │   ├── hm.train.inter      # 前 4 周
│       │   ├── hm.valid.inter      # 倒数第 2 周（标签）
│       │   └── hm.test.inter       # 最后 1 周（标签）
│       └── hm_seq/                 # RecBole 序列格式（训练用）
│           ├── hm_seq.train.inter
│           ├── hm_seq.valid.inter
│           ├── hm_seq.test.inter
│           └── hm_seq.item           # SASRecF 商品特征
├── docs/
│   ├── PROJECT_GUIDE.md            # 完整入门指南
│   ├── v2_sasrecf_fusion_experiment_report_jul09.md
│   ├── v1_experiment_report.md
│   ├── two_experiments_chronicle.md
│   └── sasrec_recbole_comparison.md
├── outputs/                        # 训练产物（见 .gitignore）
│   ├── checkpoints/sasrecf/        # SASRecF 权重
│   ├── checkpoints/sasrec/         # SASRec 权重
│   ├── recommendations/            # 召回与融合 CSV
│   └── evaluation/                 # 指标 JSON、最优权重
├── src/
│   ├── data/
│   │   ├── filter.py               # 原始数据过滤（时间窗 / Top item / 活跃用户）
│   │   ├── preprocess.py           # CSV → hm.inter
│   │   ├── split.py                # 按周切分 train / valid / test
│   │   └── build_item_features.py  # articles.csv → hm_seq.item
│   ├── recall/
│   │   ├── popular.py              # 全局热门召回
│   │   ├── category_popular.py     # 类别热门召回
│   │   ├── item2item.py            # 商品共现召回
│   │   ├── itemcf.py               # ItemCF（v1 实验用）
│   │   └── sasrec_recall.py        # SASRec / SASRecF Top-K 导出
│   ├── fusion/
│   │   └── weighted_fusion.py      # 按活跃度加权 rank 融合
│   ├── evaluate/
│   │   ├── offline_eval.py         # 四路融合 + 离线 MAP@12（主评估）
│   │   ├── weight_search.py        # valid 集权重网格搜索
│   │   ├── channel_only_eval.py    # 单通道离线评估
│   │   ├── recbole_channel_eval.py # RecBole 单通道评估
│   │   └── recbole_fusion_eval.py  # RecBole 融合评估
│   ├── pytorch_compat.py           # PyTorch 2.6 / NumPy 2.x 兼容补丁
│   └── service/                    # 预留服务层（FastAPI）
├── run_sasrecf.py                  # 训练 SASRecF（默认 configs/sasrecf.yaml）
├── run_sasrec.py                   # 训练 SASRec（通用 RecBole 入口）
├── run_sasrecf_recall.py           # 导出 SASRecF Top-100 召回
├── run_weight_search.py            # valid 权重搜索入口
├── run_recbole_channel_eval.py     # RecBole 单通道评估入口
├── run_recbole_fusion_eval.py      # RecBole 融合评估入口
├── requirements.txt
└── README.md
```

---

## 核心流程

### 主实验线（SASRecF + 四路 Offline 融合）

```
filter → preprocess → split → build_item_features
    → 训练 SASRecF → 导出 sasrecf_{valid,test}.csv
    → valid 权重搜索 → offline_eval（MAP@12）
```

四路召回：**SASRecF**、**Popular**、**Category Popular**、**Item2Item**。融合时按用户历史长度分档（high / medium / low / cold_start），各档使用不同通道权重（见 `src/fusion/weighted_fusion.py`）。

### 两套切分逻辑并存

| 用途 | 数据 | 说明 |
|------|------|------|
| RecBole 训练 | `hm_seq.{train,valid,test}.inter` | 由 `hm.{train,valid,test}.inter` 转换，含 `item_id_list` |
| 离线评估 | `hm.{train,valid,test}.inter` | valid：历史=train，标签=valid；test：历史=train+valid，标签=test |

### 关键默认参数

| 参数 | 值 | 位置 |
|------|-----|------|
| 数据窗口 | 6 周（train 4 + valid 1 + test 1） | `filter.py` / `preprocess.py` / `split.py` |
| 每用户最长行为 | 100 | 各 data / fusion 模块 |
| 序列最大长度 | 100 | `configs/sasrecf.yaml` → `MAX_ITEM_LIST_LENGTH` |
| 召回 Top-K | 100 → 融合截断 Top-12 | 召回脚本 + `offline_eval.py` |
| 主指标 | MAP@12 | `offline_eval.py` |

---

## 环境准备

```bash
conda activate dl          # 实验环境（勿用 base，缺少 pandas 等依赖）
pip install -r requirements.txt
```

依赖含 RecBole、PyTorch、pandas 等；`src/pytorch_compat.py` 会在训练/召回前自动打补丁，兼容 PyTorch 2.6+ 与 NumPy 2.x。

---

## 数据准备

将 H&M Kaggle 数据放入 `data/raw/`：

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

### 1. 可选：过滤采样

```bash
python -m src.data.filter
```

输出到 `data/raw/filtered/transactions_train.csv`（最近 6 周、Top 30k 商品、活跃用户、每用户最多 100 条行为）。

### 2. 预处理、切分与商品特征

```bash
python -m src.data.preprocess
python -m src.data.split
python -m src.data.build_item_features
```

生成：

- `data/processed/hm/hm.{inter,train,valid,test}.inter`
- `data/processed/hm_seq/hm_seq.item`

---

## 训练 SASRecF（主实验）

```bash
python run_sasrecf.py --skip-preprocess
```

说明：

- `run_sasrecf.py` 等价于 `python run_sasrec.py --config configs/sasrecf.yaml`
- 训练前自动将 `hm.*.inter` 转为 `hm_seq.*.inter`，并确保 `hm_seq.item` 存在
- checkpoint 默认保存到 `outputs/checkpoints/sasrecf/`
- 当前默认：`MAX_ITEM_LIST_LENGTH=100`、`hidden_size=128`、`loss_type=CE`、`learning_rate=5e-4`、`valid_metric=MAP@12`

若需从头跑数据预处理：

```bash
python run_sasrecf.py
```

### 训练 SASRec（v1 对照）

```bash
python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess
```

多随机种子对比：

```bash
python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess --seeds 2024,2025,2026
```

---

## 导出 SASRecF 召回

```bash
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test
```

输出：

- `outputs/recommendations/sasrecf_valid.csv`
- `outputs/recommendations/sasrecf_test.csv`

字段：`user_id, item_id, score, rank, channel`

---

## 四路融合与离线评估

### valid 权重搜索

在 valid 集上网格搜索各活跃度分层的通道权重：

```bash
python run_weight_search.py
```

输出：`outputs/evaluation/best_fusion_weights.json`

### 融合评估

```bash
# valid：查看基线或调参效果
python -m src.evaluate.offline_eval --eval-split valid

# test：使用搜索到的权重报最终成绩
python -m src.evaluate.offline_eval \
  --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

### 评估数据来源

| 阶段 | 用户历史 | 规则通道索引 | SASRecF 召回 | 标签 |
|------|----------|--------------|--------------|------|
| valid | `hm.train.inter` | `hm.train.inter` | `sasrecf_valid.csv` | `hm.valid.inter` |
| test | `hm.train + hm.valid` | `hm.train + hm.valid` | `sasrecf_test.csv` | `hm.test.inter` |

### 输出文件

- 融合推荐：`outputs/recommendations/fusion_{valid,test}.csv`
- 评估指标：`outputs/evaluation/fusion_{valid,test}_metrics.json`

---

## 单通道与 RecBole 评估

### 单通道离线评估

```bash
python -m src.evaluate.channel_only_eval --channel all --eval-split valid
```

可选 `--channel popular|category_popular|item2item|sasrecf|all`。

### RecBole full ranking（对照口径）

```bash
python run_recbole_channel_eval.py
python run_recbole_fusion_eval.py
```

用于与 RecBole 内置评估协议对照；**主实验报告以 Offline 融合 MAP@12 为准**。详见 [docs/sasrec_recbole_comparison.md](docs/sasrec_recbole_comparison.md)。

---

## 标准全流程（复制即用）

```bash
conda activate dl

# 数据
python -m src.data.filter          # 可选
python -m src.data.preprocess
python -m src.data.split
python -m src.data.build_item_features

# 训练与召回
python run_sasrecf.py --skip-preprocess
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test

# 评估
python run_weight_search.py
python -m src.evaluate.offline_eval --eval-split valid
python -m src.evaluate.offline_eval \
  --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

---

## 评估指标

离线评估（`offline_eval.py`）主指标：

- **MAP@12**（主指标）
- Recall@12、NDCG@12、Hit@12

RecBole 训练阶段指标见 `configs/sasrecf.yaml`（用于 early stopping）。

---

## 实验文档

| 文档 | 内容 |
|------|------|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | 入门指南与 FAQ |
| [docs/v2_sasrecf_fusion_experiment_report_jul09.md](docs/v2_sasrecf_fusion_experiment_report_jul09.md) | SASRecF 四路融合完整实验报告 |
| [docs/v1_experiment_report.md](docs/v1_experiment_report.md) | SASRec + Popular + ItemCF 实验 |
| [docs/two_experiments_chronicle.md](docs/two_experiments_chronicle.md) | 两次实验对照与时间线 |
| [docs/sasrec_recbole_comparison.md](docs/sasrec_recbole_comparison.md) | Offline vs RecBole 评估口径说明 |

---

## Git 与大文件

`.gitignore` 已忽略：

- `data/raw/**`、`data/processed/**` 中的大 CSV
- `outputs/**` 中的 checkpoint、召回 CSV、日志
- Python 缓存与虚拟环境

克隆后按上述流程自行准备数据并运行即可。
