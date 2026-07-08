# FashionRec-Transformer

基于 H&M 交易数据的时尚推荐项目，使用 [RecBole](https://recbole.io/) 训练 **SASRec**，并与 **Popular / ItemCF** 进行多路召回融合，最终通过离线评估输出 **MAP@12** 等指标。

## 项目结构

```
FashionRec-Transformer/
├── configs/
│   └── sasrec.yaml                 # SASRec 训练配置
├── data/
│   ├── raw/                        # 原始数据（不上传 Git，见 .gitignore）
│   │   ├── transactions_train.csv
│   │   ├── articles.csv
│   │   ├── customers.csv
│   │   └── filtered/               # 可选：过滤后的采样数据
│   └── processed/
│       ├── hm/                     # 手动时序切分（最终评估主逻辑）
│       │   ├── hm.inter
│       │   ├── hm.train.inter
│       │   ├── hm.valid.inter
│       │   └── hm.test.inter
│       └── hm_seq/                   # RecBole benchmark 序列格式（训练用）
│           ├── hm_seq.train.inter
│           ├── hm_seq.valid.inter
│           └── hm_seq.test.inter
├── outputs/
│   ├── checkpoints/sasrec/         # SASRec 模型权重
│   ├── recommendations/            # 召回与融合结果 CSV
│   ├── evaluation/                 # 离线评估指标 JSON
│   └── logs/
├── src/
│   ├── data/
│   │   ├── filter.py               # 原始数据过滤（时间窗 / Top item / 活跃用户）
│   │   ├── preprocess.py           # CSV → hm.inter
│   │   └── split.py                # 按时间切分 train / valid / test
│   ├── recall/
│   │   ├── popular.py              # 热门召回
│   │   ├── itemcf.py               # ItemCF 召回
│   │   └── sasrec_recall.py        # SASRec Top-K 导出
│   ├── fusion/
│   │   └── weighted_fusion.py      # 加权 rank 融合
│   └── evaluate/
│       └── offline_eval.py         # 多路融合 + 离线评估
├── run_sasrec.py                   # SASRec 训练入口
├── run_fusion_eval.py              # 融合评估入口
├── requirements.txt
└── README.md
```

## 核心流程

项目采用「**两套切分逻辑并存**」：

1. **RecBole 训练逻辑**（让 SASRec 能正常训练并保存 checkpoint）
   - 从 `hm.train/valid/test.inter` 生成 `hm_seq.train/valid/test.inter`
   - RecBole 使用 `benchmark_filename: [train, valid, test]` 直接读取三份 benchmark 文件

2. **最终离线评估逻辑**（项目主评估口径，用过去预测未来）
   - `valid` 评估：历史 = `train`，标签 = `valid`
   - `test` 评估：历史 = `train + valid`，标签 = `test`

## 环境准备

```bash
conda activate ai
pip install -r requirements.txt
```

## 数据准备

将 H&M Kaggle 数据放入 `data/raw/`：

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

### 1. 可选：过滤采样（小规模实验）

```bash
python -m src.data.filter
```

输出到 `data/raw/filtered/`。

### 2. 预处理与切分

```bash
python -m src.data.preprocess
python -m src.data.split
```

生成：

- `data/processed/hm/hm.inter`
- `data/processed/hm/hm.train.inter`
- `data/processed/hm/hm.valid.inter`
- `data/processed/hm/hm.test.inter`

## 训练 SASRec

```bash
python run_sasrec.py --skip-preprocess
```

说明：

- 训练前会自动把 `hm.train/valid/test.inter` 转为 `hm_seq.*.inter`（含 `item_id_list`、`item_length`）
- 设备优先级：`cuda > mps > cpu`
- checkpoint 默认保存到 `outputs/checkpoints/sasrec/`

如需从头跑数据预处理：

```bash
python run_sasrec.py
```

## 导出 SASRec 召回

分别导出 valid / test 两个 split 的召回文件：

```bash
python -m src.recall.sasrec_recall --eval-split valid --top-k 100
python -m src.recall.sasrec_recall --eval-split test --top-k 100
```

输出：

- `outputs/recommendations/sasrec_valid.csv`
- `outputs/recommendations/sasrec_test.csv`

字段格式：`user_id, item_id, score, rank, channel`

## 多路融合与离线评估

```bash
python run_fusion_eval.py --eval-split valid
python run_fusion_eval.py --eval-split test
```

### 评估数据来源

| 阶段 | 用户历史 | Popular / ItemCF 索引 | SASRec 召回 | 标签 |
|------|----------|----------------------|-------------|------|
| valid | `hm.train.inter` | `hm.train.inter` | `sasrec_valid.csv` | `hm.valid.inter` |
| test | `hm.train.inter + hm.valid.inter` | `hm.train.inter + hm.valid.inter` | `sasrec_test.csv` | `hm.test.inter` |

### 融合方式

- 三路召回：Popular + ItemCF + SASRec
- 融合策略：加权 rank 融合（`weight * 1/(rank+1)`）
- 默认权重：`popular=0.2`, `itemcf=0.3`, `sasrec=0.5`

可通过参数调整，例如：

```bash
python run_fusion_eval.py \
  --eval-split test \
  --popular-weight 0.2 \
  --itemcf-weight 0.3 \
  --sasrec-weight 0.5 \
  --final-top-k 12
```

### 输出文件

- 融合推荐：`outputs/recommendations/fusion_valid.csv` / `fusion_test.csv`
- 评估指标：`outputs/evaluation/fusion_valid_metrics.json` / `fusion_test_metrics.json`

## 评估指标

离线评估主指标（`offline_eval.py`）：

- **MAP@12**（主指标）
- Recall@12
- NDCG@12
- Hit@12

RecBole 训练阶段指标见 `configs/sasrec.yaml`（含 MAP、Recall、NDCG 等，用于 early stopping）。

## Git 与大文件

`.gitignore` 已配置忽略：

- `data/raw/**`、`data/processed/**` 中的大文件
- `outputs/**` 中的 checkpoint、CSV、日志
- 常见 Python 缓存与虚拟环境目录

仓库中仅保留目录占位（`.gitkeep`），便于克隆后直接按上述流程运行。
