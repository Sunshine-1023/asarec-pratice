# FashionRec-Transformer

基于 Transformer 序列推荐模型的时尚推荐系统，使用 [RecBole](https://recbole.io/) 训练 SASRec。

## 项目结构

```
FashionRec-Transformer/
├── data/
│   ├── raw/                  # 原始 H&M 数据
│   │   ├── transactions_train.csv
│   │   ├── articles.csv
│   │   └── customers.csv
│   └── processed/            # 预处理后的 RecBole 格式
│       ├── hm.inter
│       ├── train_labels.pkl
│       └── valid_labels.pkl
├── configs/
│   └── sasrec.yaml
├── src/
│   ├── data/
│   │   ├── preprocess.py     # CSV → hm.inter
│   │   └── split.py          # 训练/验证标签划分
│   ├── metrics/
│   │   └── mapk.py           # MAP@K 评估
│   └── service/
│       └── app.py            # FastAPI 推理服务
├── outputs/
│   ├── logs/
│   ├── recommendations/
│   └── checkpoints/
├── run_sasrec.py
├── requirements.txt
└── README.md
```

## 环境准备

```bash
conda activate ai
pip install -r requirements.txt
```

## 数据准备

将 H&M Kaggle 数据集放入 `data/raw/`：

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

## 使用流程

### 1. 数据预处理

```bash
python -m src.data.preprocess
python -m src.data.split
```

### 2. 训练模型

```bash
python run_sasrec.py
```

跳过预处理（已有 `hm.inter` 时）：

```bash
python run_sasrec.py --skip-preprocess
```

### 3. 启动推理服务

```bash
uvicorn src.service.app:app --reload --host 0.0.0.0 --port 8000
```

请求示例：

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "12345", "top_k": 12}'
```

## 评估指标

使用 `src/metrics/mapk.py` 计算 MAP@12，与 H&M 竞赛评估方式一致。

## 多路融合与统一评估

已提供三路召回：
- `src/recall/popular.py`
- `src/recall/itemcf.py`
- `src/recall/sasrec_recall.py`

并提供统一融合评估：
- `src/fusion/weighted_fusion.py`
- `src/evaluate/offline_eval.py`
- `run_fusion_eval.py`

### 运行顺序

1. 先训练 SASRec（生成 checkpoint）：

```bash
python run_sasrec.py --skip-preprocess
```

2. 导出 SASRec 召回结果（每个用户 Top-100）：

```bash
python -m src.recall.sasrec_recall --top-k 100
```

3. 运行多路融合 + 统一评估（valid 或 test）：

```bash
python run_fusion_eval.py --eval-split valid
python run_fusion_eval.py --eval-split test
```

### 数据来源说明

- `data/processed/hm/hm.train.inter`  
  用于构建 Popular / ItemCF 索引，以及评估时用户历史（valid 评估）。
- `data/processed/hm/hm.valid.inter`  
  用于 valid 评估目标；在 test 评估时会并入历史（train+valid）。
- `data/processed/hm/hm.test.inter`  
  用于 test 最终评估目标。
- `outputs/recommendations/sasrec.csv`  
  SASRec 召回通道输入（由 `sasrec_recall.py` 导出）。

### 输出文件

- 融合召回结果：`outputs/recommendations/fusion_valid.csv` / `fusion_test.csv`
- 统一评估指标：`outputs/evaluation/fusion_valid_metrics.json` / `fusion_test_metrics.json`
# asarec-pratice
