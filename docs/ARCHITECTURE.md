# FashionRec 架构说明

## 设计目标

当前代码按“数据契约优先、候选先物化、排序可替换、产物按运行隔离”组织。重构不改变 SASRecF、Popular、Category Popular、Item2Item 和加权 RRF 的算法含义，主要解决 ID 不一致、脚本互相导入、候选重复计算、全局产物串用以及评估静默回退问题。

## 分层与依赖方向

```text
Makefile
        │
        ├── python -m fashionrec.baseline
        └── python -m fashionrec.industrial
                │
                ▼
各应用独立的 data / models / recall / ranking / evaluation / pipeline
        │
        └── shared kernel（domain / interfaces / io / metrics / runtime）
```

对应目录：

```text
src/fashionrec/
├── baseline/
│   ├── data/ models/ recall/ ranking/ evaluation/
│   └── pipeline/           # Baseline 独立 DAG 与 stage builders
├── industrial/
│   ├── data/ models/ recall/ ranking/ evaluation/
│   └── pipeline/           # Industrial 独立 DAG 与 stage builders
├── shared/
│   ├── domain/             # ID 与 Candidate
│   ├── interfaces/         # RecallChannel / Ranker 稳定边界
│   ├── io/                 # 中立 CSV / Parquet I/O
│   ├── metrics/            # MAP/Recall/NDCG 等纯数学实现
│   └── runtime/            # CLI dispatch、argv、pipeline runner/contracts
├── data/ recall/ ranking/ training/ evaluation/ candidates/
│                           # 旧 import 兼容 facade，不承载正式实现
└── pipeline/               # 旧 --profile 兼容 facade，不是正式入口
```

依赖方向固定为 `baseline|industrial application → shared kernel`。Baseline 与 Industrial 禁止互相导入，应用也禁止通过旧顶层 facade 间接调用另一套实现；shared 禁止反向导入任一应用。两套 DAG 的子进程分别调用 `python -m fashionrec.baseline <command>` 和 `python -m fashionrec.industrial <command>`。

## 核心数据契约

- 商品 ID：内部统一为十位字符串，例如 `706016001` 与 `0706016001` 都映射为 `0706016001`。
- 候选记录：固定包含 `user_id`、`item_id`、`channel`、`score`、`rank`、`split`。
- 候选并集：同一用户、商品、通道仅保留一条证据；同一商品来自多个通道时保留多路证据，供融合或学习排序使用。
- 时间因果：切分前不按未来总活跃度筛用户或截断历史；可选 Top-item 集合只在 train 拟合。`hm.model_train.inter` 仅负责模型拟合资格，完整 `hm.train.inter` 仍供 valid 历史与冷启动评估使用。
- 输入确定性：未启用 `--with-filter` 时始终读取原始 transactions；启用后只读取本次刚生成的 filtered transactions，不根据历史文件是否存在自动切换。
- checkpoint 口径：每个 RecBole 验证 epoch 都进入粗筛，按 `valid_metric_bigger` 方向保留真正的指标 Top-5；最终模型只按完整 valid 用户周的 MAP@12 选择，不读取 test 标签。
- checkpoint 隔离：shortlist 目标目录必须为空；相同 run/seed 重跑时直接失败并保留旧文件，复用旧结果应跳过训练，新训练应使用新的 run ID。

## 正式流水线

```text
baseline profile
  数据准备
  → SASRecF 训练
  → valid 用户周 MAP@12 选择 checkpoint
  → SASRecF valid/test 召回
  → 三路规则召回 + SASRecF 四路候选物化
  → valid 权重搜索
  → Weighted RRF valid/test 排序
  → 离线指标与运行清单

industrial profile
  events / baskets / next-basket labels / PIT user features
  → 独立 SASRecF 训练与 valid/test 扩展多路候选
  → 最近 N 个因果 train 快照规则候选
  → 全候选 cross features + train/valid/test ranking parquet
  → LightGBM LambdaRank 训练与打分
  → next-basket RRF vs LambdaRank valid/test 对照
```

两个应用入口分别创建固定 profile 的 `RunContext`，无需通过 `--profile` 选择。`baseline` 与 `industrial` 即使使用相同 run-id，也写入不同顶层命名空间。正式模式默认 `strict=True`：候选文件、序列召回等关键依赖缺失时直接失败；同一应用/run-id 的配置哈希变化也会拒绝续跑。

单次运行目录：

```text
outputs/runs/<profile>/<run_id>/
├── resolved_config.json
├── manifest.json
├── checkpoints/sasrecf/
├── checkpoints/sasrecf_selected.pth
├── recall/
├── candidates/
│   ├── valid.csv
│   └── test.csv
├── ranking/
├── evaluation/
│   └── sasrecf_checkpoint_selection.json
└── logs/
```

## 排序层

baseline 的固定排序器是 `WeightedRRFRanker`。industrial 同时训练 `LightGBMRanker`，但保留同一候选集上的 RRF 作为对照。

`src/fashionrec/industrial/ranking/features.py` 将候选并集透视为“一用户一商品”表，输出用户历史长度、通道覆盖数、最佳通道排名、最大通道分、各通道 `present / score / rank`、训练标签，以及 LightGBM LambdaRank 所需的用户 group sizes；模型训练与推理实现位于 `industrial/models/lambdarank/`。

industrial 的 `ranker-dataset` 会把固定候选、PIT 用户/商品/交叉特征和 next-basket 标签拼成 train/valid/test parquet，再由 `ranker-train` 和 `ranker-predict` 消费。

## 公开入口

正式实验统一使用 Makefile：

```bash
make baseline WITH_FILTER=1
make industrial WITH_FILTER=1
```

`Makefile` 根据 `PROFILE` 选择一个应用入口，但不再调用通用 `fashionrec pipeline --profile`。分阶段运行必须固定 `PROFILE` 和 `RUN_ID`，例如 `make train PROFILE=industrial RUN_ID=exp-001`。分阶段目标会跳过 ranker 执行，但不会把 Industrial 的数据/标签协议降级成 Baseline。

正式应用入口是 `python -m fashionrec.baseline` 与 `python -m fashionrec.industrial`。顶层 `python -m fashionrec` 保留 profile-data 和旧命令兼容，不再承担两条正式训练链的选择职责。

两套应用分别拥有 `configs/baseline/{experiment.yaml,models/sasrecf.yaml}` 与 `configs/industrial/{experiment.yaml,models/sasrecf.yaml}`。`configs/experiment.yaml` 和根目录模型 YAML 仅作为旧调用兼容文件保留。
