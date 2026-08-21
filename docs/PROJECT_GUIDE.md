# FashionRec 双链路使用指南

本项目使用 H&M 购买数据预测用户下一周期可能购买的商品，最终输出每位用户不重复的 Top-12，并以 MAP@12 为主指标。当前有两条正式、物理隔离的训练链：Baseline 用于稳定对照，Industrial 用于购物篮语义、扩展召回和学习排序实验。

## 1. 应该运行哪条链

| 链路 | 适合场景 | 核心组成 |
|---|---|---|
| Baseline | 快速验证、稳定对照、资源有限 | SASRecF + Popular + Category Popular + Item2Item + Weighted RRF |
| Industrial | 验证 next-basket、PIT 特征和 LightGBM | 单一 SASRecF + 六路规则召回 + LambdaRank + RRF 对照 |

正式入口只有：

```bash
make baseline WITH_FILTER=1
make industrial WITH_FILTER=1
```

也可以直接调用应用 CLI：

```bash
PYTHONPATH=src python -m fashionrec.baseline --help
PYTHONPATH=src python -m fashionrec.industrial --help
```

顶层 `python -m fashionrec` 只提供三个明确入口：`baseline`、`industrial` 和 `profile-data`。

## 2. 当前目录结构

```text
src/fashionrec/
├── shared/                         # 极小、中立的共享内核
│   ├── domain/                     # user/item ID、Candidate
│   ├── interfaces/                 # RecallModel、Ranker 等接口
│   ├── io/                         # CSV/Parquet 基础读写
│   ├── metrics/                    # MAP/Recall/NDCG 等纯指标
│   ├── experiment/                 # 配置、RunContext、产物路径
│   └── runtime/                    # CLI/DAG 执行与运行时辅助
├── baseline/                       # 完整旧协议应用
│   ├── data/
│   ├── models/sasrecf/
│   ├── recall/
│   ├── ranking/
│   ├── evaluation/
│   ├── pipeline/
│   └── cli.py
├── industrial/                     # 完整新协议应用
│   ├── data/
│   ├── models/{sasrecf,lambdarank}/
│   ├── recall/
│   ├── ranking/
│   ├── evaluation/
│   ├── pipeline/
│   └── cli.py
├── cli.py                          # 明确应用路由
├── __init__.py
└── __main__.py
```

旧顶层 `data/recall/ranking/training/evaluation/experiment/pipeline` 等 facade 和对应 import API 已删除。

测试按所有权分类：

```text
tests/
├── shared/
├── baseline/
├── industrial/
├── pipelines/
└── integration/
```

## 3. 配置和产物隔离

正式配置：

```text
configs/baseline/experiment.yaml
configs/baseline/models/sasrecf.yaml
configs/industrial/experiment.yaml
configs/industrial/models/sasrecf.yaml
```

根目录 `configs/experiment.yaml`、`configs/sasrecf.yaml` 和 `configs/sasrec.yaml` 已删除；历史报告仍保留当时的命令记录。

运行产物按 profile 和 run-id 隔离：

```text
outputs/runs/baseline/<run-id>/
outputs/runs/industrial/<run-id>/
```

即使两条链使用相同 run-id，也不会覆盖彼此的处理数据、checkpoint、候选、排序模型、权重或指标。同一 profile 使用相同 run-id 但更换配置时会拒绝续跑。

## 4. Baseline 流程

```text
原始 transactions/articles
→ 时间切分与购物日序列
→ SASRecF 训练
→ valid MAP@12 选择唯一 checkpoint
→ valid/test SASRecF 召回
→ Popular / Category Popular / Item2Item
→ 四路候选并集
→ valid 搜索 Weighted RRF 权重
→ valid/test MAP@12 评估
```

Baseline 不生成 events、basket parquet、next-basket labels、PIT 特征或 LambdaRank 模型。它是一条较轻、可复现的对照链。

## 5. Industrial 流程

```text
原始 transactions/customers/articles
→ user-day-item events
→ baskets、weekly snapshots、next-basket labels
→ PIT user/item/cross features
→ 训练并选择一个 SASRecF checkpoint
→ valid/test 扩展候选
→ 六路规则召回
→ 唯一 SASRecF 为多个 train snapshot 生成序列证据
→ train/valid/test LambdaRank 表
→ 训练一个 LightGBM LambdaRank
→ valid/test 打分
→ RRF 与 LambdaRank 对照评估
```

六路规则召回为 Popular、Category Popular、Item2Item、Repurchase、Style、Content；SASRecF 作为额外序列通道进入候选和 LambdaRank 特征。

Industrial 只训练一个 SASRecF。历史训练快照复用这一 checkpoint，但用户输入历史仍按快照日截断。这是用户选择的“简单复用”协议：

- `history_as_of=true`：输入序列不直接包含快照之后购买；
- `causal_model=false`：模型参数、词表和 checkpoint 选择可能看过快照之后数据；
- 因此 LambdaRank 离线指标可能偏高，不能解释为严格无泄漏 PIT 结果。

## 6. 分阶段运行

分阶段执行必须固定 `PROFILE` 和 `RUN_ID`：

```bash
make data PROFILE=baseline RUN_ID=exp-001 WITH_FILTER=1
make train PROFILE=baseline RUN_ID=exp-001
make select-checkpoint PROFILE=baseline RUN_ID=exp-001
make recall PROFILE=baseline RUN_ID=exp-001
make candidates PROFILE=baseline RUN_ID=exp-001
make weights PROFILE=baseline RUN_ID=exp-001
make evaluate PROFILE=baseline RUN_ID=exp-001
```

Industrial 使用同样的基础目标，并可单独执行学习排序阶段：

```bash
make ranker PROFILE=industrial RUN_ID=exp-002
```

需要复用已训练模型并重跑下游时：

```bash
make downstream PROFILE=baseline RUN_ID=exp-001
```

## 7. 数据与评估语义

- 商品 ID 在内部统一为十位字符串。
- SASRecF 序列按 user-day-item 去重，同一天多件商品不编造先后。
- Industrial 热度、复购和用户历史使用购物篮适配语义，避免数量重复行放大信号。
- Baseline 使用行级未来购买集合评估。
- Industrial 使用 next-basket 去重集合评估，数量不会重复增加 MAP 命中。
- checkpoint 只使用 valid 选择；test 只用于最终评估。
- valid 同时承担选模和调参，因此真正的泛化结论应看锁定参数后的 test。

## 8. 主要产物

```text
outputs/runs/<profile>/<run-id>/
├── resolved_config.json
├── manifest.json
├── data/
├── checkpoints/
│   ├── sasrecf/
│   └── sasrecf_selected.pth
├── recall/
├── candidates/
├── ranking/
├── evaluation/
└── logs/
```

Industrial 的 `ranking/` 还会保存 LambdaRank parquet、模型、特征 schema、打分候选和简单复用报告。

## 9. 验证与常见问题

运行代码级完整检查：

```bash
make check
```

这会验证单元测试、双应用 CLI、顶层明确路由和结构边界，但不会启动 3.2GB 原始交易数据上的全量训练。

常见判断：

- “测试通过”等于代码和微型数据闭环成立，不等于 8GB 显存下全量训练已经实测完成。
- Baseline 和 Industrial 可以并行使用不同 run-id；不要在同一 run 中混用 profile。
- Industrial 指标提升必须同时查看 RRF 对照、候选覆盖率和 `causal_model=false` 警告。
- 想做严格 PIT 的 SASRecF→LambdaRank，需要每个历史快照独立训练序列模型；当前为了训练成本没有采用该方案。

更详细的依赖边界和产物契约见 [ARCHITECTURE.md](ARCHITECTURE.md)。历史实验分数和旧命令只记录在对应版本报告中，不代表当前正式入口。
