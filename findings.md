# Findings

本文件记录本轮项目体检发现，外部资料与代码证据分开记录。

## 初步结构

- 项目已具备时间切分、RecBole SASRec/SASRecF、Popular/Category Popular/Item2Item/模型召回、候选并集、Weighted RRF、valid 权重搜索和 run-scoped 产物目录。
- 当前主口径是 H&M 交易数据上的周级 next-item 集合推荐，离线指标以 MAP@12 为主。
- 需要重点验证：训练样本是否仍是单一 next-item 目标；候选覆盖/去重/已购过滤是否符合业务；融合权重是否过拟合单个 valid 周；线上服务、特征一致性和监控是否缺失。

## 代码证据

- `data/build_sequences.py:41-89`：训练集按交互滚动生成 next-item 样本；valid/test 在切分内不滚动，分别使用 train、train+valid 历史，因果性清晰，但目标仍是“购买事件”，没有曝光/点击负反馈。
- `data/split.py:195-276`：固定为 4 周 train + 1 周 valid + 1 周 test；`experiment.yaml` 的 `backtest_windows: 3` 当前只被解析和测试，没有实际回测执行链路。
- `data/filter.py:99-140`：Top-item 集合只在 train 拟合是正确的；但 `min_user_purchases` 与 `max_user_behaviors` 在过滤阶段被明确作为兼容参数忽略，旧文档仍容易让人以为已做用户/行为过滤。
- `recall/item2item.py:69-94`：Item2Item 是用户内去重后的全对全共现计数，未做 PMI/余弦/流行度校正，也没有时间距离、顺序、品牌/价格等相似度。
- `ranking/weighted_rrf.py:18-38`：最终排序只有 `weight / rank`，完全不使用候选原始分、商品属性、价格/库存/新鲜度、用户偏好等特征；`ranking/features.py` 只是为未来 LambdaRank 留出的特征边界，项目里没有实际 ranker 训练/预测。
- `candidates/union.py:29-43`：并集 Top-300 的保留优先级是单通道最佳 rank/score，可能在进入学习排序前就丢掉多通道一致但单路 rank 稍后的商品。
- `evaluation/weight_search.py:153-206`：四个活跃度桶的权重在同一个 valid 周上坐标下降搜索，只有 2 passes/步长 0.05；缺少多时间窗 OOF、置信区间和权重稳定性检验。
- `offline_eval.py:305-333`：主评估是每个用户的购买集合 MAP/Recall/NDCG/Hit@12；没有按曝光、去重购买、品类覆盖、长尾、新品、库存可售等线上约束评估。

## 历史实验基线

- `docs/v2_sasrecf_fusion_experiment_report_jul09.md:261-268,288-297` 记录：SASRecF 全库 test MAP@12=0.0156，四路离线融合 test MAP@12=0.0205；文档也明确指出两者协议不同，不能把 +31.3% 当作严格模型增益。
- 同报告 `:407-409`：约 24%~31% 评估用户没有 SASRecF 召回，cold-start 主要依赖规则通道。这是当前性能上限和工业化优先级的重要证据。
- 当前仓库没有真实 `data/raw`/`outputs` 运行产物（仅 `.gitkeep`），本轮未重新训练；pytest 全部通过（67 passed）。
