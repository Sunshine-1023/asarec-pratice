# SASRec 训练记录（2026-07-08）

## 实验概览

| 项目 | 内容 |
|------|------|
| 实验 ID | `SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd` |
| 启动命令 | `python run_sasrec.py`（未加 `--skip-preprocess`，含数据预处理） |
| 训练时间 | 2026-07-08 17:34:16 ~ 19:08:25（约 94 分钟） |
| 设备 | CUDA |
| 配置文件 | `configs/sasrec.yaml` |
| 原始日志 | `log/SASRec/SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd.log` |
| TensorBoard | `log_tensorboard/SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd/` |
| 模型 Checkpoint | `outputs/checkpoints/sasrec/SASRec-Jul-08-2026_17-34-54.pth` |

## 训练状态

- **30 / 30 epoch 全部完成**
- **未触发早停**（`stopping_step=5`，valid NDCG@12 在后期仍有提升）
- 训练结束后 RecBole 自动 test 评估因 PyTorch 2.6+ `torch.load(weights_only=True)` 报错中断
- **模型已成功保存**；后续已通过 `run_sasrec_test.py` 补跑 test 评估

---

## 数据与切分

### 预处理输出（本次 run 重新生成）

| 文件 | 行数 |
|------|------|
| `data/processed/hm/hm.inter` | 2,936,884 |
| `data/processed/hm/hm.train.inter` | 2,559,733 |
| `data/processed/hm/hm.valid.inter` | 196,828 |
| `data/processed/hm/hm.test.inter` | 180,323 |

### 时间窗口

- **train**：`< 2020-09-09`
- **valid**：`2020-09-09 ~ 2020-09-15`
- **test**：`2020-09-16 ~ 2020-09-22`

### 用户数

| 切分 | 用户数 |
|------|--------|
| train | 256,008 |
| valid | 47,990 |
| test | 44,923 |

### RecBole 序列格式（`hm_seq.*.inter`）

| 文件 | 行数 |
|------|------|
| `hm_seq.train.inter` | 2,303,725 |
| `hm_seq.valid.inter` | 153,856 |
| `hm_seq.test.inter` | 145,797 |

---

## 模型与训练超参

| 参数 | 值 |
|------|-----|
| model | SASRec |
| loss_type | CE（全库 softmax 交叉熵） |
| hidden_size | 128 |
| inner_size | 512 |
| n_layers | 2 |
| n_heads | 4 |
| MAX_ITEM_LIST_LENGTH | 50 |
| hidden_dropout_prob | 0.2 |
| attn_dropout_prob | 0.2 |
| learning_rate | 0.001 |
| epochs | 30 |
| train_batch_size | 2048 |
| eval_batch_size | 2048 |
| valid_metric（早停监控） | NDCG@12 |
| stopping_step | 5 |
| eval_step | 1 |
| topk | 12 |

---

## 逐 Epoch 训练记录（Valid 集）

> 说明：`train loss` 为 RecBole 记录的 **全 epoch batch loss 累加和**（非平均值）。  
> 平均每 batch CE ≈ `train loss / 1125`（本 run 约 1125 个 batch/epoch）。

| Epoch | Train Loss | Train Time | Valid NDCG@12 | MAP@12 | Recall@12 | Hit@12 | 保存模型 |
|------:|-----------:|-----------:|--------------:|-------:|----------:|-------:|:--------:|
| 0 | 9252.3249 | 182.7s | 0.0188 | 0.0147 | 0.0329 | 0.0329 | ✅ |
| 1 | 8443.4039 | 182.4s | 0.0209 | 0.0164 | 0.0361 | 0.0361 | ✅ |
| 2 | 8251.6002 | 192.9s | 0.0213 | 0.0166 | 0.0370 | 0.0370 | ✅ |
| 3 | 8154.3493 | 181.4s | 0.0220 | 0.0173 | 0.0378 | 0.0378 | ✅ |
| 4 | 8090.7533 | 180.7s | 0.0220 | 0.0172 | 0.0381 | 0.0381 | ✅ |
| 5 | 8045.2384 | 180.5s | 0.0223 | 0.0175 | 0.0384 | 0.0384 | ✅ |
| 6 | 8010.0426 | 183.7s | 0.0221 | 0.0173 | 0.0381 | 0.0381 | |
| 7 | 7981.4421 | 185.3s | 0.0222 | 0.0174 | 0.0385 | 0.0385 | |
| 8 | 7956.8151 | 185.4s | 0.0225 | 0.0176 | 0.0391 | 0.0391 | ✅ |
| 9 | 7937.3779 | 181.2s | 0.0225 | 0.0176 | 0.0389 | 0.0389 | ✅ |
| 10 | 7919.4174 | 183.7s | 0.0225 | 0.0177 | 0.0387 | 0.0387 | ✅ |
| 11 | 7903.2671 | 183.5s | 0.0225 | 0.0177 | 0.0389 | 0.0389 | ✅ |
| 12 | 7889.5031 | 181.0s | 0.0226 | 0.0178 | 0.0389 | 0.0389 | ✅ |
| 13 | 7875.7086 | 183.1s | 0.0227 | 0.0178 | 0.0391 | 0.0391 | ✅ |
| 14 | 7864.4897 | 183.6s | 0.0226 | 0.0178 | 0.0389 | 0.0389 | |
| 15 | 7853.3990 | 183.8s | 0.0227 | 0.0178 | 0.0392 | 0.0392 | ✅ |
| 16 | 7844.1399 | 181.0s | 0.0227 | 0.0177 | 0.0392 | 0.0392 | ✅ |
| 17 | 7833.8507 | 183.8s | 0.0228 | 0.0178 | 0.0393 | 0.0393 | ✅ |
| 18 | 7824.8065 | 184.3s | 0.0226 | 0.0177 | 0.0389 | 0.0389 | |
| 19 | 7817.0830 | 185.3s | 0.0228 | 0.0178 | 0.0393 | 0.0393 | ✅ |
| 20 | 7809.7097 | 182.2s | 0.0227 | 0.0177 | 0.0393 | 0.0393 | |
| 21 | 7802.3823 | 184.6s | 0.0228 | 0.0179 | 0.0393 | 0.0393 | ✅ |
| 22 | 7795.8794 | 189.3s | 0.0225 | 0.0177 | 0.0386 | 0.0386 | |
| 23 | 7789.0118 | 184.5s | 0.0228 | 0.0179 | 0.0392 | 0.0392 | ✅ |
| 24 | 7783.3794 | 182.6s | 0.0228 | 0.0180 | 0.0391 | 0.0391 | ✅ |
| 25 | 7777.3645 | 184.3s | 0.0227 | 0.0178 | 0.0391 | 0.0391 | |
| 26 | 7771.8935 | 185.1s | 0.0229 | 0.0180 | 0.0393 | 0.0393 | ✅ |
| 27 | 7766.8930 | 184.8s | 0.0229 | 0.0180 | 0.0395 | 0.0395 | ✅ |
| 28 | 7761.4727 | 182.5s | **0.0230** | 0.0180 | **0.0398** | **0.0398** | ✅ |
| 29 | 7757.5717 | 184.6s | **0.0230** | 0.0180 | 0.0397 | 0.0397 | ✅ |

---

## 最佳结果摘要

### Valid（训练内监控，RecBole 口径）

| 指标 | 最佳值 | 出现 Epoch |
|------|--------|-----------|
| NDCG@12 | **0.0230** | 28、29 |
| Recall@12 | **0.0398** | 28 |
| MAP@12 | 0.0180 | 24、26、27、28、29 |
| Hit@12 | **0.0398** | 28 |

### Test（后续补跑，`run_sasrec_test.py`）

来源：`outputs/evaluation/sasrec_test_metrics.json`

| 指标 | 值 |
|------|-----|
| NDCG@12 | 0.0198 |
| MAP@12 | 0.0154 |
| Recall@12 | 0.0345 |
| MRR@12 | 0.0154 |
| Hit@12 | 0.0345 |
| Precision@12 | 0.0029 |

### Test 融合（RecBole 口徑，valid 搜權後固定）

來源：`outputs/evaluation/recbole_fusion_weight_search.json`

| 指標 | 值 |
|------|-----|
| MAP@12 | **0.0157** |
| Recall@12 | 0.0354 |
| NDCG@12 | 0.0202 |
| 融合權重 | popular=0.2, itemknn=0.4, sasrec=0.4 |

> 第一版完整實驗報告（含 Pop / ItemKNN / Fusion RecBole 對照）見 `outputs/evaluation/v1_experiment_report.md`。

---

## Loss 说明

- 配置为 `loss_type: CE`，对 **全 item 空间（约 15001 类）** 做交叉熵
- 随机猜测 CE 基线约为 `ln(15001) ≈ 9.62`
- Epoch 0 平均 batch CE ≈ `9252.32 / 1125 ≈ 8.22`，已优于随机
- Epoch 29 平均 batch CE ≈ `7757.57 / 1125 ≈ 6.90`，持续下降

---

## 训练曲线趋势（简要）

- **Train Loss**：9252 → 7758（单调下降）
- **Valid NDCG@12**：0.0188 → 0.0230（整体上升，后期在 0.0225~0.0230 波动）
- **单 epoch 训练耗时**：约 181~189 秒
- **单 epoch 验证耗时**：约 3.1~3.4 秒

---

## 异常与后续处理

### 训练结束时报错

```
_pickle.UnpicklingError: Weights only load failed
```

- **原因**：PyTorch 2.6+ 默认 `torch.load(weights_only=True)`，RecBole checkpoint 含 optimizer 等非纯权重对象
- **影响**：仅影响 `run_sasrec.py` 末尾自动 test 评估；**不影响 checkpoint 保存**
- **修复**：已在 `src/pytorch_compat.py` 增加兼容补丁

### 补跑 Test 评估

```bash
python run_sasrec_test.py --eval-split test
```

---

## 关联文件

| 类型 | 路径 |
|------|------|
| 模型权重 | `outputs/checkpoints/sasrec/SASRec-Jul-08-2026_17-34-54.pth` |
| 训练文本日志 | `log/SASRec/SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd.log` |
| TensorBoard | `log_tensorboard/SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd/` |
| Test 指标 JSON | `outputs/evaluation/sasrec_test_metrics.json` |
| RecBole 融合指标 | `outputs/evaluation/recbole_fusion_weight_search.json` |
| 第一版实验报告 | `outputs/evaluation/v1_experiment_report.md` |
| 两次实验对照 | `outputs/evaluation/two_experiments_chronicle.md` |
| 本报告 | `outputs/evaluation/SASRec-hm_seq-Jul-08-2026_training_report.md` |
