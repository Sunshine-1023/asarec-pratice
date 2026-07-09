# 第一版 RecBole 口徑對比表（SASRec / Pop / ItemKNN / Fusion）

> **評估協議**：RecBole full ranking，`hm_seq` benchmark（train / valid / test）  
> **Checkpoint**：`outputs/checkpoints/sasrec/SASRec-Jul-08-2026_17-34-54.pth`  
> **實驗批次**：2026-07-08 第一版（詳見 `v1_experiment_report.md`）

---

## Test 集 MAP@12 對照（核心數字）

```text
Pop      0.0033
ItemKNN  0.0087
SASRec   0.0154
Fusion   0.0157
```

| 模型 | MAP@12 | Recall@12 | NDCG@12 | Hit@12 | MRR@12 |
|------|-------:|----------:|--------:|-------:|-------:|
| **Pop** | 0.0033 | — | — | — | — |
| **ItemKNN** | 0.0087 | — | — | — | — |
| **SASRec** | **0.0154** | **0.0345** | **0.0198** | **0.0345** | **0.0154** |
| **Fusion（最佳）** | **0.0157** | **0.0354** | **0.0202** | **0.0354** | **0.0157** |

Fusion 最佳權重（valid 網格搜索、test 固定）：`popular=0.2, itemknn=0.4, sasrec=0.4`。

---

## Valid 集（早停與搜權）

| 模型 | MAP@12 | Recall@12 | NDCG@12 | Hit@12 |
|------|-------:|----------:|--------:|-------:|
| **SASRec**（best epoch 28） | 0.0180 | 0.0398 | 0.0230 | 0.0398 |
| **Fusion（最佳）** | 0.0183 | 0.0412 | 0.0235 | 0.0412 |

> SASRec valid 指標來自訓練日誌（早停監控 **NDCG@12**，epoch 28 最佳）。

---

## 評估說明

### 協議要點

| 項目 | 說明 |
|------|------|
| 數據 | `data/processed/hm_seq/hm_seq.{train,valid,test}.inter` |
| 排序 | 對全庫 item full ranking，Top-K = 12 |
| Pop / ItemKNN | RecBole 內建模型，與 SASRec **共用同一 dataset / token 映射** |
| Fusion | 三路 **分數線性加權**（非 rank 倒數融合），valid 上 8 組網格搜權 |
| 實現 | `run_recbole_channel_eval.py`、`run_recbole_fusion_eval.py` |

### Pop / ItemKNN vs ItemCF

| 名稱 | RecBole 口徑 | Offline 口徑 |
|------|-------------|-------------|
| **Pop / Popular** | ✅ 本表 | ✅ `channel_comparison.md` |
| **ItemKNN** | ✅ 本表 | — |
| **ItemCF** | ❌ 無 | ✅ `channel_comparison.md`（共現 + 餘弦，非 RecBole 內建） |

ItemCF 與 ItemKNN 算法相近，但 **僅 ItemKNN 參與 RecBole 統一評估**；ItemCF 只在 offline 召回管線中使用。

### Fusion 搜權細節

- 搜權集：**valid**
- 網格：8 組（popular ∈ {0.1, 0.2, 0.3} × itemknn ∈ {0.2, 0.3, 0.4}，sasrec = 1 − pop − itemknn）
- 融合公式：`score = w_pop × pop_score + w_itemknn × itemknn_score + w_sasrec × sasrec_score`
- test 指標：固定 valid 最佳權重，**不再調參**

---

## 指標來源

| 模型 / Split | 來源文件 |
|--------------|----------|
| SASRec valid | 訓練日誌 epoch 28（`SASRec-hm_seq-Jul-08-2026_training_report.md`） |
| SASRec test | `outputs/evaluation/sasrec_test_metrics.json` |
| Pop / ItemKNN test | `run_recbole_channel_eval.py` → `recbole_channel_metrics.json`（MAP 見上表） |
| Fusion valid / test | `outputs/evaluation/recbole_fusion_weight_search.json` |
| SASRec 匯總 | `outputs/evaluation/sasrec_recbole_metrics.json` |

---

## 與 Offline 口徑的關係

專案另有 **offline 召回–融合評估**（`run_channel_eval.py` / `run_fusion_eval.py`），協議為：

- 各通道先召回 Top-100，再 rank 融合 Top-12
- 歷史 = 過去切分（valid 用 train；test 用 train+valid）
- 指標見 `outputs/evaluation/channel_comparison.md`

**Offline 與 RecBole 數字不可直接橫比**（用戶集合、候選範圍、融合方式均不同）。  
第一版 **主結論以本表 RecBole 口徑為準**（SASRec 0.0154，Fusion 0.0157）。

---

## 簡要解讀

1. **單通道**：SASRec（0.0154）≫ ItemKNN（0.0087）> Pop（0.0033）。
2. **融合增益**：Fusion test MAP@12 由 0.0154 → **0.0157**（+1.9%），Recall / NDCG / Hit 同步略升。
3. **主導通道**：分數仍高度由 SASRec 主導；Pop / ItemKNN 僅補少量漏召回。
4. **Valid → Test**：各指標均有所下降，符合時間外推。
5. Test 上 `MAP@12 ≈ MRR@12 ≈ 0.0154`，且 `Hit@12 = Recall@12 = 0.0345` → 約 3.45% 用戶 Top-12 至少命中一次。

---

## 復現命令

```bash
conda activate dl
python run_recbole_channel_eval.py      # Pop / ItemKNN / SASRec
python run_recbole_fusion_eval.py       # Fusion 網格搜權 + test
```

---

## 關聯文檔

| 文檔 | 說明 |
|------|------|
| `v1_experiment_report.md` | 第一版完整實驗報告 |
| `SASRec-hm_seq-Jul-08-2026_training_report.md` | SASRec 逐 epoch 訓練記錄 |
| `channel_comparison.md` | Offline 口徑（Popular / ItemCF / Fusion） |
| `two_experiments_chronicle.md` | Jul-08 vs Jul-09 兩次實驗對照 |
