# 兩次實驗完整過程對照（Jul-08 vs Jul-09）

> 本文檔回溯兩次主線實驗的**完整執行過程**、關鍵決策、踩坑記錄與文檔缺口補遺。  
> 詳細報告分別見：  
> - 第一版：`v1_experiment_report.md` + `SASRec-hm_seq-Jul-08-2026_training_report.md`  
> - 第二版：`sasrecf_fusion_experiment_report_jul09.md`

---

## 1. 實驗總覽

| 維度 | 第一版（Jul-08） | 第二版（Jul-09） |
|------|-----------------|------------------|
| 日期 | 2026-07-08 | 2026-07-09 |
| 序列模型 | SASRec | SASRecF（+8 商品特徵） |
| 規則召回 | Popular + ItemCF/ItemKNN | Popular + Category Popular + Item2Item |
| 融合 | 3 路 | 4 路 + 活躍度分層 |
| **主 MAP 口徑** | **RecBole full ranking** | **Offline 召回–融合** |
| SASRec(F) test MAP@12 | **0.0154**（RecBole） | 0.0156（RecBole）/ 0.0205（Offline 融合） |
| 融合 test MAP@12 | **0.0157**（RecBole 分數加權） | **0.0205**（Offline rank 融合） |
| Checkpoint | `SASRec-Jul-08-2026_17-34-54.pth` | `SASRecF-Jul-09-2026_10-49-42.pth` |

> ⚠️ 兩次實驗的融合 MAP **不可直接橫比**：評估協議、數據窗口、通道數均不同。

---

## 2. 數據管線對照

### 2.1 過濾策略

| 參數 | 第一版（當時代碼） | 第二版（當前代碼） |
|------|-------------------|-------------------|
| 時間窗 | `MONTHS=3` | `WEEKS=6` |
| 實際日期（Jul-08 run） | 2020-06-22 ~ 2020-09-22（93 天） | 2020-08-12 ~ 2020-09-22（42 天） |
| TOP_ITEMS | 30,000 | 30,000 |
| MIN_USER_PURCHASES | 5 | 5 |
| MAX 行為/用戶 | **50** | **100** |

### 2.2 切分協議（相同邏輯）

兩版 valid / test 均為**最後各 1 週**；train 取剩餘部分：

| 切分 | 第一版（3 月窗） | 第二版（6 週窗） |
|------|-----------------|-----------------|
| train | < 2020-09-09（~11 週） | 2020-08-12 ~ 2020-09-08（4 週） |
| valid | 2020-09-09 ~ 09-15 | 同上 |
| test | 2020-09-16 ~ 09-22 | 同上 |

### 2.3 規模對照

| 指標 | 第一版 | 第二版 |
|------|--------|--------|
| hm.inter 行數 | 2,936,884 | 1,125,756 |
| train 行數 | 2,559,733 | 768,247 |
| valid 行數 | 196,828 | 185,140 |
| test 行數 | 180,323 | 172,369 |
| RecBole items | 15,001 | 27,436 |
| hm_seq.train | 2,303,725 | 673,104 |

---

## 3. 逐步執行時間線

### 3.1 第一版（Jul-08）

```text
[17:34] python run_sasrec.py
          ├─ preprocess → hm.inter (2,936,884)
          ├─ split → train/valid/test
          ├─ 轉 hm_seq benchmark
          └─ RecBole 訓練 SASRec 30 epoch (~94 min)
[19:08] 訓練結束，自動 test 評估失敗（torch.load）
          └─ checkpoint 已保存 ✅
[後續]  python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess
          → test MAP@12 = 0.0154
[後續]  python run_rule_recall.py --eval-split both
          → 匯出規則三路召回 CSV（popular/category_popular/item2item）
[後續]  python run_fusion_weight_search.py + python run_offline_eval.py --eval-split test --weights-json outputs/evaluation/best_fusion_weights.json
          → valid 搜權後固定到 test，離線 Fusion MAP@12 = 0.0205
[並行]  offline 召回 CSV + run_fusion_eval（非本報告主口徑）
```

### 3.2 第二版（Jul-09）

```text
[上午]  filter → preprocess → split → build_item_features（6 週資料）
[10:49] python run_sasrecf.py
          ├─ 同上數據管線 + hm_seq.item（27,435 商品 × 8 特徵）
          └─ SASRecF 訓練，早停 epoch 12，best epoch 6
[12:12] run_sasrecf_recall.py --eval-split valid（24,858 用戶）
[12:30] offline_eval valid → 基線 MAP@12 = 0.00627（item_id bug）
[修復]  weighted_fusion.normalize_item_id()
[12:40] offline_eval valid → MAP@12 = 0.0215 ✅
[12:46] weight_search.py 完成 → best MAP@12 = 0.0226
[13:24] run_sasrecf_recall.py --eval-split test
[13:25] offline_eval test + best_weights → MAP@12 = 0.0205 ✅
```

---

## 4. 模型與訓練差異

| 項目 | SASRec（v1） | SASRecF（v2） |
|------|-------------|--------------|
| hidden_size | 128 | 128 |
| MAX_ITEM_LIST_LENGTH | 50 | 100 |
| learning_rate | 0.001 | 0.0005 |
| train_batch_size | 2048 | 1024 |
| valid_metric | NDCG@12 | MAP@12 |
| loss | CE | CE |
| 商品特徵 | 無 | 8 維類別特徵 |
| 最佳 epoch | 28（共 30） | 6（早停 12） |
| RecBole valid MAP@12 | 0.0180 | 0.0197 |
| RecBole test MAP@12 | 0.0154 | 0.0156 |
| 訓練耗時 | ~94 min | ~17 min × 12 epoch |

---

## 5. 融合策略差異

| 項目 | 第一版 RecBole 融合 | 第二版 Offline 融合 |
|------|---------------------|---------------------|
| 實現 | （歷史腳本已移除） | `run_fusion_weight_search.py` + `run_offline_eval.py` |
| 融合對象 | 三路 **full-ranking 分數** | 各路 **Top-K 候選 rank** |
| 公式 | 線性分數加權 | `w × 1/(rank+1)` 累加 |
| 搜權 | valid 8 組固定網格 | 分 4 活躍度層 × 座標下降（~335 組/層） |
| exclude_seen | 無 | 搜尋對比 false/true，選 false |
| 最佳權重 | pop=0.2, itemknn=0.4, sasrec=0.4 | 分層權重（見 best_fusion_weights.json） |

---

## 6. 踩坑與修復清單

| # | 問題 | 影響版本 | 修復 |
|---|------|----------|------|
| 1 | PyTorch 2.6 `torch.load(weights_only=True)` | v1 | `pytorch_compat.py` + 補跑 test |
| 2 | `np.long` 不存在（NumPy 2.x） | v2 | `pytorch_compat.py` |
| 3 | 直接跑 `src/**/*.py` → `No module named 'src'` | v2 | 各腳本注入 `sys.path` |
| 4 | `(base)` 環境無 pandas | v2 | 使用 `conda activate dl` |
| 5 | **item_id 前導零不一致**（hm 9 位 vs RecBole 10 位） | v2 | `normalize_item_id()` |
| 6 | cold_start 用戶無 SASRecF 召回（~25–31%） | v2 | 設計限制，靠規則通道 |
| 7 | 代碼 WEEKS=6 但磁碟資料可能是 3 個月舊 run | 當前 | 重跑 filter 前需確認參數 |

### item_id bug 影響（第二版，修復前）

| 狀態 | Offline 融合 MAP@12 |
|------|---------------------|
| bug 存在 | 0.0063（≈純規則） |
| 修復後 | 0.0215（基線）→ 0.0226（搜權） |

---

## 7. 文檔曾缺失、現已補充的細節

| 細節 | 原狀態 | 現位置 |
|------|--------|--------|
| 第一版過濾實際日期 2020-06-22~09-22 | 缺 | `v1_experiment_report.md` §2.2 |
| `run_sasrec.py` 一鍵管線說明 | 缺 | `v1_experiment_report.md` §2.7 |
| RecBole 融合 8 組網格 + 分數加權 | 缺 | `v1_experiment_report.md` §3.4 |
| 實驗 ID / 日誌 / TensorBoard 路徑 | 僅訓練報告有 | `v1_experiment_report.md` §10 |
| CE loss 基線與 checkpoint 選擇邏輯 | 僅訓練報告有 | `v1_experiment_report.md` §10.1 |
| 訓練報告中的 Fusion test 0.0157 | 缺 | `SASRec-hm_seq-Jul-08-2026_training_report.md` |
| 兩次實驗完整對照 | 缺 | 本文檔 |
| 第二版 item_id bug 與修復 | 有 | `sasrecf_fusion_experiment_report_jul09.md` §12 |

### 仍建議後續補充（可選）

1. **第一版 SASRecF 未做**：若要在 RecBole 口徑下公平對比 v1/v2，需用相同數據窗口重跑 SASRecF + RecBole 融合。
2. **第二版 RecBole 四路融合**：目前僅 Offline 融合；Pop/Item2Item 無 RecBole 統一評估腳本。
3. **SASRecF 訓練逐 epoch 表**：v2 尚無類似 v1 的 `training_report.md`。
4. **`sasrec_recbole_comparison.md`**：仍寫「Fusion 無 RecBole 口徑」，與 v1 報告更新後不一致，可同步修訂。

---

## 8. 復現命令速查

### 第一版（RecBole 主口徑）

```bash
conda activate dl
python run_sasrec.py
python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess
python run_rule_recall.py --eval-split both
python run_fusion_weight_search.py
python run_offline_eval.py --eval-split test --weights-json outputs/evaluation/best_fusion_weights.json
```

### 第二版（Offline 融合主口徑）

```powershell
conda activate dl
python src/data/filter.py
python src/data/preprocess.py
python src/data/split.py
python src/data/build_item_features.py
python run_sasrecf.py
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test
python run_offline_eval.py --eval-split valid
python run_fusion_weight_search.py
python run_offline_eval.py --eval-split test --weights-json outputs/evaluation/best_fusion_weights.json
```

---

## 9. 核心結論（跨實驗）

1. **SASRec 系列在 RecBole 口徑下**：兩版 test MAP@12 接近（0.0154 vs 0.0156），SASRecF 略升但增益有限。
2. **融合增益取決於口徑**：
   - RecBole 分數融合（v1）：0.0154 → **0.0157**（+1.9%）
   - Offline rank 融合（v2）：RecBole 單模 0.0156 → Offline 融合 **0.0205**（+31%，不同協議）
3. **工程細節決定指標可信度**：v2 的 item_id 對齊 bug 曾使融合 MAP 低估 3 倍以上。
4. **數據窗口影響巨大**：3 個月（340 萬行）vs 6 週（112 萬行）導致商品數、用戶分布完全不同。

---

*最後更新：2026-07-09*
