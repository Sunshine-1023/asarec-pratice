# SASRecF 四路融合實驗報告（第二版主線）

> **實驗日期**：2026-07-09  
> **數據集**：H&M Personalized Fashion Recommendations（Kaggle）  
> **主框架**：RecBole 1.2.1 + 專案自研離線評估  
> **序列模型**：SASRecF（帶 8 維商品類別特徵）  
> **Checkpoint**：`outputs/checkpoints/sasrecf/SASRecF-Jul-09-2026_10-49-42.pth`  
> **狀態**：✅ 主線 17 步全部完成（含 test 最終評估）

---

## 1. 實驗目標

1. 在 H&M 交易數據上訓練 **SASRecF** 序列推薦模型，以 **MAP@12** 為主指標。
2. 建立三路規則召回：**Popular**、**Category Popular**、**Item2Item（共現）**。
3. 將四路候選（SASRecF + 三路規則）做**活躍度分層加權 rank 融合**。
4. 在 valid 集上搜尋各分層最優融合權重，固定後在 test 集做最終評估。
5. 明確區分 **RecBole 全庫排序口徑** 與 **專案離線召回-融合口徑**，避免指標橫比誤解。

---

## 2. 實驗環境

| 項目 | 配置 |
|------|------|
| Python 環境 | `conda activate dl`（**勿用 base**，base 無 pandas 等依賴） |
| Python 路徑 | `C:\Users\89275\miniconda3\envs\dl\python.exe` |
| GPU | CUDA（訓練時顯存約 8.3G/8.0G，仍可完成） |
| 配置文件 | `configs/sasrecf.yaml` |
| 主指標 | MAP@12（與 H&M 競賽提交 K=12 一致） |

---

## 3. 整體流程

```text
原始 CSV（data/raw/）
  │
  ├─[1] filter.py          時間窗過濾 + Top 商品 + 活躍用戶 + 每用戶最近 100 條
  ├─[2] preprocess.py      → hm.inter
  ├─[3] split.py           → hm.train / valid / test.inter（按週切分）
  ├─[4] build_item_features.py → hm_seq.item（商品特徵）
  │
  ├─[5] run_sasrecf.py     生成 hm_seq benchmark + 訓練 SASRecF
  ├─[6] run_sasrecf_recall.py  匯出 SASRecF Top-100 召回 CSV
  │
  ├─[7] offline_eval.py    四路召回 + 活躍度融合 + valid/test MAP@12
  └─[8] weight_search.py   valid 網格搜權 → best_fusion_weights.json
                              ↓
                         test 最終評估（固定權重）
```

---

## 4. 數據處理

### 4.1 過濾參數（`src/data/filter.py`）

| 參數 | 值 | 說明 |
|------|-----|------|
| `WEEKS` | 6 | 保留最近 6 週交易 |
| `TOP_ITEMS` | 30,000 | 窗口內 Top 熱門商品 |
| `MIN_USER_PURCHASES` | 5 | 活躍用戶門檻 |
| `MAX_USER_BEHAVIORS` | 100 | 每用戶最多保留最近 100 條（與 SASRecF `MAX_ITEM_LIST_LENGTH` 對齊） |

### 4.2 時間窗口與切分（`src/data/split.py`）

| 項目 | 值 |
|------|-----|
| 總窗口 | 2020-08-12 ~ 2020-09-22（42 天 / 6 週） |
| Train | 4 週（2020-08-12 ~ 2020-09-08） |
| Valid | 1 週（2020-09-09 ~ 2020-09-15） |
| Test | 1 週（2020-09-16 ~ 2020-09-22） |

### 4.3 數據規模統計

#### `hm` 交互切分（離線評估主數據）

| 切分 | 交互行數 | 用戶數 |
|------|----------|--------|
| 全量 `hm.inter` | 1,125,756 | — |
| Train | 768,247 | 95,143 |
| Valid | 185,140 | 35,851 |
| Test | 172,369 | 33,840 |

#### `hm_seq` 序列 benchmark（RecBole 訓練用）

| 切分 | 序列行數 |
|------|----------|
| Train | 673,104 |
| Valid | 102,245 |
| Test | 106,770 |

| 項目 | 數值 |
|------|------|
| 商品數（含 padding） | 27,436 |
| 商品特徵數 | 8（見 §5.2） |
| `hm_seq.item` 行數 | 27,435 |

### 4.4 兩套切分邏輯說明

| 邏輯 | 用途 | 歷史定義 |
|------|------|----------|
| **hm**（手動時序切分） | 離線評估主口徑 | valid：歷史=train；test：歷史=train+valid |
| **hm_seq**（RecBole benchmark） | SASRecF 訓練與全庫推理 | 由 RecBole 序列任務定義 |

> 兩套口徑的 MAP@12 **不可直接橫比**，但可用於分別衡量「全庫排序能力」與「召回-融合管線效果」。

### 4.5 洩漏檢查（已確認）

- Train 與 Test 在 `(user_id, item_id, timestamp)` 層面**無交集**（按時間切分，無未來標籤洩漏）。
- 用戶可跨切分重複出現，屬時序推薦正常現象。

---

## 5. SASRecF 模型訓練

### 5.1 模型與超參（`configs/sasrecf.yaml`）

| 類別 | 參數 | 值 |
|------|------|-----|
| 模型 | `model` | SASRecF |
| 序列長度 | `MAX_ITEM_LIST_LENGTH` | 100 |
| 隱藏維度 | `hidden_size` | 128 |
| Transformer | `n_layers` / `n_heads` | 2 / 4 |
| FFN | `inner_size` | 512 |
| Dropout | `hidden_dropout_prob` / `attn_dropout_prob` | 0.2 / 0.2 |
| 訓練 | `train_batch_size` / `learning_rate` | 1024 / 0.0005 |
| 早停 | `valid_metric` / `stopping_step` | MAP@12 / 5 |
| 召回 | `recall_top_k` | 100 |

### 5.2 商品特徵（8 維）

```
product_type_name, product_group_name, colour_group_name, section_name,
garment_group_name, department_name, index_name, index_group_name
```

### 5.3 RecBole 訓練結果

| 指標 | Valid（最佳 epoch 6） | Test |
|------|----------------------|------|
| **MAP@12** | **0.0197** | **0.0156** |
| Recall@12 | 0.0399 | 0.0324 |
| NDCG@12 | 0.0244 | 0.0195 |
| Hit@12 | 0.0399 | 0.0324 |

- 早停於 epoch 12，最佳 checkpoint 為 epoch 6。
- 評估模式：`mode: full`（對全庫 ~27,436 商品排序）。

---

## 6. 多路召回

### 6.1 四路通道配置

| 通道 | 實現 | 召回 Top-K | 說明 |
|------|------|------------|------|
| **sasrecf** | `run_sasrecf_recall.py` | 100 | RecBole 全庫推理後取 Top-100 |
| **popular** | `src/recall/popular.py` | 50 | 全局熱門（加權購買次數） |
| **category_popular** | `src/recall/category_popular.py` | 50 | 用戶歷史類別下的熱門商品 |
| **item2item** | `src/recall/item2item.py` | 50 | 共現相似（8 週窗口，top_sim_k=20，seed=10） |

### 6.2 SASRecF 召回覆蓋

| 切分 | hm 評估用戶數 | 成功匯出用戶數 | 缺失用戶數 | 缺失原因 |
|------|--------------|---------------|-----------|----------|
| Valid | 35,851 | 24,858 | 10,993 (30.7%) | 無 train 歷史（cold_start），RecBole valid 數據集無序列 |
| Test | 33,840 | 25,856 | 7,984 (23.6%) | 同上（cold_start） |

缺失用戶在融合時 `sequence` 權重為 0，僅靠規則通道召回。

### 6.3 召回產物

| 文件 | 行數 | 說明 |
|------|------|------|
| `outputs/recommendations/sasrecf_valid.csv` | 2,485,800 | 24,858 用戶 × 100 |
| `outputs/recommendations/sasrecf_test.csv` | 2,585,600 | 25,856 用戶 × 100 |

---

## 7. 融合策略

### 7.1 活躍度分層（`src/fusion/weighted_fusion.py`）

| 分層 | 條件（train 歷史長度） | Valid 人數 | Test 人數 |
|------|----------------------|-----------|----------|
| `high` | ≥ 10 | 5,747 | 6,963 |
| `medium` | 3 ~ 9 | 15,073 | 15,695 |
| `low` | 1 ~ 2 | 4,038 | 3,198 |
| `cold_start` | 0 | 10,993 | 7,984 |

### 7.2 融合公式

對每個通道的候選列表，按排名 `rank`（從 0 起）計算：

```
score(item) += channel_weight × (1 / (rank + 1))
```

同一 item 跨通道得分累加，取得分最高的 **Top-12** 作為最終推薦。

### 7.3 預設權重模板（搜權前基線）

| 分層 | sequence | popular | category_popular | item2item |
|------|----------|---------|------------------|-----------|
| high | 0.60 | 0.10 | 0.10 | 0.20 |
| medium | 0.40 | 0.15 | 0.15 | 0.30 |
| low | 0.15 | 0.35 | 0.25 | 0.25 |
| cold_start | 0.00 | 0.55 | 0.30 | 0.15 |

---

## 8. Valid 權重搜尋

### 8.1 搜尋配置（`src/evaluate/weight_search.py`）

| 項目 | 值 |
|------|-----|
| 演算法 | 座標下降（coordinate descent） |
| 輪數 | 2 passes（第二輪無提升則提前停止） |
| 步長 | 0.05 |
| 目標 | 最大化 valid MAP@12 |
| 比較模式 | `exclude_seen=false` vs `exclude_seen=true`，取較高者 |

### 8.2 `exclude_seen` 說明

| 模式 | 行為 | Valid 最佳 MAP@12 |
|------|------|------------------|
| `exclude_seen=false` | 不排除歷史已購商品 | **0.0226**（已選用） |
| `exclude_seen=true` | 融合時過濾歷史已購 | 0.0102 |

最終選用 **`exclude_seen=false`**（H&M 重購場景下命中更高）。

### 8.3 搜尋後最佳權重

| 分層 | sequence | popular | category_popular | item2item |
|------|----------|---------|------------------|-----------|
| high | 0.60 | 0.05 | 0.05 | 0.30 |
| medium | 0.55 | 0.15 | 0.10 | 0.20 |
| low | 0.25 | 0.40 | 0.15 | 0.20 |
| cold_start | 0.00 | 0.55 | 0.30 | 0.15 |

產物：`outputs/evaluation/best_fusion_weights.json`

### 8.4 搜尋觀察

- 高/中活躍用戶：**提高 sequence 權重**（0.55~0.60）效果最佳。
- 低活躍用戶：**提高 popular 權重**（0.40）優於預設。
- 全分層：**item2item 權重普遍上升**（high 達 0.30）。
- 相對基線 MAP@12 0.0215 → 搜權後 **0.0226**（+4.8%）。

---

## 9. 評估結果匯總

### 9.1 兩套口徑對照

| 方法 | 評估口徑 | Valid MAP@12 | Test MAP@12 |
|------|----------|-------------|-------------|
| SASRecF 單模型 | RecBole 全庫排序 | 0.0197 | 0.0156 |
| 四路融合（預設權重） | Offline 召回-融合 | 0.0215 | — |
| **四路融合（搜權後）** | **Offline 召回-融合** | **0.0226** | **0.0205** |

> Offline 融合 test MAP@12（0.0205）相對 RecBole SASRecF test（0.0156）提升 **+31.3%**。  
> 兩者評估協議不同，此對比反映「管線整體效果」，非同一任務下的嚴格對照。

### 9.2 Offline 詳細指標

#### Valid（預設權重，`fusion_valid_metrics.json`）

| 指標 | 值 |
|------|-----|
| MAP@12 | 0.0215 |
| Recall@12 | 0.0461 |
| NDCG@12 | 0.0348 |
| Hit@12 | 0.1218 |
| 用戶數 | 35,851 |

#### Valid（搜權後，`best_fusion_weights.json`）

| 指標 | 值 |
|------|-----|
| MAP@12 | **0.0226** |

#### Test（搜權後，`fusion_test_metrics.json`）

| 指標 | 值 |
|------|-----|
| **MAP@12** | **0.0205** |
| Recall@12 | 0.0448 |
| NDCG@12 | 0.0334 |
| Hit@12 | 0.1190 |
| 用戶數 | 33,840 |
| exclude_seen | false |

---

## 10. 產物清單

### 10.1 數據文件

```
data/raw/filtered/
  transactions_train.csv    # 過濾後原始交易（1,125,756 行）
  articles.csv
  customers.csv

data/processed/hm/
  hm.inter
  hm.train.inter            # 768,247 行
  hm.valid.inter            # 185,140 行
  hm.test.inter             # 172,369 行

data/processed/hm_seq/
  hm_seq.train.inter        # 673,104 行
  hm_seq.valid.inter        # 102,245 行
  hm_seq.test.inter         # 106,770 行
  hm_seq.item               # 27,435 商品 × 8 特徵
```

### 10.2 模型與召回

```
outputs/checkpoints/sasrecf/
  SASRecF-Jul-09-2026_10-49-42.pth    # 最佳 checkpoint（epoch 6）

outputs/recommendations/
  sasrecf_valid.csv                   # SASRecF valid Top-100
  sasrecf_test.csv                    # SASRecF test Top-100
  fusion_valid.csv                    # valid 融合 Top-12
  fusion_test.csv                     # test 融合 Top-12（最終提交候選）
```

### 10.3 評估與權重

```
outputs/evaluation/
  fusion_valid_metrics.json           # valid 基線融合指標
  fusion_test_metrics.json            # test 最終融合指標
  best_fusion_weights.json            # valid 搜權最佳權重
  sasrecf_fusion_experiment_report_jul09.md  # 本報告
```

---

## 11. 復現命令

> **統一使用 `dl` 環境**，從專案根目錄執行。

```powershell
# 0. 啟用環境
conda activate dl

# 1–4. 數據管線
python src/data/filter.py
python src/data/preprocess.py
python src/data/split.py
python src/data/build_item_features.py

# 5. 訓練 SASRecF（含 hm_seq benchmark 生成）
python run_sasrecf.py

# 6. SASRecF 召回匯出
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test

# 7. valid 基線融合評估
python run_offline_eval.py --eval-split valid

# 8. valid 權重搜尋
python run_fusion_weight_search.py

# 9. test 最終評估（固定搜權結果）
python run_offline_eval.py --eval-split test --weights-json outputs/evaluation/best_fusion_weights.json
```

---

## 12. 已知問題與修復記錄

### 12.1 item_id 格式不一致（已修復）

| 來源 | 格式範例 |
|------|----------|
| `hm.*.inter` 標籤 | `734592001`（9 位） |
| `sasrecf_*.csv`（RecBole 匯出） | `0888024005`（10 位，補前導 0） |

**影響**：修復前 SASRecF 通道在 offline 評估中幾乎無法命中，融合 MAP@12 僅 ~0.006（等同純規則召回）。

**修復**：在 `src/fusion/weighted_fusion.py` 的 `load_channel_recall_csv` 中加入 `normalize_item_id()`（去掉數字型 ID 的前導零）。修復後 SASRecF 單通道 offline MAP@12 ≈ 0.0208，與 RecBole 0.0197 接近。

### 12.2 Python 環境

直接執行 `python xxx.py` 若終端為 `(base)` 會報 `ModuleNotFoundError: No module named 'pandas'`。請使用 `conda activate dl`。

### 12.3 直接執行 `src/**/*.py` 的 import 問題

部分腳本需從專案根目錄以模組方式執行，或腳本內已注入 `sys.path`。推薦：

```powershell
python -m src.evaluate.offline_eval --eval-split valid
```

### 12.4 cold_start 用戶覆蓋缺口

約 24%~31% 的評估用戶無 SASRecF 召回（無 train 歷史），融合完全依賴規則通道，是整體 MAP 的主要下界因素之一。

---

## 13. 結論

1. **SASRecF + 四路融合**在 offline 口徑下優於 SASRecF 單模型：test MAP@12 從 0.0156（RecBole）提升到 **0.0205**（融合管線）。
2. **活躍度分層 + valid 搜權**帶來小幅但穩定的增益（valid 0.0215 → 0.0226）。
3. **規則通道**（popular / category_popular / item2item）對 cold_start 和低活躍用戶不可或缺。
4. 實驗主線已完整閉環；`fusion_test.csv` 可作為最終 Top-12 推薦候選輸出。

---

## 14. 與第一版實驗的差異

| 項目 | 第一版（Jul-08） | 第二版（Jul-09，本報告） |
|------|-----------------|------------------------|
| 序列模型 | SASRec | **SASRecF**（8 特徵） |
| 規則召回 | Popular + ItemCF/ItemKNN | Popular + **Category Popular** + **Item2Item** |
| 數據窗口 | **3 個月**（2020-06-22~09-22，~293 萬行） | **6 週**（2020-08-12~09-22，~112 萬行） |
| 用戶歷史上限 | 50 條/用戶 | **100 條**（全鏈路對齊） |
| 主 MAP 口徑 | **RecBole full ranking** | **Offline 召回–融合** |
| 權重搜尋 | RecBole 8 組分數加權網格 | **分層座標下降** + exclude_seen 對比 |
| 完整過程對照 | — | `two_experiments_chronicle.md` |

---

*報告生成時間：2026-07-09*
