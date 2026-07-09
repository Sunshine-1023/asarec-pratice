# 第一版實驗報告（SASRec + Popular + ItemCF / ItemKNN）

> 實驗日期：2026-07-08  
> 數據集：H&M Personalized Fashion Recommendations（Kaggle）  
> 主框架：RecBole 1.2.1 + 專案自研離線評估  
> SASRec Checkpoint：`outputs/checkpoints/sasrec/SASRec-Jul-08-2026_17-34-54.pth`

本文件彙整**第一版實驗**的實驗邏輯、數據處理、算法設計、模型參數，以及 SASRec / Popular / ItemKNN 與融合後的 **RecBole MAP@12** 結果（SASRec test **0.0154**，Fusion test **0.0157**）。

---

## 1. 實驗目標與整體邏輯

### 1.1 目標

1. 在 H&M 交易數據上訓練序列推薦模型 **SASRec**，並以 **MAP@12** 為主指標評估。
2. 建立兩條傳統召回基線：**Popular（熱門）**、**ItemCF / ItemKNN（物品協同過濾）**。
3. 將三路候選做**加權 rank 融合**，驗證多路融合能否優於單通道。
4. **MAP@12 統一採 RecBole full ranking 口徑**（`hm_seq` benchmark，對全庫 item 排序），便於 SASRec / Pop / ItemKNN / Fusion 橫向對比。

### 1.2 整體流程

```text
原始 CSV
  → 過濾（可選）
  → 預處理為 hm.inter
  → 按時間切分 train / valid / test
  → 轉成 RecBole 序列格式 hm_seq.*.inter
  → 訓練 SASRec（RecBole）
  → 三路召回（Popular / ItemCF / SASRec）
  → 加權融合
  → RecBole 評估（MAP@12 等，全庫排序）
```

### 1.3 評估口徑（RecBole 統一基準）

本報告所有 **MAP@12** 及主結論均採 **RecBole full ranking** 口徑：

| 項目 | 說明 |
|------|------|
| 數據 | `hm_seq.train/valid/test.inter` |
| 協議 | 對全部 item 打分，取 Top-12 |
| 歷史構造 | RecBole benchmark 序列任務內建 |
| 用途 | SASRec 訓練、早停、Pop / ItemKNN / Fusion 統一對比 |
| 實現 | （歷史 RecBole 對照腳本已移除）現用 `run_rule_recall.py`、`run_fusion_weight_search.py`、`run_offline_eval.py` |

> 專案另有 offline 召回–融合評估（Top-100 候選池再融合），其 MAP 與 RecBole **不可直接橫比**；本報告不以其作為主結論。

---

## 2. 數據處理

### 2.1 原始數據

| 文件 | 說明 |
|------|------|
| `data/raw/transactions_train.csv` | 用戶–商品購買交易（必要） |
| `data/raw/articles.csv` | 商品側信息（過濾與後續特徵用） |
| `data/raw/customers.csv` | 用戶側信息（過濾用） |

### 2.2 過濾（可選，`src/data/filter.py`）

預設策略：

| 參數 | 預設值 | 含義 |
|------|--------|------|
| `MONTHS` | 3 | 只保留最近 3 個月交易 |
| `TOP_ITEMS` | 30,000 | 保留窗口內購買次數最高的商品 |
| `MIN_USER_PURCHASES` | 5 | 用戶至少 5 次購買 |
| `MAX_USER_BEHAVIORS` | 50 | 每用戶最多保留 50 條行為 |

輸出：`data/raw/filtered/`。

**第一版實際過濾窗口**（本次 run 日誌）：`2020-06-22 ~ 2020-09-22`（約 93 天 / 3 個月）。

> 第一版實際訓練所用交互約 **15,000 items** 量級（RecBole 日誌：`The number of items: 15001`），來源為過濾後再經預處理／切分後的有效集合。

### 2.3 預處理（`src/data/preprocess.py`）

1. 優先讀取 `data/raw/filtered/transactions_train.csv`（無則用完整 `transactions_train.csv`）。
2. 保留最近 3 個月；用戶最少購買次數 ≥ 5。
3. 轉成 RecBole 交互格式 `hm.inter`：

| 列名 | 含義 |
|------|------|
| `user_id:token` | `customer_id` |
| `item_id:token` | `article_id` |
| `timestamp:float` | Unix 秒時間戳 |

第一版實際產出：

| 文件 | 行數 |
|------|------|
| `data/processed/hm/hm.inter` | 2,936,884 |

### 2.4 時間切分（`src/data/split.py`）

按最近自然日窗口切分（**valid 7 天 + test 7 天**）：

| 切分 | 時間窗口 | 交互數 | 用戶數 |
|------|----------|--------|--------|
| train | `< 2020-09-09` | 2,559,733 | 256,008 |
| valid | `2020-09-09 ~ 2020-09-15` | 196,828 | 47,990 |
| test | `2020-09-16 ~ 2020-09-22` | 180,323 | 44,923 |

輸出：

- `hm.train.inter` / `hm.valid.inter` / `hm.test.inter`

### 2.5 RecBole 序列格式轉換

訓練前將三份 `.inter` 轉為序列 benchmark（每行一個「歷史序列 → 下一個 item」樣本）：

| 文件 | 行數 |
|------|------|
| `hm_seq.train.inter` | 2,303,725 |
| `hm_seq.valid.inter` | 153,856 |
| `hm_seq.test.inter` | 145,797 |

主要字段：`user_id`, `item_id_list`, `item_length`, `item_id`, `timestamp`。

### 2.6 RecBole 加載後統計（第一版訓練日誌）

| 指標 | 數值 |
|------|------|
| users | 257,278 |
| items | 15,001 |
| interactions（序列樣本合計語義下的統計） | 2,603,378 |
| 平均用戶行為長度 | ≈ 10.12 |
| 稀疏度 | ≈ 99.93% |

### 2.7 一鍵訓練與數據再生（`run_sasrec.py`）

第一版未分步手動跑 filter，而是直接：

```bash
python run_sasrec.py   # 未加 --skip-preprocess
```

腳本內部順序：

1. `build_inter_file()` → `hm.inter`（若 filtered 存在則優先讀取）
2. `split_by_time()` → train / valid / test
3. 將三份 `.inter` 轉為 `hm_seq.*.inter` 序列 benchmark
4. RecBole 訓練 SASRec 並保存 checkpoint

> 因此 Jul-08 實驗的數據與模型來自**同一次 run**，保證一致。

---

## 3. 算法設計

### 3.1 SASRec（主模型）

- **類型**：Self-Attentive Sequential Recommendation（Transformer 風格自注意力序列推薦）。
- **輸入**：用戶歷史 item 序列（截斷到 `MAX_ITEM_LIST_LENGTH`）。
- **輸出**：對候選 item（評估時為全庫）打分，取 Top-K。
- **訓練框架**：RecBole，`benchmark_filename = [train, valid, test]`。
- **早停監控**：`valid_metric = NDCG@12`，`stopping_step = 5`。
- **第一版損失**：全庫 **CE（Cross Entropy）**。

Checkpoint：

```text
outputs/checkpoints/sasrec/SASRec-Jul-08-2026_17-34-54.pth
```

訓練狀態：

- 30 / 30 epoch 跑完，未觸發早停。
- 最佳驗證約在 **epoch 28**（NDCG@12 = 0.0230）。
- 訓練結束自動 test 曾因 PyTorch 2.6+ `torch.load(weights_only=True)` 報錯中斷；之後用兼容補丁補跑 test。

### 3.2 Popular（熱門召回）

- **實現**：`src/recall/popular.py`
- **思路**：統計歷史交互中 item 出現頻次，按頻次降序構成全局熱門榜。
- **召回**：對每個用戶取 Top-K，並排除其已購歷史。
- **索引來源**：
  - valid 評估：`hm.train.inter`
  - test 評估：`hm.train.inter + hm.valid.inter`

### 3.3 ItemCF / ItemKNN（物品協同過濾）

| 名稱 | 場景 | 說明 |
|------|------|------|
| **ItemCF** | Offline 召回與融合 | `src/recall/itemcf.py`，基於共現 + 余弦相似度，為每 item 保留 Top 相似鄰居 |
| **ItemKNN** | RecBole 統一評估 / 融合 | RecBole 內建 ItemKNN，與 SASRec 共用同一 token 空間 |

ItemCF 關鍵參數（第一版）：

| 參數 | 值 |
|------|----|
| `min_cooccur` | 2 |
| `top_sim_k` | 100 |
| `max_user_items` | 50 |

召回方式：根據用戶歷史 item，聚合相似鄰居分數，排除已購，取 Top-K。

### 3.4 加權 Rank 融合

- **實現**：`src/fusion/weighted_fusion.py`（Offline 召回融合）；目前可运行流程为 `run_fusion_weight_search.py` + `run_offline_eval.py`。
- **Offline 公式**（僅供召回管線參考，非本報告 MAP 口徑）：

\[
\text{score}(i) = \sum_{c \in \{\text{pop},\text{itemcf},\text{sasrec}\}} w_c \cdot \frac{1}{\text{rank}_c(i)+1}
\]

- 融合後取 **Top-12**（對齊 H&M 競賽 K=12）。
- RecBole 口徑融合（分數加權，valid 網格搜索後固定至 test）最佳：

| 通道 | 權重 |
|------|------|
| popular | 0.2 |
| itemknn | 0.4 |
| sasrec | 0.4 |

**融合搜權細節**（`run_fusion_weight_search.py` → `best_fusion_weights.json`）：

| 項目 | 說明 |
|------|------|
| 搜權集 | valid |
| 網格大小 | 8 組（popular ∈ {0.1,0.2,0.3} × itemknn ∈ {0.2,0.3,0.4}，sasrec = 1 − pop − itemknn） |
| 融合方式 | **分數加權**（非 rank 倒數），三路 full-ranking 分數線性組合後再排序 |
| 固定至 test | valid 最佳權重直接套用 test，不再調參 |

---

## 4. 第一版模型參數（SASRec）

> 以下為**實際跑出 MAP@12=0.0154 那次訓練**的參數（見訓練報告）。  
> 注意：倉庫內 `configs/sasrec.yaml` 後續可能已改為更長序列 / BPR 等優化配置，**不應以後續改動覆寫第一版結論**。

| 參數 | 第一版實際值 |
|------|-------------|
| model | SASRec |
| loss_type | **CE** |
| hidden_size | **128** |
| inner_size | 512 |
| n_layers | 2 |
| n_heads | 4 |
| MAX_ITEM_LIST_LENGTH | **50** |
| hidden_dropout_prob | 0.2 |
| attn_dropout_prob | 0.2 |
| hidden_act | gelu |
| layer_norm_eps | 1e-12 |
| initializer_range | 0.02 |
| learning_rate | **0.001** |
| epochs | 30 |
| train_batch_size | 2048 |
| eval_batch_size | 2048 |
| eval_step | 1 |
| stopping_step | 5 |
| valid_metric | NDCG@12 |
| topk | 12 |
| metrics | MAP, Recall, NDCG, MRR, Hit, Precision |
| seed | 2020（RecBole 默認） |
| device | CUDA |

訓練耗時約 **94 分鐘**（2026-07-08 17:34 ~ 19:08）。

### 訓練過程摘要

| 階段 | Train Loss（累加） | Valid NDCG@12 | Valid MAP@12 |
|------|-------------------:|--------------:|-------------:|
| Epoch 0 | 9252.32 | 0.0188 | 0.0147 |
| Epoch 28（最佳附近） | — | **0.0230** | **0.0180** |
| Epoch 29（收尾） | ≈7757.57 | 0.0230 | 0.0180 |

---

## 5. 評估協議（RecBole）

- **Full ranking**：對全部 item 排序（`eval_args.mode: full`）。
- **Top-K = 12**（對齊 H&M 競賽提交 K=12）。
- **SASRec**：加載 valid 最佳 checkpoint（epoch 28）評估。
- **Pop / ItemKNN**：在**同一共享 dataset / token 映射**上 fit + evaluate。
- **Fusion**：valid 上網格搜索權重，固定後在 test 上報告（`recbole_fusion_weight_search.json`）。
- 主指標：**MAP@12**；輔指標：Recall@12 / NDCG@12 / Hit@12 / MRR@12。

---

## 6. 實驗結果：MAP@12 與相關指標（RecBole 口徑）

### 6.1 各模型 Test 集對比（主報告）

| 模型 | MAP@12 | Recall@12 | NDCG@12 | Hit@12 | MRR@12 |
|------|-------:|----------:|--------:|-------:|-------:|
| **Pop** | 0.0033 | — | — | — | — |
| **ItemKNN** | 0.0087 | — | — | — | — |
| **SASRec** | **0.0154** | **0.0345** | **0.0198** | **0.0345** | **0.0154** |
| **Fusion（最佳）** | **0.0157** | **0.0354** | **0.0202** | **0.0354** | **0.0157** |

Fusion 最佳權重（valid 搜索、test 固定）：`popular=0.2, itemknn=0.4, sasrec=0.4`。

#### Test MAP@12 對照（第一版核心數字）

```text
Pop      0.0033
ItemKNN  0.0087
SASRec   0.0154
Fusion   0.0157
```

### 6.2 Valid 集（早停與搜權）

| 模型 | MAP@12 | Recall@12 | NDCG@12 | Hit@12 |
|------|-------:|----------:|--------:|-------:|
| **SASRec**（best epoch 28） | 0.0180 | 0.0398 | 0.0230 | 0.0398 |
| **Fusion（最佳）** | 0.0183 | 0.0412 | 0.0235 | 0.0412 |

### 6.3 結果解讀

1. **單通道排序**：SASRec（0.0154）≫ ItemKNN（0.0087）> Pop（0.0033）。
2. **融合增益**：Fusion test MAP@12 由 0.0154 提升至 **0.0157**（+1.9%），Recall / NDCG / Hit 同步略升。
3. **主導通道**：分數仍高度由 SASRec 主導；Pop / ItemKNN 僅補少量漏召回。
4. **早停依據**：訓練監控 valid NDCG@12（epoch 28 最佳 0.0230），與 MAP@12 趨勢一致。

---

## 10. 實驗過程細節（Jul-08 執行記錄）

> 完整逐 epoch 表見 `SASRec-hm_seq-Jul-08-2026_training_report.md`。

| 項目 | 內容 |
|------|------|
| 實驗 ID | `SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd` |
| 啟動時間 | 2026-07-08 17:34:16 |
| 結束時間 | 2026-07-08 19:08:25（約 **94 分鐘**） |
| 設備 | CUDA |
| 每 epoch 訓練 | ≈ 181–189 秒 |
| 每 epoch 驗證 | ≈ 3.1–3.4 秒 |
| 每 epoch batch 數 | ≈ 1125 |
| 訓練日誌 | `log/SASRec/SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd.log` |
| TensorBoard | `log_tensorboard/SASRec-hm_seq-Jul-08-2026_17-34-16-72e7fd/` |

### 10.1 Loss 與 Checkpoint 選擇

- **Loss**：全庫 CE，約 15001 類；隨機基線 CE ≈ ln(15001) ≈ **9.62**。
- Epoch 0 平均 batch CE ≈ 9252/1125 ≈ **8.22**；Epoch 29 ≈ **6.90**。
- **Checkpoint 選擇依據**：valid **NDCG@12** 最高（epoch 28，0.0230），非 MAP@12。
- Test MAP@12=0.0154 來自該 checkpoint 的歷史 RecBole full-sort 評估（原 test-only 入口已移除）。

### 10.2 異常與補跑

| 問題 | 處理 |
|------|------|
| 訓練末尾 test 評估失敗 | PyTorch 2.6+ `torch.load(weights_only=True)` 與 RecBole checkpoint 不兼容 |
| 修復 | `src/pytorch_compat.py` 補丁 |
| 現有可用命令 | `python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess`（会重训并评估） |

### 10.3 第一版未覆蓋但第二版已做的改進

| 項目 | 第一版 | 第二版（Jul-09） |
|------|--------|------------------|
| 序列模型 | SASRec | SASRecF + 8 維商品特徵 |
| 數據窗口 | **3 個月**（代碼 `MONTHS=3`） | **6 週**（代碼 `WEEKS=6`） |
| 歷史上限 | 50 條/用戶 | 100 條/用戶（全鏈路對齊） |
| 規則召回 | Popular + ItemCF | Popular + Category Popular + Item2Item |
| 融合路數 | 3 路 | 4 路 + **活躍度分層** |
| 主評估口徑 | **RecBole full ranking** | **Offline 召回–融合**（RecBole 作 SASRecF 單模型基準） |
| 權重搜尋 | RecBole 8 組分數加權網格 | Offline 分層座標下降（~335 組/層） |
| 商品規模 | ~15,001 | ~27,436 |
| 早停指標 | NDCG@12 | MAP@12 |

詳細對照見：`outputs/evaluation/two_experiments_chronicle.md`。

---

## 7. 復現命令（第一版路徑）

```bash
# 1) 數據
python -m src.data.filter          # 可選
python -m src.data.preprocess
python -m src.data.split

# 2) 訓練 SASRec（會生成 hm_seq 並訓練）
python run_sasrec.py

# 3) 補跑 RecBole test（若訓練末尾 evaluate 失敗）
python run_sasrec.py --config configs/sasrec.yaml --skip-preprocess

# 4) 匯出 SASRec 召回
python -m src.recall.sasrec_recall --eval-split valid --top-k 100
python -m src.recall.sasrec_recall --eval-split test --top-k 100

# 5) RecBole 統一口徑評估（本報告 MAP@12 來源）
python run_rule_recall.py --eval-split both
python run_fusion_weight_search.py
python run_offline_eval.py --eval-split test --weights-json outputs/evaluation/best_fusion_weights.json
```

---

## 8. 產物索引

| 類型 | 路徑 |
|------|------|
| SASRec checkpoint | `outputs/checkpoints/sasrec/SASRec-Jul-08-2026_17-34-54.pth` |
| SASRec 訓練報告 | `outputs/evaluation/SASRec-hm_seq-Jul-08-2026_training_report.md` |
| SASRec RecBole 指標 | `outputs/evaluation/sasrec_test_metrics.json`、`sasrec_recbole_metrics.json` |
| RecBole 融合網格（MAP 主來源） | `outputs/evaluation/recbole_fusion_weight_search.json` |
| RecBole 對比摘要 | `outputs/evaluation/sasrec_recbole_comparison.md` |
| 兩次實驗對照 | `outputs/evaluation/two_experiments_chronicle.md` |
| 本報告 | `outputs/evaluation/v1_experiment_report.md` |

---

## 9. 第一版結論（摘要）

1. **數據管線完整**：過濾 → `hm.inter` → 時間切分 → `hm_seq` → 訓練 → RecBole 全庫評估 → 三路融合。
2. **RecBole Test MAP@12（主結論）**：
   - SASRec = **0.0154**（顯著優於 Pop 0.0033、ItemKNN 0.0087）
   - Fusion = **0.0157**（valid 搜權後固定，`popular=0.2, itemknn=0.4, sasrec=0.4`）
3. **融合有效但增益有限**：Fusion 略優於 SASRec 單模型，說明 Pop/ItemKNN 有互補，但 SASRec 仍為主導通道。
4. **後續方向**：拉長歷史序列、加大 hidden size、嘗試 BPR / 更低學習率、SASRecF（側特徵）、更細緻融合權重與分數校準。

---

*文檔對應實驗批次：2026-07-08 第一版 SASRec（CE / hidden=128 / L=50 / lr=1e-3）及同期 Popular、ItemCF、ItemKNN、雙口徑融合實驗。*
