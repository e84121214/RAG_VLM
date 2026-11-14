# RAG_VLM

本地文件檢索實驗：比較 BM25、BGE-M3 語義檢索及 RRF 混合檢索。
目前功能為找回原文與評估來源排名，尚未接入生成式回答模型。

## 資料夾導覽

| 資料夾 | 用途 |
|---|---|
| `data/raw/` | 原始報告 PDF、DWG 台帳圖與測試照片 |
| `data/processed/` | 報告 Markdown、分章 PDF；不改寫原始資料 |
| `data/testsets/` | 測試題、標準答案與來源行號 |
| `src/` | 本地檢索、向量建立及排名報告匯出程式 |
| `tests/` | 程式的離線檢查 |
| `outputs/evaluations/` | 三種方法的評估 JSON |
| `outputs/rankings/` | 逐題來源排行榜 HTML 與完整排名 JSON |
| `outputs/vlm/` | 保存的 VLM 測試輸出 |
| `docs/technical/` | Code 節點等技術文件 |
| `docs/diagrams/` | 工作流與架構圖 |
| `docs/progress/` | 工作進度、接續筆記 |
| `meetings/` | 按日期分類的簡報、報告及會前準備 |
| `archive/` | 舊建置工具、驗證中間檔、預覽及初始化範例 |
| `models/` | 已下載的本地模型權重 |
| `.document-cache/` | 向量索引與工具快取 |
| `.venv/` | uv 管理的 Python 環境 |

`pyproject.toml`、`uv.lock`、`.python-version`、`.gitignore` 和本說明保留於根目錄，供 uv 與 Git 使用。

## 平常怎麼執行

在專案根目錄開啟 PowerShell。搬移後，程式路徑改為 `src/`：

```powershell
# 單題檢索
uv run python src/bge_local.py retrieve "鄭子寮一下雨就淹，是什麼原因？" --mode hybrid -k 3

# 匯出每題兩種檢索的前 20 名
uv run python src/bge_local.py rankings

# 純語義／混合評估，結果保存到 outputs/evaluations/
uv run python src/bge_local.py eval --mode semantic
uv run python src/bge_local.py eval --mode hybrid

# BM25 單題檢索，不需要載入模型
uv run python src/local_rag.py retrieve "鄭子寮雨季積水的原因？" -k 3

# 離線程式檢查
uv run python -m unittest discover -s tests
```

### 原題庫與口語改寫題庫

`eval` 現在會在全部題目跑完後，集中列出逐題 `Rank`、`Hit@1/3/5/10`、`RR@10`，再顯示整體平均。
單題 Hit 為 0 或 1；`RR@10` 是首次命中排名的倒數，前十名未命中為 0。`MRR@10` 是各題 RR 的平均，並非單題指標。顯示值取三位小數，平均使用未四捨五入的倒數排名計算。

- 原題庫：`data/testsets/rag_testset_v1.md`，仍保留全部 30 題，其中 24 題有來源可評分。
- 口語配對改寫集：`data/testsets/rag_testset_colloquial_v1.md`，只改寫 Q01–Q24 的問句，答案及來源沿用原題；不含下一階段的一般民眾新問題。
- `--testset` 選擇的是**問題清單**，不是要搜尋的報告。兩次仍搜尋同一份報告 Markdown，使用相同模型與索引，不需重跑 `download` 或 `index`。

```powershell
# 原題庫：混合檢索
uv run python src/bge_local.py eval --mode hybrid --testset data/testsets/rag_testset_v1.md --output outputs/evaluations/original_hybrid_20261114.json

# 口語改寫題庫：混合檢索
uv run python src/bge_local.py eval --mode hybrid --testset data/testsets/rag_testset_colloquial_v1.md --output outputs/evaluations/colloquial_hybrid_20261114.json
```

比較三種方法時，兩份題庫各跑一次 `--mode bm25`、`--mode semantic`、`--mode hybrid`，並將輸出檔名中的方法一起修改，避免覆寫。亦可省略 `--output`，程式預設檔名會包含方法與題庫名稱；相同方法、相同題庫重跑時仍會覆寫同名檔。
要閱讀改寫題的完整候選片段，可使用 `uv run python src/bge_local.py rankings --testset data/testsets/rag_testset_colloquial_v1.md`。

### 一次產生三種方法的比較總表

```powershell
uv run python src/bge_local.py rankings --testset data/testsets/rag_testset_colloquial_v2.md
```

此指令各計算一次 BM25 與語義排名，再用相同候選計算原設定 RRF，不需另外執行三次 `eval` 才能比較。模型只在本次程序中載入一次，沿用既有文件索引。
每次在 `outputs/rankings/` 產生同名的三份檔案，檔名包含題庫名稱及時間戳，避免不同方法互相覆寫：

- `.md`：三種方法的整體 Hit@1/3/5/10、MRR@10，以及逐題排名、Hit、RR@10 總表。
- `.html`：同一份總表，加上 BM25、語義、RRF 候選的可展開全文與加分明細。
- `.json`：原始候選及設定、題庫／語料雜湊，另含 `evaluation` 三種方法的前十名評估指標與排名。

終端也會印出三種方法的整體指標。候選中第 11 名會在來源排名中保留為 11，但在前十名評估的 Hit 與 RR@10 為 0；沒有標準來源的題目不納入指標。`eval --mode ...` 仍是單一方法評估，不會自動載入其他方法。

此改寫集用來檢查原題措辭對結果的影響，不是獨立的新題驗證集。保留必要地名、日期及資訊需求；應先核對問句等義性，再固定設定對照三種方法及逐題變化。若出現 BM25 下降，也需看語義與混合的變化，不能單憑這一點宣告假設已證實。問句草稿尚未依檢索分數調整。

### 第一階段：只調整 RRF 常數與權重

先用 `rankings` 保存候選排名，再將實際產生的 `.json` 路徑傳給以下指令：

```powershell
uv run python src/bge_local.py fusion-experiment --rankings outputs/rankings/你的排名檔名.json
```

這會讀取同一份各前20名的快照，比較原設定（常數60、權重1:1）、只改常數為10、只改BM25:語義權重為2:1。
不載入模型、不重建索引，並另存 JSON 與 HTML 到 `outputs/experiments/`。結果僅評估有標準來源的24題，所有指標截斷於前10名。
這些是開發集的參數比較，沒有自動選定最佳設定或改動日常檢索的預設。

### 第二階段：改變 chunk 大小與重疊

一次跑一組，所有長度單位都是字元。以下比較 1500／250 與原始排行榜：

```powershell
uv run python src/bge_local.py chunk-experiment --size 1500 --overlap 250 --baseline outputs/rankings/你的原始排名檔名.json
```

下一組可使用 `--size 750 --overlap 125`。後續測重疊時，固定 `--size`，只改 `--overlap`。
`--baseline` 可省略；提供時會先確認文件與測試集的雜湊相同。

- 原始 3000／500 使用既有 `.document-cache/bge_m3_index.npz`。
- 其他組別索引獨立保存在 `.document-cache/chunks/size_大小_overlap_重疊.npz`，驗證相符時可重用。
- 結果保存在 `outputs/experiments/chunks/size_大小_overlap_重疊/時間戳/`。
- 每次會保存完整排行榜的 JSON／HTML，以及 `summary.json`、`summary.html`。
- 同時評估 BM25、語義、混合，混合固定 RRF 常數60、權重1:1、各前20名。第一階段的 `TRIALS` 不影響這個設定。
- 不修改預設切塊或原始文本；實驗命令設定只作用於當次程式執行。
- 切塊不同時需重新計算文件向量，因此會載入本地模型；不需重新下載模型。
- 評估仍用標準行號交集，可能受到片段長度影響，需搭配完整原文人工檢查。原文及標準來源行號均不改動。
- 若要做第三階段的組合測試，可把該組新產生的排行榜 JSON 傳給 `fusion-experiment`，不需要再次建立向量。

使用 `uv run python src/bge_local.py`，讓程式使用專案的 CUDA PyTorch 環境。
直接寫 `uv run src/bge_local.py` 會使用檔頭的獨立腳本依賴管理，可能與專案環境不同。

## 模型與文件向量

這台電腦已下載模型並建立索引，整理資料夾後不需重新下載或重建。
只有模型或實際切塊內容改變時，才需要重新建立索引：

```powershell
uv run python src/bge_local.py index
```

GitHub 保存程式、環境設定、模型版本紀錄、測試集、目前使用的報告 Markdown，以及 `outputs/` 的實驗輸出。
模型權重、來源 PDF、研究文件、簡報及本機快取不納入此 repository。

新電腦下載專案後，先以 `uv sync` 建立套件環境，再執行：

```powershell
uv run python src/bge_local.py download
uv run python src/bge_local.py index
```

模型為 `BAAI/bge-m3`，固定 revision 為 `5617a9f61b028005a4858fdac845db406aefb181`。
這個版本取自本機下載的 Hugging Face metadata，並記錄在 `src/bge_local.py`，下載時會使用同一版本。
報告語料路徑設定在 `src/local_rag.py` 的 `DOCUMENT`。

## 閱讀結果

- 開啟 `outputs/rankings/` 最新的 `.html`，即可用瀏覽器逐題展開片段全文。
- 每題的「查看原設定 RRF 加分明細」也可依混合名次展開完整原文，不必再按片段編號到兩路排名找全文；既有 HTML 不會自動更新。
- `outputs/evaluations/retrieval_eval.json` 是執行 `src/local_rag.py eval` 後的 BM25 評估輸出；本機已存在的結果會被覆寫，若要新實驗，請先備份或使用 BGE 評估指令的 `--output` 指定另一個檔名。
- 原始結果 JSON 保留產生當時的路徑欄位，因此舊檔可能記載搬移前的路徑；研究數據未修改。
- 目前檢索命中定義是與標準來源行號有交集，不等於生成答案正確率；Q25–Q30 無標準行號，不納入原有 24 題命中評估。

上方導覽也包含本機使用的資料夾；不在 GitHub 保存範圍內的資料夾，下載專案後不會出現。執行程式時會建立需要的輸出與快取目錄。
