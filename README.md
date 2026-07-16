# 憶起 — NuMaker-M55M1 智慧相框 PoC

在 NuMaker-M55M1（Cortex-M55 @220 MHz + Ethos-U55 NPU）上驗證智慧相框的完整技術鏈：

- **Day 1** — 雙 TFLite 模型（人臉 + 手勢 YOLOv8n）共存 HyperRAM、NPU 交替推論（Face 42 ms / Gesture 31 ms，500 次切換零劣化）→ [final_report.md](final_report.md)
- **Day 2** — 板載 ESP-12F（AT 韌體、UART8 115200）連 Wi-Fi 打 FastAPI 後端（100 次請求零掉包，平均 50 ms）→ [day2_report.md](day2_report.md)
- **Day 3** — LLM（gemini-3-flash via LiteLLM proxy）生成家人旅遊故事，後端渲染成中文點陣圖，LCD 左照片輪播、右故事欄同屏顯示
- **Week 2（本 branch `feature/album-filter`）** — 相機（CCAP/HM1055）+ NPU 人臉辨識相簿過濾：相框認得站在面前的人，輪播自動切成他參加過的相簿 → 見下方「[人臉辨識相簿過濾](#人臉辨識相簿過濾week-2featurealbum-filter)」

## Demo 畫面（RUN_DAY3_DEMO=1，預設）

```
┌──────────────────────────┬─┬─────────┐
│                          │ │ 日月潭    │  LT7381 800×480
│   照片輪播 640×480        │琥│ 2026年…  │  ← 右欄內容由後端
│   (4:3 橫拍滿版、          │珀│ ────────│    以微軟正黑體渲染
│    直拍置中填暖黑底)        │線│ 故事內文  │    成 4-bpp 圖傳來
│                          │ │ 19px…   │    (TIM4, 37 KB)
└──────────────────────────┴─┴─────────┘
```

開機流程：畫版面 → Wi-Fi → `POST /generate`（LLM 10–30 秒，右欄顯示狀態）→
`GET /textimg` → 故事上屏 → 左區開始輪播 SD 卡照片。

照片區右下角另有 240×240 即時相機預覽：偵測到臉畫紅框，辨識成功轉綠框並
切換輪播內容（Week 2 功能，見下方專章）。

## Clone 位置（必讀）

`.uvprojx` 用 `..\..\..\..\` 相對路徑指向 BSP 的 `Library/` 與 `ThirdParty/`，
**必須 clone 到 BSP 的這個位置才能建置**：

```
cd <BSP>\SampleCode\NuEdgeWise
git clone https://github.com/Yu-Ting-Sun/yiki-m55m1.git m55m1_dual_model_poc
```

`<BSP>` = `M55M1BSP-3.01.003`，且需完整包含：

| BSP 相依 | 用途 |
|---|---|
| `Library/`（CMSIS + StdDriver + Device，含 `CMSIS\Lib\KEIL\cmsis_dsp.lib`、`cmsis_nn.lib`） | 驅動與 DSP |
| `ThirdParty\tflite_micro\`（含 `Lib\tflu.lib` 預建庫） | TFLM 推論 |
| `ThirdParty\ml-embedded-evaluation-kit\` | ARM ML API/log/profiler 標頭 |
| `ThirdParty\openmv\`（含 `omv\Lib\omv.lib`） | 影像工具庫 |
| `ThirdParty\eigen`、`ThirdParty\FatFs` | 數學庫 / SD 檔案系統 |

> 本專案使用的 BSP 來自 **NuML_Studio v1.0.3** 內附的 M55M1BSP 範本（官方
> 下載的 BSP 若 `Library/` 或上表 ThirdParty 不齊，從 NuML_Studio 範本補）。
> Keil 需安裝 M55M1 DFP（device `M55M1H2LJAE`）與 Nu-Link 驅動。

## 快速開始

### 1. 韌體設定檔

```
copy day2_config.example.h day2_config.h    # 填入 Wi-Fi SSID/密碼、後端 IP
```

沒有 `day2_config.h` 也能建置（自動 fallback 到 example 的佔位值），
但板子會連不上網 — 正式跑一定要填。`day2_config.h` 已被 git 忽略，
憑證不會進版控。

### 2. 後端（PC，與板子同一個 2.4 GHz Wi-Fi；Python 3.10+）

```powershell
cd backend
pip install -r requirements.txt

# LLM 金鑰（擇一）：
$env:OPENAI_API_KEY = "sk-..."   # 走 LiteLLM proxy（預設 NCKU NetDB，模型 gemini-3-flash）
# $env:GEMINI_API_KEY = "..."    # 或直連 Google AI Studio（需清掉 LLM_API_BASE）

uvicorn main:app --host 0.0.0.0 --port 8000
```

防火牆記得放行 8000/TCP（詳見 [backend/README.md](backend/README.md)，
含用手機驗證對外連通的步驟）。語音端點需另裝 ffmpeg，Day-3 demo 不播語音、可略過。

### 3. SD 卡

- `0:\pictures\` — 根目錄放散照（JPG），**子資料夾 = 相簿**（一趟旅程一夾），
  每個相簿放 `label.json`：`{"users": ["user1", "user2"]}` 標記參加者
  （PNG/HEIC 不支援，可用 `scripts/prepare_pictures.py` 轉檔）
- `0:\face_mobilenet.tflite` — 人臉 embedding 模型（Vela 版，3.17 MB，
  從 BSP `SampleCode\NuEdgeWise\FaceRecognition\Model\` 複製）— 人臉辨識必需
- `0:\faces\` — `embeddings.txt`（已註冊使用者的參考向量）與
  `enroll_<label>.raw`（待註冊自拍，見下方註冊流程）
- `face_model.tflite` + `gesture_model.tflite` 放根目錄 — 只有跑 Day-1
  推論測試（`RUN_DAY1_TESTS=1`）才需要，demo 模式不用

### 4. 建置 + 燒錄

Keil MDK（Arm Compiler 6）開 `KEIL\DualModelPoC.uvprojx` 直接 Build（F7）+
Download（F8），或 CLI：

```powershell
& "C:\Keil_v5\UV4\UV4.exe" -b KEIL\DualModelPoC.uvprojx -j0 -o build.log
```

產出 `KEIL\release\DualModelPoC.bin`。UART log 在 115200 8N1。

### 5. 預期結果

上電後 LCD 先出現暖黑故事欄 + 琥珀分隔線，右欄狀態依序
`WiFi connecting` → `Writing story` → `Loading story`，約 15–35 秒後
中文故事上屏，左區開始每 3 秒輪播一張照片。UART 會印每步耗時與完整故事文字。

## main.cpp 模式開關

| 開關 | 預設 | 功能（優先權由上而下） |
|---|---|---|
| `RUN_ESP_PROBE` | 0 | ESP-12F AT 韌體偵測 |
| `RUN_DAY2_TESTS` | 0 | Wi-Fi + HTTP 鏈路驗證（4 項測試） |
| `RUN_DAY3_DEMO` | **1** | 相框版面 + LLM 故事 demo |
| `RUN_SLIDESHOW` | 1 | 純照片輪播（僅當 DAY3=0） |
| `RUN_CAMERA_PREVIEW` | **1** | 相機即時預覽（照片區右下角） |
| `RUN_FACE_DETECT` | **1** | 即時人臉偵測（紅框，需 PREVIEW） |
| `RUN_FACE_RECOG` | **1** | 人臉辨識 + 相簿過濾（綠框，需 DETECT） |
| `RUN_FACE_ENROLL` | 0 | 現場註冊模式：連拍 8 張寫入 SD（需 RECOG） |
| `RUN_PHOTO_ENROLL` | **1** | 開機掃 SD `faces\enroll_*.raw` 自動註冊（需 RECOG） |
| `RUN_DAY1_TESTS` | 0 | 雙模型 NPU 推論測試（需 SD 模型檔） |

## 人臉辨識相簿過濾（Week 2，feature/album-filter）

相框認得站在面前的人：辨識成功後輪播自動切成「他參加過的相簿聯集」
（相簿裡沒有他的照片也照播）；沒有人臉或認不得的人（5 秒）→ 恢復播全部。
2026-07-16 於板上驗證通過，含「App 自拍註冊 → 真人辨識」路線。

### 資料流（每張照片的 3 秒 hold 期間逐格執行）

```
CCAP/HM1055 240×240 RGB565（一次性觸發擷取）
 → yolo-fastest_192_face 偵測（模型編譯進 APROM，512 KB SRAM arena）→ 紅框
 → 取最大臉框 crop → resize → RGB888 → int8（uint8−128）
 → FaceMobileNet embedding（SD → HyperRAM FACE slot，512 KB arena，256 維）
 → cosine 比對 0:\faces\embeddings.txt（門檻 0.50，最佳者勝）→ 綠框
 → debounce（同一人 3 次 → 切 playlist；5 秒沒認到任何人 → 播全部）
 → Slideshow_SetFilter()（label.json 聯集，於照片邊界生效）
```

主要新增檔案：`Camera.c`、`FaceDetect.cpp`、`FaceRecog.cpp`、
`Recognizer.cpp`、`Model/FaceMobileNetModel.cpp`、`Model/NN_Model_INT8.tflite.cpp`
（偵測模型 C array）、`scripts/selfie_to_frame.py`；`Slideshow.c` 重構為
相簿/playlist/filter 架構。

### 使用者註冊（三條路）

| 方式 | 流程 | 定位 |
|---|---|---|
| **照片註冊**（預設開） | `python scripts/selfie_to_frame.py --label user1 自拍.jpg`（可一次多張）→ 產出的 `enroll_user1.raw` 丟進 SD `faces\` → 重開機自動註冊，完成後改名 `.done` | App 註冊路線的裝置端；**不用重燒韌體** |
| 現場註冊（`RUN_FACE_ENROLL=1` 重燒） | 站在鏡頭前，開機自動連拍 8 張取樣寫入 SD | 參考品質最好（live cosine 0.8+） |
| App 註冊（未實作，見交接注意） | App 上傳自拍＋名字 → 後端轉 raw 下發 SD → 重開機 | 產品最終形態 |

### 調參

| 參數 | 位置 | 預設 | 說明 |
|---|---|---|---|
| `RECOG_THRESHOLD` | FaceRecog.cpp | 0.50 | 照片參考的 cosine 峰值僅 ~0.53（live 參考 0.8+），設 0.6 會全部拒絕 |
| `FILTER_SWITCH_HITS` | main.cpp | 3 | 連續同名辨識幾次才切換（穿插的失敗 frame 不重置計數） |
| `FILTER_CLEAR_MS` | main.cpp | 5000 | 多久沒認到任何人就恢復播全部 |

### 交接注意 — App / 後端組必讀

1. **embedding 空間不可混用**。SD 上的 `face_mobilenet.tflite` 是 **Vela
   編譯版**（含 ethos-u custom op），只有板上 NPU 跑得動，一般電腦跑不動，
   repo 裡也沒有原始版。所以**推論一律留在裝置**：後端只負責把自拍轉成
   `enroll_<label>.raw`（照抄 `selfie_to_frame.py`：EXIF 轉正 → 中心方形
   裁切 → 240×240 → RGB565 小端序）下發 SD，板子開機自己算。若未來想在
   後端直接算 embedding，必須取得 Vela 編譯**前**的同一顆 int8 模型，並
   複製一模一樣的前處理（同一個偵測器與框法、同 crop、RGB888、
   int8 = uint8−128）——任何一步不同，向量空間就對不上，比對全滅。
2. **照片 vs 真人的量化差距（2026-07-16 實測）**：同一個人，自拍照參考對
   真人查詢的 cosine 峰值只有 **~0.53**；真人參考對真人是 **0.8+**。因此
   照片註冊的使用者綠框會紅綠交錯（門檻 0.50 餘裕僅 0.03），但 debounce
   保證 playlist 切換穩定。改善方向：每人多張參考照（檔名 `-2`、`-3`
   尾碼）、用相框現場光線拍、或 **progressive enrollment**（首次辨識成功
   後用現場 frame 自動補一條 live 參考，直接升級到 0.8 等級——建議 App
   版實作這個）。
3. **App 功能對應**：註冊 = 上傳自拍＋名字（後端轉檔下發）；相簿上傳 =
   勾選參加者 → 後端把 `{"users": [...]}` 寫進該相簿資料夾的
   `label.json`。`label.json` 與 `embeddings.txt` 都是純文字、後端可直接
   讀寫——這是刻意的設計，韌體只讀不管理。
4. **embeddings.txt 格式（易踩雷）**：一行一人
   `label:v0:v1:...:v255:`，**冒號分隔且行尾也要冒號**——parser 只在遇到
   下一個冒號時收字，用逗號分隔會整行解析成空向量、永遠比對失敗。同名
   多行 = 同一人多張參考，比對取最佳分。
5. **已知問題**：(a) UART 在 ESP Wi-Fi 段之後會亂碼（baud 漂移，未修）——
   驗證辨識請看螢幕框色與實際播放內容，別信後段 log。(b) 辨識切換
   playlist 時照片區會短暫黑屏（清前組殘影＋JPEG 解碼時間，設計如此，
   也可視為切換成功的視覺回饋）。(c) 一次只處理畫面中**最大**的一張臉。

## 後端 API（backend/main.py）

| 端點 | 用途 |
|---|---|
| `POST /generate {face_id}` | 查旅程 → LLM 生成故事（中英）→ 回 `story_text` + `audio_id`，同步渲染故事圖 |
| `GET /textimg/{audio_id}` | 故事欄點陣圖（TIM4：`"TIM4"+w:u16LE+h:u16LE+4bpp`，156×480 = 37,448 B） |
| `GET /textimg/{audio_id}/preview` | 同圖 PNG（瀏覽器檢查用） |
| `GET /audio/{audio_id}` | 故事語音 WAV 16 kHz/16-bit/mono（edge-tts，相框未使用） |
| `POST /trips/*`、`GET /trips*` | 手機 App 的旅程記錄 API（SQLite） |

## HyperRAM 佈局（8 MB）

```
0x82000000  +-------------------------+
            | 1 MB front guard        |  (linker-overflow margin)
0x82100000  +-------------------------+  FACE_MODEL_ADDR
            | FACE slot     (3 MB)    |
0x82400000  +-------------------------+  GESTURE_MODEL_ADDR
            | GESTURE slot  (3 MB)    |
0x82700000  +-------------------------+  SLIDESHOW_FB_ADDR
            | 1 MB rear spare (照片FB) |
0x82800000  +-------------------------+
```

> Week 2 借用：DAY3 模式不從 SD 載 FACE/GESTURE 模型，FACE slot 改放
> 人臉 embedding 模型（3.17 MB，尾端溢入 GESTURE slot 前段——DAY3 不用
> 手勢，安全）。兩顆推論 arena（偵測/embedding 各 512 KB）則在 SRAM01。

## 專案沿革

自 `Visually-Impaired-Assistance-System` fork（策略 A：SD → HyperRAM 載模型）。
`DualModelPoC.uvprojx` 由舊 `ObjectTracker.uvprojx` 裁剪而來（移除視覺追蹤、
UVC、VoicePlayer 等群組，加入雙模型/Wi-Fi/UI 模組）。詳細除錯史與設計決策
見 [final_report.md](final_report.md)、[day2_report.md](day2_report.md)。
