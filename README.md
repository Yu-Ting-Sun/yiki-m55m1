# 憶起 — NuMaker-M55M1 智慧相框 PoC

在 NuMaker-M55M1（Cortex-M55 @220 MHz + Ethos-U55 NPU）上驗證智慧相框的完整技術鏈：

- **Day 1** — 雙 TFLite 模型（人臉 + 手勢 YOLOv8n）共存 HyperRAM、NPU 交替推論（Face 42 ms / Gesture 31 ms，500 次切換零劣化）→ [final_report.md](final_report.md)
- **Day 2** — 板載 ESP-12F（AT 韌體、UART8 115200）連 Wi-Fi 打 FastAPI 後端（100 次請求零掉包，平均 50 ms）→ [day2_report.md](day2_report.md)
- **Day 3** — LLM（gemini-3-flash via LiteLLM proxy）生成家人旅遊故事，後端渲染成中文點陣圖，LCD 左照片輪播、右故事欄同屏顯示

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

- `0:\pictures\` — 手機 JPG 照片（橫拍會滿版顯示；PNG/HEIC 不支援，
  可用 `scripts/prepare_pictures.py` 轉檔）
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
| `RUN_DAY1_TESTS` | 0 | 雙模型 NPU 推論測試（需 SD 模型檔） |

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

## 專案沿革

自 `Visually-Impaired-Assistance-System` fork（策略 A：SD → HyperRAM 載模型）。
`DualModelPoC.uvprojx` 由舊 `ObjectTracker.uvprojx` 裁剪而來（移除視覺追蹤、
UVC、VoicePlayer 等群組，加入雙模型/Wi-Fi/UI 模組）。詳細除錯史與設計決策
見 [final_report.md](final_report.md)、[day2_report.md](day2_report.md)。
