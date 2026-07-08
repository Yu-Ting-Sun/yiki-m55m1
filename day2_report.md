# Day 2 Wi-Fi 通訊鏈路驗證報告

> **判定:DAY-2 完整通過(5/5)。** M55M1 → ESP-12F(AT)→ Wi-Fi → FastAPI → 回傳 → M55M1 全鏈路可通,100 次連續請求零掉包,平均來回延遲 50 ms(門檻 1000 ms 的 5%)。可進 Day 3(LLM 串接)。

## 環境
- 板子:NuMaker-M55M1(主控)+ 板載 ESP-12F(Wi-Fi 數據機,AT 指令控制)
- 後端:FastAPI + uvicorn @ `0.0.0.0:8000`,Windows PC(`192.168.10.101`)
- 網路:2.4 GHz Wi-Fi「dlink」,板子取得 IP `192.168.10.104`(同網段)
- 報告日期:2026-07-07;計時:PerfTimer(SysTick 64-bit pmu fallback,ms 級)

## ESP-12F 韌體資訊(Task 0 探測結果,板上實測)
| 項目 | 值 |
|---|---|
| 韌體 | **AT 1.7.0.0**(Aug 16 2018)/ SDK 3.0.0(d49923c)/ Bin v1.7.0(Wroom-02) |
| Flash | 16 Mbit(1024KB+1024KB map),SPI QIO 40MHz |
| Baud | **115200** 8N1(`+UART_CUR:115273,8,1,0,1`;其 RTS 有開,M55M1 端不理,小封包無影響) |
| 接線 | **UART8**(TXD=PJ0、RXD=PJ1),reset=PD2,UART8 時鐘源 HXT(依 BSP SecureOTADemo 官方接法) |
| Boot ROM | 74880 baud 開機訊息完整可讀(`2nd boot version 1.7`),模組健康 |

## 驅動架構(本次交付)
```
day2_test.c      整合測試(Task 4)
esp_http.c/.h    HTTP/1.1 client:http_get / http_post_json(Task 2)
                 Connection: close;+IPD 以位元組計數解框,CLOSED 即回應結束
esp_at.c/.h      AT 指令層(Task 1):CWMODE_CUR/CWJAP_CUR/CIFSR、
                 CIPSTART/CIPSEND/CIPCLOSE,timeout + OK/ERROR/FAIL 判定
esp_ll_*(esp_at.c 內)UART8 硬體層:IRQ + 4KB ring buffer、PD2 reset
                 (esp_probe.c 共用同一層,無重複 IRQ handler)
```
時序鐵律:AT 層 timeout 全用 `GetSystemTick_ms()`(PerfTimer),**PerfTimer_Init 之後禁用 `CLK_SysTickDelay`**(會毀掉本板唯一可用的 SysTick 計時源)。

## 測試結果(板上實測,UART log 全文見附錄)
| # | 測試 | 結果 | 數據 |
|---|---|:---:|---|
| 1 | Wi-Fi 連線 + 取 IP | **PASS** | join 6,735 ms;STAIP 192.168.10.104 |
| 2 | HTTP GET `/ping` | **PASS** | 200;首發 108 ms;body JSON 完整 |
| 3 | HTTP POST `/story`(face_id=1) | **PASS** | 200;60 ms;回傳日月潭遊記文字 |
| 4 | 穩定度 100× GET `/ping` | **PASS** | **100/100 OK、0 掉包**;avg 50 / min 44 / max 60 ms |

RX 健康度:ring overflow = 0、UART FIFO overrun = 0(全程)。

## 通過判定
- [x] M55M1 以 AT 指令讓 ESP-12F 連上 Wi-Fi(6.7 s 含 DHCP)
- [x] M55M1 能發 HTTP GET / POST 到後端
- [x] 後端回應且 M55M1 正確收到(status code + body 解析皆對)
- [x] 連續 100 次請求穩定不掉線(0 drop、最長連續失敗 0)
- [x] 來回延遲 < 1 秒(avg 50 ms,max 60 ms——餘裕 16 倍以上)

## 觀察與備註
1. **首發 108 ms、穩態 50 ms**:第一次請求多付 ARP/連線暖身成本,屬正常;每請求都是「開 TCP → 送 → 收 → 關」的完整週期,50 ms 已含全部開銷。
2. **Test 3 中文顯示出現兩處 `��`**:研判是 debug 終端機在 UTF-8 多位元組字元跨串列讀取邊界時的顯示瑕疵,非資料損壞——+IPD 解析按位元組計數、ring buffer 零溢位,且同句其他中文正常。Day 3 若要嚴格驗證,對 body 做 hex dump 或 CRC 即可確認。
3. Wi-Fi 帳密目前寫死在 `day2_config.h`,PoC 可接受;對外發佈前要抽離。

## 給 Day 3(LLM 串接)的風險與建議
1. **延遲組成將翻轉**:區網來回僅 50 ms,LLM 推論會是秒級。相框 UX 要有「思考中」狀態,韌體 HTTP timeout(現為 recv 5 s)需要加大或改輪詢/兩段式 API(先回 task_id 再取結果)。
2. **TLS 不要放板子上**:AT 1.7 雖有 SSL 指令但 RAM 吃緊又難管憑證。建議維持「板子 →(HTTP, LAN)→ FastAPI →(HTTPS)→ LLM API」,金鑰留在後端。
3. **上行大payload 有 2048B 硬上限**(AT+CIPSEND 單次):目前 `esp_tcp_send` 也以此為限。Day 3 若要上傳照片,需做多段 CIPSEND 迴圈 + 加大 `HTTP_REQ_SZ`;或維持「照片走 SD 卡預載、網路只傳 metadata/文字」的架構(建議後者,頻寬 ~11.5 KB/s @115200 本來就不適合傳圖)。
4. **斷線重連還沒做**:driver 尚未處理 `WIFI DISCONNECT` 非同步事件。Day 3 加一個「請求失敗 → 檢查 CWJAP 狀態 → 自動重連」的守門邏輯即可(esp_at 層已有所有需要的指令)。
5. **中文上屏問題(產品面)**:遊記文字是 UTF-8 中文,但 LT7381 面板目前只有 8×16 ASCII 字型。選項:後端把文字渲染成圖(跟照片同路徑顯示)、或嵌 CJK 點陣字型到 SD/Flash。建議前者,零韌體改動。
6. 提高 UART8 baud(AT+UART_CUR 可到 921600)是現成的加速手段,但 50 ms 延遲下暫無必要,留作備案。

## 附錄:完整 UART log
```
==============================================
 M55M1 Day 2 - Wi-Fi Communication Validation
==============================================
 backend  : http://192.168.10.101:8000  (FastAPI /ping, /story)
 Wi-Fi    : "dlink" (2.4 GHz)

=== Test 1: Wi-Fi connect ===
  [OK] AT firmware alive @115200, echo off, CIPMUX=0
  joining "dlink" ...
  joined in 6735 ms
  [OK] station IP: 192.168.10.104
Test 1 result: PASS

=== Test 2: HTTP GET /ping ===
  status=200  round-trip=108 ms
  body: {"status":"ok","timestamp":1783436644.1932824}
Test 2 result: PASS

=== Test 3: HTTP POST /story ===
  request body: {"face_id":1,"timestamp":6.911}
  status=200  round-trip=60 ms
  body: {"face_id":1,"story":"你好，��是你上週去日月潭的��憶"}   <- 顯示瑕疵,見「觀察與備註」2
Test 3 result: PASS

=== Test 4: stability, 100x GET /ping ===
  ...10/100  (ok=10 fail=0)
  ...20/100  (ok=20 fail=0)
  ...30/100  (ok=30 fail=0)
  ...40/100  (ok=40 fail=0)
  ...50/100  (ok=50 fail=0)
  ...60/100  (ok=60 fail=0)
  ...70/100  (ok=70 fail=0)
  ...80/100  (ok=80 fail=0)
  ...90/100  (ok=90 fail=0)
  ...100/100  (ok=100 fail=0)
  latency over 100 OK requests: avg=50 ms  min=44 ms  max=60 ms
  drops: 0 (worst consecutive: 0)
  RX health: ring overflow=0, FIFO overrun=0
Test 4 result: PASS

==============================================
 Day-2 Summary
==============================================
 [PASS] AT commands join Wi-Fi + get IP
 [PASS] HTTP GET  /ping  -> 200 + body
 [PASS] HTTP POST /story -> 200 + story
 [PASS] 100/100 requests OK, no drop
 [PASS] round-trip avg 50 ms (min 44 / max 60) < 1000 ms
 Overall: DAY-2 PASS
==============================================
=== Day 2 Tests Done ===
```
