# Day-2/Day-3 後端（FastAPI）

M55M1 → ESP-12F → Wi-Fi → 這個後端。端點：

| 端點 | 方法 | 用途 |
|---|---|---|
| `/ping` | GET | 連通性 + 延遲測試（Day 2） |
| `/story` | POST | 帶 JSON body 的往返測試（Day 2，保留） |
| `/generate` | POST | `{"face_id":1}` → LLM 生成故事（中+英）+ `audio_id`，TTS 背景開跑（Day 3） |
| `/audio/{audio_id}` | GET | 下載 16kHz/16bit/mono WAV；TTS 沒好會等（最多 90 秒）才回 |
| `/audio/{audio_id}/status` | GET | 查 TTS 狀態 `{"status":"pending/ready/error","size":N}`（除錯用） |

## Day-3 依賴與環境變數

```powershell
pip install -r requirements.txt
# ffmpeg 需在 PATH — 只有語音端點 /audio 需要（Day-3 相框已不播語音，可略過）

# 預設走 NCKU NetDB LiteLLM proxy（https://litellm.netdb.csie.ncku.edu.tw，
# 已內建為預設 LLM_API_BASE）、模型 gemini-3-flash。只需設 proxy 發的 key：
$env:OPENAI_API_KEY = "sk-..."

# 換模型：$env:LLM_MODEL = "..."（proxy 有哪些模型可用下面指令查）
# curl.exe -s -H "Authorization: Bearer sk-..." https://litellm.netdb.csie.ncku.edu.tw/v1/models
# 或不設 key、改設 $env:LLM_MOCK = "1" 用固定假故事測管線
```

## Day-3 手動測試

```powershell
# 1. 生成故事（回 story_text / story_text_en / audio_id）
#    ⚠ PowerShell 下 JSON body 要用單引號包 \" 跳脫，雙引號會被 PS 吃掉
curl.exe -s -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{\"face_id\":1}'

# 2. 查 TTS 狀態（可略過，/audio 本身會等）
curl.exe -s http://127.0.0.1:8000/audio/<audio_id>/status

# 3. 下載 WAV 並確認格式（要 pcm_s16le / 16000 Hz / 1 ch）
curl.exe -s -o test.wav http://127.0.0.1:8000/audio/<audio_id>
ffprobe -hide_banner test.wav
```

WAV 檔存在 `backend/audio_cache/`，用完可整個資料夾刪掉。

## 啟動

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

> ⚠️ `--host 0.0.0.0` 一定要加：讓後端綁定所有網卡，同網段的 M55M1
> 才連得到。綁預設的 `127.0.0.1` 只有本機自己連得到。

uvicorn 預設會印 access log，每筆請求都看得到來源 IP——
板子打進來的第一個請求就是鏈路通了的證據。

## 啟動後的三個確認步驟

### 1. 查電腦的區網 IP

```powershell
ipconfig
```

找目前連 Wi-Fi 那張網卡的「IPv4 位址」（通常長得像 `192.168.x.x`）。
這個 IP 之後要填進韌體的 `BACKEND_HOST`。

### 2. 確認 M55M1（ESP-12F）跟電腦連同一個 Wi-Fi

- ESP8266 只支援 **2.4 GHz**，如果路由器是 2.4G/5G 雙頻分開的 SSID，
  電腦跟板子都要連 **2.4 GHz 那個**。
- 手機熱點也可以，但電腦跟板子都要掛在同一個熱點下。
- 部分公司/校園 Wi-Fi 有 AP isolation（裝置互 ping 不通），有這種情況
  就改用手機熱點。

### 3. 用手機瀏覽器從「外部」測後端

手機連同一個 Wi-Fi，瀏覽器開：

```
http://<電腦IP>:8000/ping
```

看到 `{"status":"ok","timestamp":...}` 才算後端對外可連。
（在電腦自己上開 localhost:8000 成功不算數——那測不到防火牆。）

## Windows 防火牆

手機測不通、但電腦本機開 localhost 通 → 幾乎都是防火牆擋了 8000 port。
用**系統管理員** PowerShell 開一條規則：

```powershell
netsh advfirewall firewall add rule name="M55M1 Day2 FastAPI 8000" dir=in action=allow protocol=TCP localport=8000
```

測完想收掉：

```powershell
netsh advfirewall firewall delete rule name="M55M1 Day2 FastAPI 8000"
```

（另一個常見情況：uvicorn 第一次啟動時 Windows 跳出「允許存取」對話框，
點了「允許」通常就等於開好規則。）
