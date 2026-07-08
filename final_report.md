# Day 1 雙模型可行性驗證報告(v2 Final)

> **判定:DAY-1 完整通過(6/6)。** 佈局 v2(arena 切片對調 + heap 縮減)修復了 v1 的 gesture 延遲超標;本輪 Summary 的 PASS 為修正判定邏輯後的可信結果。

## 環境
- 板子:NuMaker-M55M1(Cortex-M55 @ 220 MHz + Ethos-U55 256-MAC, 1.5MB SRAM, 8MB HyperRAM)
- 專案:`m55m1_dual_model_poc`(繼承 `Visually-Impaired-Assistance-System`,策略 A:SD → HyperRAM)
- Vela:`Ethos_U55_256` / `Ethos_U55_High_End_Embedded` / `Shared_Sram`
- 報告日期:2026-07-07;計時:SysTick 64-bit fallback(本板 DWT CYCCNT 不可用)
- 架構:雙 interpreter **常駐**(各自 arena、各 Init 一次),切換 = 直接 invoke 另一物件

## 最終記憶體佈局(v2,實測位址)
```
SRAM01 段(heap 由 256KB 縮至 64KB 後,arena 區段基址 0x81F10400):
GESTURE arena 320 KB @0x81F10400 ── 整塊在 SRAM01(用量 299,848 B)✓
FACE    arena 896 KB @0x81F60400 ── 頭部(activations)在 SRAM01,
                                    尾部 persistent 區 ~196 KB 溢入 HyperRAM guard
HyperRAM (8 MB):
0x82000000 ─ 1 MB front guard(吸收 FACE arena 尾部)
0x82100000 ─ FACE 模型 2,854,848 B(slot 3 MB)
0x82400000 ─ GESTURE 模型 2,431,552 B(slot 3 MB)
0x82700000 ─ 1 MB rear spare
```

## Vela 編譯結果
| 項目 | face_model | gesture_model |
|---|---|---|
| 編譯後大小 | 2,854,848 B | 2,431,552 B(沿用舊已編譯模型)|
| 算子 / CPU ops | **1(ethos-u)/ 0** = 100% NPU | **1(ethos-u)/ 0**(板上實證)|
| MACs / 推論 | 395,876,160 | — |
| Vela 預估推論 | 15.56 ms @500MHz → ~35.4 ms @220MHz(實測 42.7,達預估 83%)| — |
| Vela NPU-active cycle | **81.5%** | — |
| Arena 實測用量 | 851,204 B | 299,848 B |

## 載入測試(Test 1)— PASS
| 模型 | 大小 | 載入 | 吞吐 | CRC32 自檢 |
|------|-----:|-----:|-----:|:---:|
| 人臉 | 2,854,848 B | 373 ms | ~7.3 MB/s | 0x21B4B873 ✅ |
| 手勢 | 2,431,552 B | 327 ms | ~7.1 MB/s | 0xEBDA1FD1 ✅ |

4 KB 塊讀 + D-cache clean/invalidate 後二遍 CRC;TFL3 magic 正確。

## Interpreter 建立(一次性)
FACE 66 ms、GESTURE 167 ms(含 AllocateTensors)。

## 推論測試(100 次)— 全 PASS
| 場景 | avg | max | min | sanity | <100ms |
|------|----:|----:|----:|:---:|:---:|
| 人臉單獨 | **42.715 ms** | 42.719 | 42.712 | 63,473 B 非零(99.95%)| ✅ |
| 手勢單獨 | **30.631 ms** | 30.638 | 30.631 | 2,279 B 非零 | ✅ |

(log 之 std 欄位為整數截斷假象;實際抖動 = max−min,µs 級,NPU 高度確定性。)

## 交替測試(Test 4,500 次)— PASS
| 模型 | avg | max | 次數 |
|------|----:|----:|----:|
| FACE | 42.712 ms | 42.714 | 250 |
| GESTURE | 30.637 ms | 30.639 | 250 |

- **500/500 無 crash**、輸出 sanity 全 OK
- 與單獨跑數據一致 → **零切換退化**;切換開銷 ≈ 0(常駐,無 re-Init)
- 交替吞吐:一組 face+gesture ≈ 73.3 ms → **~13.6 組/秒**

## 通過判定(全數通過)
- [x] 兩模型載入不重疊位址 + CRC 自檢
- [x] 人臉推論 < 100 ms + 輸出有效(42.7 ms)
- [x] 手勢推論 < 100 ms + 輸出有效(30.6 ms)
- [x] 500 次交替無 crash + 輸出有效
- [x] CRC 驗證通過
- [x] NPU 利用率 > 70%(算子映射 100%;Vela cycle 81.5%)

## 過程中的關鍵發現(對後續專案通用)
1. **TFLM arena 頭尾差異**:activations 在 arena 頭部(高流量)、persistent 中繼資料在尾部(低流量)。實測:gesture arena 整塊放 HyperRAM → 177.3 ms(5.8×);FACE 僅尾部溢出 → 幾乎零代價(+2%)。**結論:arena 頭部必須在 SRAM01,尾部可溢 HyperRAM。**
2. **NPU 僅能存取 Flash + SRAM01(+ Nuvoton 平台的 HyperRAM)**:SRAM2/DTCM 不可放 arena。
3. **Vela 報告以 500 MHz 假設時脈**:讀時間預估要換算 220 MHz(×2.27)。
4. **CRC/計時等基礎設施 bug 史**(已修):CRC 查表多項式 typo(0xEDB88420→0xEDB88320)、DWT 本板不可用(SysTick fallback)、測試判定漏 AND 延遲條件造成 v1 假 PASS。
5. **切換策略**:MicroMutableOpResolver 不可重複註冊 op → 不能 re-Init 切換;正解 = 雙 interpreter 常駐 + 各自 arena,切換零成本。

## 版本歷程
| 版本 | 佈局 | FACE | GESTURE | 判定 |
|---|---|---:|---:|---|
| v1 | FACE 優先切片,heap 256KB | 41.843 ms | 177.282 ms | ❌ gesture 超標(Summary 假 PASS,判定 bug)|
| **v2** | **GESTURE 優先切片,heap 64KB** | **42.715 ms** | **30.631 ms** | ✅ **6/6 全過** |

## 下一步
1. **Day 2:ESP-12F Wi-Fi 通訊驗證**
2. 相框整合:Slideshow(已實作)+ 真實 camera 輸入 + 兩模型接真實前處理
3. 選配:接 `ethosu_profiler` PMU 取得矽上 NPU 利用率實測值;gesture 換成真正的手勢模型後重驗 arena 佈局
