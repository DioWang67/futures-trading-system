# MX 微型台指 SMC+PA 量化回測系統

Smart Money Concepts + Price Action 策略的完整回測、參數優化與 Walk-Forward 驗證系統。

## 專案結構

```
trading-bot/
├── src/
│   ├── data/
│   │   └── fetcher.py          # Shioaji 資料擷取 + 模擬資料產生
│   ├── strategy/
│   │   ├── smc.py              # SMC 模組（BOS/CHoCH/OB/FVG）
│   │   ├── pa.py               # PA 模組（Pin Bar/Engulfing）
│   │   └── bt_strategy.py      # Backtrader 策略整合
│   ├── backtest/
│   │   ├── engine.py           # 回測引擎
│   │   ├── optimizer.py        # Optuna 參數優化
│   │   └── walk_forward.py     # Walk-Forward 驗證
│   └── report/
│       └── generator.py        # HTML 報告產生器
├── tests/                      # pytest 單元測試
├── config/
│   └── settings.yaml           # 所有設定（API、策略參數、優化門檻）
├── data/                       # K 棒資料快取
├── reports/                    # 回測報告輸出
├── run_backtest.py             # 主程式入口
└── pyproject.toml
```

## 環境需求

- Python 3.10+
- 永豐 Shioaji API 帳號（模擬帳號即可）

## 安裝

```bash
cd trading-bot
pip install -e ".[dev]"
```

或手動安裝套件：

```bash
pip install backtrader optuna pandas numpy pyyaml jinja2 plotly matplotlib pyarrow pytest
```

## 設定帳號

編輯 `config/settings.yaml`：

```yaml
shioaji:
  api_key: "你的_API_KEY"
  secret_key: "你的_SECRET_KEY"
  simulation: true          # 模擬帳號設 true
```

## 使用方式

### 1. 用模擬資料快速測試

```bash
python run_backtest.py --sample
```

這會用隨機產生的資料跑完整流程：預設參數回測 → Optuna 優化 → Walk-Forward 驗證 → 輸出報告。

### 2. 用真實資料（需要 Shioaji 帳號）

```bash
python run_backtest.py
```

首次會從 Shioaji API 拉取近 3 年的 MX 歷史 K 棒（60 分 + 15 分），
資料會快取在 `data/` 目錄，後續不會重複下載。

### 3. 只跑參數優化

```bash
python run_backtest.py --sample --optimize-only
```

### 4. 只跑 Walk-Forward 驗證

```bash
python run_backtest.py --sample --wf-only
```

## 策略邏輯

### HTF（60 分鐘）趨勢判斷
- 偵測 Swing High / Swing Low
- BOS（Break of Structure）= 趨勢延續
- CHoCH（Change of Character）= 趨勢反轉
- 以最後一個結構突破方向為當前趨勢

### LTF（15 分鐘）進場
1. 找到與 HTF 趨勢同方向的 **Order Block**（BOS 前最後一根反向 K 棒）
2. 價格回測到 OB 區間
3. **FVG**（Fair Value Gap）作為加分過濾條件
4. 出現 **Pin Bar** 或 **Engulfing** 確認進場
5. SL = OB 另一端 + buffer
6. TP = R:R 1.5 或下一個結構高低點（取較近者）

## 參數優化

使用 Optuna 自動搜尋最佳參數，同時達到以下門檻才算達標：

| 指標 | 門檻 |
|------|------|
| 勝率 | > 55% |
| 獲利因子 | > 1.5 |
| 夏普比率 | > 1.2 |
| 最大回撤 | < 15% |
| R:R | > 1.5 |

可在 `config/settings.yaml` 的 `optimization` 區塊調整。

## Walk-Forward 驗證

- 3 年資料切 6 段
- 每段前 70% 訓練（Optuna 優化），後 30% 測試
- 每段測試都要達標才算真正通過

## 輸出報告

報告自動輸出到 `reports/` 目錄，HTML 格式包含：

- 績效指標卡片（勝率、PF、夏普、MDD、R:R）
- Plotly 互動式權益曲線圖
- 每筆交易紀錄表格
- Walk-Forward 各區段詳細結果

## 跑測試

```bash
python -m pytest tests/ -v
```

## 可調參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `swing_lookback` | Swing point 回看期數 | 5 |
| `bos_min_move` | BOS 最小突破點數 | 15 |
| `ob_max_age` | OB 最大存活 K 棒數 | 20 |
| `ob_body_ratio` | OB 最小實體比 | 0.4 |
| `fvg_min_gap` | FVG 最小缺口點數 | 5 |
| `fvg_enabled` | 是否啟用 FVG 過濾 | true |
| `pin_bar_ratio` | Pin bar 影線/K棒比 | 0.6 |
| `engulf_ratio` | 吞噬最小包覆比 | 1.0 |
| `rr_ratio` | 風報比 | 1.5 |
| `sl_buffer` | 停損 buffer 點數 | 2 |
