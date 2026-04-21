# Trading Bot

See [ROADMAP.md](ROADMAP.md) for the current TMF live-trading gaps and next-step plan.

TradingView webhook 自動化交易系統，同時路由訊號到：
- **永豐金證券 (Shioaji)** — 台灣 MXF 小台指期貨
- **Lucid Trading Prop Firm (Rithmic)** — CME MES 微型 E-mini S&P 500

## 功能特色

- 即時倉位追蹤，下單前自動判斷持倉狀態
- 支援開倉、平倉、反轉（先平再開）、全平
- 同向重複訊號自動忽略
- 兩個 broker 獨立運行，互不影響
- Shioaji 每 23 小時自動重新登入（處理 token 過期）
- 完整錯誤隔離，單一 broker 異常不影響另一個

## 環境需求

- Python 3.11+
- 永豐金證券 API key + CA 憑證
- Rithmic 帳號 (Lucid Prop Firm)

## 安裝

```bash
cd trading-bot
pip install -r requirements.txt
cp .env.example .env
# 編輯 .env 填入你的帳號資訊
```

## 環境變數說明

| 變數 | 說明 |
|------|------|
| `SHIOAJI_API_KEY` | 永豐 API Key |
| `SHIOAJI_SECRET_KEY` | 永豐 Secret Key |
| `SHIOAJI_CA_PATH` | CA 憑證檔案路徑 |
| `SHIOAJI_CA_PASSWORD` | CA 憑證密碼 |
| `RITHMIC_USER` | Rithmic 帳號 |
| `RITHMIC_PASSWORD` | Rithmic 密碼 |
| `RITHMIC_SYSTEM_NAME` | Rithmic 系統名稱（如 `Rithmic Test`） |
| `RITHMIC_APP_NAME` | 應用程式名稱 |
| `RITHMIC_URL` | Rithmic 伺服器 URL |
| `WEBHOOK_SECRET` | Webhook 驗證金鑰 |
| `LOG_LEVEL` | 日誌等級（DEBUG / INFO / WARNING / ERROR） |

## 啟動

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

伺服器啟動後：
- Webhook endpoint: `POST http://your-server:8000/webhook`
- Health check: `GET http://your-server:8000/health`

## TradingView Alert 設定

1. 在 TradingView 建立 Alert
2. 選擇 **Webhook URL**: `https://your-server:8000/webhook`
3. 在 Alert 的 **Settings** 中加入 Header：
   - `X-Webhook-Secret`: 你在 `.env` 中設定的 `WEBHOOK_SECRET`
4. Alert Message 格式（JSON）：

```json
{
  "action": "buy",
  "sentiment": "long",
  "quantity": 1,
  "ticker": "MXF",
  "price": "{{close}}",
  "time": "{{timenow}}"
}
```

### Action 對應邏輯

| action | 目前持倉 | 執行動作 |
|--------|---------|---------|
| `buy` | flat | 開多 |
| `buy` | short | 平空 + 開多 |
| `buy` | long | 忽略（已持多） |
| `sell` | flat | 開空 |
| `sell` | long | 平多 + 開空 |
| `sell` | short | 忽略（已持空） |
| `exit` | long/short | 全部平倉 |
| `exit` | flat | 忽略 |

## 日誌

- Console 即時輸出
- 檔案日誌：`trading.log`（自動 rotate 10MB，保留 30 天）
