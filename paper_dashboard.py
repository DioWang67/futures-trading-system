from __future__ import annotations

import hmac
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from config import settings
from trade_store import TradeStore


ROOT = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = ROOT / ".tmp" / "local_paper_state.json"
DEFAULT_DB_PATH = ROOT / "data" / "trades.db"

app = FastAPI(title="Paper Trading Dashboard", version="1.2.0")

# Reuse a single TradeStore per database path; instantiating one per
# request leaks a thread-local sqlite connection each time the
# browser polls (every 5s in the default dashboard).
_trade_store_lock = threading.Lock()
_trade_stores: dict[str, TradeStore] = {}


def _get_trade_store(db_path: Path) -> TradeStore:
    key = str(db_path)
    with _trade_store_lock:
        store = _trade_stores.get(key)
        if store is None:
            store = TradeStore(db_path=db_path)
            _trade_stores[key] = store
        return store


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_admin_secret(x_admin_secret: str = Header(default="")) -> None:
    expected = settings.admin.secret.get_secret_value()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Admin secret not configured on server",
        )
    if x_admin_secret and hmac.compare_digest(x_admin_secret, expected):
        return

    raise HTTPException(status_code=403, detail="Invalid admin secret")


def _require_dashboard_auth(
    request: Request,
    x_admin_secret: str = Header(default=""),
    x_csrf_token: str = Header(default="", alias="X-CSRF-Token"),
) -> None:
    expected = settings.admin.secret.get_secret_value()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Admin secret not configured on server",
        )

    if x_admin_secret and hmac.compare_digest(x_admin_secret, expected):
        return

    cookie_secret = request.cookies.get("paper_dashboard_admin_secret") or ""
    if not cookie_secret or not hmac.compare_digest(cookie_secret, expected):
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    csrf_cookie = request.cookies.get("paper_dashboard_csrf") or ""
    if not csrf_cookie or not x_csrf_token or not hmac.compare_digest(csrf_cookie, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


def _load_runner_state(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_STATE_PATH
    if not path.exists():
        return {
            "last_processed_bar_time": "",
            "managed_trade": None,
            "metadata": {},
            "runtime": {"status": "not_started"},
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "last_processed_bar_time": "",
            "managed_trade": None,
            "metadata": {},
            "runtime": {
                "status": "state_read_error",
                "last_cycle_result": {"status": "error", "reason": str(exc)},
            },
        }


def _iso_age_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (_utc_now() - dt).total_seconds())


def _safe_trade_store_payload(db_path: Path) -> dict[str, Any]:
    try:
        store = _get_trade_store(db_path)
        position = store.get_latest_position("shioaji") or {
            "side": "flat",
            "quantity": 0,
            "entry_price": 0.0,
            "snapshot_at": "",
        }
        return {
            "position": position,
            "summary": store.get_daily_summary(),
            "summary_7d": store.get_period_summary(days=7),
            "recent_fills": store.get_recent_fills(20),
            "recent_signals": store.get_recent_signal_events(30),
            "recent_protective_events": store.get_recent_protective_events(20),
            "recent_risk_events": store.get_recent_risk_events(10),
            "storage_error": None,
        }
    except Exception as exc:
        return {
            "position": {
                "side": "unknown",
                "quantity": 0,
                "entry_price": 0.0,
                "snapshot_at": "",
            },
            "summary": {"date": "", "brokers": [], "total_pnl": 0, "total_trades": 0},
            "summary_7d": {"days": 7, "from": "", "brokers": [], "total_pnl": 0, "total_trades": 0},
            "recent_fills": [],
            "recent_signals": [],
            "recent_protective_events": [],
            "recent_risk_events": [],
            "storage_error": str(exc),
        }


def _derive_runner_health(
    status: str,
    last_cycle_age_seconds: float | None,
    next_cycle_in_seconds: float | None,
) -> str:
    if status in {"state_read_error", "error"}:
        return "error"
    if status == "not_started":
        return "offline"
    if last_cycle_age_seconds is not None and last_cycle_age_seconds > 7200:
        return "stale"
    if next_cycle_in_seconds is not None and next_cycle_in_seconds > 7200:
        return "stale"
    return "ok"


def _build_dashboard_payload() -> dict[str, Any]:
    state = _load_runner_state()
    metadata = dict(state.get("metadata") or {})
    runtime = dict(state.get("runtime") or {})
    protective = dict(runtime.get("protective_exit") or {})
    watchdog = dict(runtime.get("watchdog") or {})
    quote = dict(runtime.get("quote") or {})

    last_cycle_age = _iso_age_seconds(runtime.get("last_cycle_at", ""))
    next_cycle_in = None
    if runtime.get("next_cycle_eta"):
        next_cycle_in = _iso_age_seconds(runtime["next_cycle_eta"])
        if next_cycle_in is not None:
            next_cycle_in = max(0.0, next_cycle_in)

    storage = _safe_trade_store_payload(DEFAULT_DB_PATH)
    runner_status = str(runtime.get("status") or ("not_started" if not runtime else "unknown"))

    return {
        "generated_at": _utc_now().isoformat(),
        "runner": {
            "status": runner_status,
            "health": _derive_runner_health(
                runner_status, last_cycle_age, next_cycle_in
            ),
            "last_processed_bar_time": state.get("last_processed_bar_time", ""),
            "managed_trade": state.get("managed_trade"),
            "metadata": metadata,
            "runtime": runtime,
            "protective_exit": protective,
            "watchdog": watchdog,
            "quote": quote,
            "last_cycle_age_seconds": last_cycle_age,
            "next_cycle_in_seconds": next_cycle_in,
        },
        "position": storage["position"],
        "summary": storage["summary"],
        "summary_7d": storage["summary_7d"],
        "recent_fills": storage["recent_fills"],
        "recent_signals": storage["recent_signals"],
        "recent_protective_events": storage["recent_protective_events"],
        "recent_risk_events": storage["recent_risk_events"],
        "storage_error": storage["storage_error"],
    }


@app.get("/api/dashboard")
async def api_dashboard(_: None = Depends(_require_dashboard_auth)) -> dict[str, Any]:
    return _build_dashboard_payload()


@app.get("/api/analytics/summary")
async def api_analytics_summary(
    _: None = Depends(_require_dashboard_auth),
    days: int = Query(default=7, ge=1, le=365),
    reconcile_hours: int = Query(default=24, ge=1, le=168),
) -> dict[str, Any]:
    store = _get_trade_store(DEFAULT_DB_PATH)
    return {
        "generated_at": _utc_now().isoformat(),
        "summary": store.get_period_summary(days=days),
        "daily": store.get_daily_summary(),
        "reconciliation": store.get_signal_fill_reconciliation(hours=reconcile_hours),
    }


@app.get("/api/analytics/signals")
async def api_analytics_signals(
    _: None = Depends(_require_dashboard_auth),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    store = _get_trade_store(DEFAULT_DB_PATH)
    return {
        "generated_at": _utc_now().isoformat(),
        "signals": store.get_recent_signal_events(limit=limit),
        "fills": store.get_recent_fills(limit=min(limit, 500)),
    }


def _is_dashboard_cookie_authenticated(request: Request) -> bool:
    expected = settings.admin.secret.get_secret_value()
    cookie_secret = request.cookies.get("paper_dashboard_admin_secret") or ""
    return bool(expected and cookie_secret and hmac.compare_digest(cookie_secret, expected))


def _build_login_page() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TMF Dashboard Login</title>
  <style>
    body { font-family: "Segoe UI", "PingFang TC", sans-serif; margin: 0; background: #f4efe5; color: #183126; }
    .wrap { max-width: 420px; margin: 10vh auto; background: rgba(255,255,255,0.92); border: 1px solid rgba(24,49,38,0.12); border-radius: 18px; padding: 24px; }
    h1 { margin-top: 0; font-size: 24px; }
    label { display:block; margin-bottom: 8px; color: #68766f; font-size: 14px; }
    input { width: 100%; padding: 12px; border-radius: 10px; border: 1px solid rgba(24,49,38,0.2); margin-bottom: 12px; }
    button { width: 100%; padding: 12px; border: 0; border-radius: 10px; background: #146d4a; color: #fff; font-weight: 700; cursor: pointer; }
    p { color: #68766f; font-size: 13px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>TMF Dashboard</h1>
    <p>請輸入管理密鑰登入（不使用 URL query 參數）。</p>
    <form method="post" action="/login">
      <label for="admin_secret">Admin Secret</label>
      <input id="admin_secret" name="admin_secret" type="password" autocomplete="current-password" required>
      <button type="submit">登入</button>
    </form>
  </div>
</body>
</html>"""


@app.post("/login")
async def dashboard_login(admin_secret: str = Form(default="")) -> RedirectResponse:
    expected = settings.admin.secret.get_secret_value()
    submitted_secret = (admin_secret or "").strip()
    if not expected or not submitted_secret or not hmac.compare_digest(submitted_secret, expected):
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    csrf_token = secrets.token_urlsafe(24)
    redirect = RedirectResponse(url="/", status_code=303)
    redirect.set_cookie(
        "paper_dashboard_admin_secret",
        submitted_secret,
        httponly=True,
        samesite="strict",
        max_age=8 * 60 * 60,
    )
    redirect.set_cookie(
        "paper_dashboard_csrf",
        csrf_token,
        httponly=False,
        samesite="strict",
        max_age=8 * 60 * 60,
    )
    return redirect


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    if not _is_dashboard_cookie_authenticated(request):
        return HTMLResponse(content=_build_login_page(), status_code=200)

    csrf_token = request.cookies.get("paper_dashboard_csrf") or ""
    if (request.cookies.get("paper_dashboard_admin_secret") or "") and not csrf_token:
        csrf_token = secrets.token_urlsafe(24)

    html = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TMF Trading Monitor</title>
  <style>
    :root {
      --bg: #f4efe5;
      --card: rgba(255,255,255,0.88);
      --ink: #183126;
      --muted: #68766f;
      --line: rgba(24,49,38,0.12);
      --good: #146d4a;
      --warn: #ad6c1f;
      --bad: #b04236;
      --shadow: 0 18px 40px rgba(24,49,38,0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "PingFang TC", "Noto Sans TC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(20,109,74,0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(173,108,31,0.12), transparent 22%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
    }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 28px 18px 40px; }
    .hero { display:flex; justify-content:space-between; align-items:flex-end; gap:20px; margin-bottom:20px; }
    h1 { margin: 0; font-size: 34px; letter-spacing: -0.03em; }
    .sub { margin-top: 8px; color: var(--muted); }
    .stamp { color: var(--muted); font-size: 14px; }
    .grid { display:grid; grid-template-columns:repeat(12, 1fr); gap:16px; }
    .card {
      grid-column: span 4;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }
    .wide { grid-column: span 6; }
    .full { grid-column: span 12; }
    .label { font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
    .big { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .pill { display:inline-flex; padding:8px 12px; border-radius:999px; font-size:14px; font-weight:700; }
    .ok { background: rgba(20,109,74,0.12); color: var(--good); }
    .warn { background: rgba(173,108,31,0.14); color: var(--warn); }
    .bad { background: rgba(176,66,54,0.14); color: var(--bad); }
    .kv { display:grid; grid-template-columns:140px 1fr; gap:8px 12px; margin-top:14px; font-size:14px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-word; }
    .notice { margin-top: 12px; padding: 12px 14px; border-radius: 14px; background: rgba(176,66,54,0.08); color: var(--bad); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    @media (max-width: 980px) {
      .hero { display:block; }
      .card, .wide { grid-column: span 12; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>TMF Trading Monitor</h1>
        <div class="sub">觀察 runner、部位、保護單、quote/watchdog 與最近滑價事件。</div>
      </div>
      <div class="stamp" id="refreshedAt">最後刷新 --</div>
    </div>

    <div class="grid">
      <section class="card">
        <div class="label">Runner</div>
        <div id="runnerStatus" class="pill warn">loading</div>
        <div class="kv">
          <div>Symbol</div><div id="symbol">-</div>
          <div>Timeframe</div><div id="freqs">-</div>
          <div>Quantity</div><div id="quantity">-</div>
          <div>Last Cycle</div><div id="lastCycle">-</div>
          <div>Next Cycle</div><div id="nextCycle">-</div>
          <div>Last Bar</div><div id="lastBar" class="mono">-</div>
          <div>Health</div><div id="runnerHealth">-</div>
        </div>
        <div id="storageNotice"></div>
      </section>

      <section class="card">
        <div class="label">Position</div>
        <div class="big" id="positionSummary">flat x0</div>
        <div class="kv">
          <div>Entry Price</div><div id="entryPrice">-</div>
          <div>Snapshot At</div><div id="snapshotAt">-</div>
          <div>Managed Trade</div><div id="managedTradeSummary">-</div>
          <div>Today PnL</div><div id="totalPnl">-</div>
          <div>Today Trades</div><div id="totalTrades">-</div>
        </div>
      </section>

      <section class="card">
        <div class="label">Protection</div>
        <div id="protectiveStatus" class="pill warn">idle</div>
        <div class="kv">
          <div>Close Only</div><div id="closeOnly">-</div>
          <div>Watchdog</div><div id="watchdogReason">-</div>
          <div>Side / Qty</div><div id="protectiveSideQty">-</div>
          <div>SL / TP</div><div id="protectiveLevels">-</div>
          <div>Trigger</div><div id="protectiveTrigger">-</div>
          <div>Submit</div><div id="protectiveSubmit">-</div>
        </div>
      </section>

      <section class="card wide">
        <div class="label">Quote And Runner State</div>
        <div class="kv">
          <div>Quote Age</div><div id="quoteAge">-</div>
          <div>Last Tick</div><div id="quotePrice">-</div>
          <div>Quote Stale</div><div id="quoteStale">-</div>
          <div>Last Result</div><div id="lastResult" class="mono">-</div>
          <div>Broker Breakdown</div><div id="brokerSummary">-</div>
        </div>
      </section>

      <section class="card wide">
        <div class="label">Managed Trade</div>
        <div id="managedTradeDetails" class="kv">
          <div>Status</div><div>none</div>
        </div>
      </section>

      <section class="card full">
        <div class="label">Recent Protective Events</div>
        <table>
          <thead>
            <tr>
              <th>Triggered</th>
              <th>Status</th>
              <th>Reason</th>
              <th>Side</th>
              <th>Trigger</th>
              <th>Submit</th>
              <th>Fill</th>
              <th>Slip</th>
            </tr>
          </thead>
          <tbody id="protectiveTable">
            <tr><td colspan="8">no data</td></tr>
          </tbody>
        </table>
      </section>

      <section class="card full">
        <div class="label">Recent Fills</div>
        <table>
          <thead>
            <tr>
              <th>Filled At</th>
              <th>Broker</th>
              <th>Action</th>
              <th>Qty</th>
              <th>Price</th>
              <th>PnL</th>
            </tr>
          </thead>
          <tbody id="fillsTable">
            <tr><td colspan="6">no data</td></tr>
          </tbody>
        </table>
      </section>

      <section class="card full">
        <div class="label">Recent Risk Events</div>
        <table>
          <thead>
            <tr>
              <th>Created</th>
              <th>Type</th>
              <th>Broker</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody id="riskTable">
            <tr><td colspan="4">no data</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>

  <script>
    const csrfToken = "__CSRF_TOKEN__";
    function fmtMoney(value, digits = 0) {
      if (value == null || value === "") return "-";
      return new Intl.NumberFormat("zh-TW", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
    }
    function fmtTs(value) {
      if (!value) return "-";
      return value.replace("T", " ").replace("+00:00", " UTC");
    }
    function fmtAge(seconds) {
      if (seconds == null) return "-";
      if (seconds < 60) return `${seconds.toFixed(0)} 秒`;
      if (seconds < 3600) return `${(seconds / 60).toFixed(1)} 分`;
      return `${(seconds / 3600).toFixed(1)} 小時`;
    }
    function pillClass(value) {
      if (["ok", "idle", "running", "sleeping", "submitted", "filled", false, "false"].includes(value)) return "ok";
      if (["warn", "stale", "offline", "not_started", "armed", "triggered"].includes(value)) return "warn";
      return "bad";
    }
    function setPill(id, value) {
      const el = document.getElementById(id);
      el.className = `pill ${pillClass(value)}`;
      el.textContent = String(value ?? "-");
    }
    function clearChildren(el) {
      while (el.firstChild) el.removeChild(el.firstChild);
    }
    function setNotice(message) {
      const holder = document.getElementById("storageNotice");
      clearChildren(holder);
      if (!message) return;
      const notice = document.createElement("div");
      notice.className = "notice";
      notice.textContent = message;
      holder.appendChild(notice);
    }
    function renderBrokerSummary(rows) {
      const el = document.getElementById("brokerSummary");
      clearChildren(el);
      if (!rows.length) {
        el.textContent = "-";
        return;
      }
      rows.forEach((row, idx) => {
        const line = document.createElement("div");
        line.textContent = `${row.broker}: ${row.trade_count} 筆, PnL ${fmtMoney(row.total_pnl || 0)}`;
        el.appendChild(line);
        if (idx < rows.length - 1) {
          el.appendChild(document.createElement("br"));
        }
      });
    }
    function renderManagedTrade(managed) {
      const managedEl = document.getElementById("managedTradeDetails");
      clearChildren(managedEl);
      const pairs = managed ? [
        ["Status", managed.status],
        ["Side", managed.side],
        ["Qty", managed.quantity],
        ["SL / TP", `${managed.sl} / ${managed.tp}`],
        ["Entry Bar", managed.entry_bar_time],
        ["Submitted", fmtTs(managed.submitted_at)],
        ["Exit Reason", managed.exit_reason || "-"],
      ] : [["Status", "none"]];
      pairs.forEach(([label, value]) => {
        const left = document.createElement("div");
        left.textContent = String(label);
        const right = document.createElement("div");
        right.textContent = value == null ? "-" : String(value);
        if (label === "Entry Bar") right.className = "mono";
        managedEl.appendChild(left);
        managedEl.appendChild(right);
      });
    }
    function renderTable(tbodyId, rows, emptyCols) {
      const tbody = document.getElementById(tbodyId);
      clearChildren(tbody);
      if (!rows.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = emptyCols;
        td.textContent = "no data";
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }
      rows.forEach((cells) => {
        const tr = document.createElement("tr");
        cells.forEach((cell) => {
          const td = document.createElement("td");
          td.textContent = cell.text;
          if (cell.className) td.className = cell.className;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    }
    async function refresh() {
      const headers = csrfToken ? { "X-CSRF-Token": csrfToken } : {};
      const res = await fetch("/api/dashboard", { cache: "no-store", headers });
      if (!res.ok) {
        throw new Error(`dashboard fetch failed: ${res.status}`);
      }
      const data = await res.json();
      const runner = data.runner || {};
      const meta = runner.metadata || {};
      const runtime = runner.runtime || {};
      const managed = runner.managed_trade;
      const position = data.position || {};
      const summary = data.summary || {};
      const protective = runner.protective_exit || {};
      const watchdog = runner.watchdog || {};
      const quote = runner.quote || {};

      setPill("runnerStatus", runner.status || "unknown");
      setPill("protectiveStatus", protective.status || "idle");

      document.getElementById("symbol").textContent = meta.symbol || "-";
      document.getElementById("freqs").textContent = `${meta.ltf_freq || "-"} / ${meta.htf_freq || "-"}`;
      document.getElementById("quantity").textContent = meta.quantity ?? "-";
      document.getElementById("lastCycle").textContent = fmtAge(runner.last_cycle_age_seconds);
      document.getElementById("nextCycle").textContent = fmtAge(runner.next_cycle_in_seconds);
      document.getElementById("lastBar").textContent = runner.last_processed_bar_time || "-";
      document.getElementById("runnerHealth").textContent = runner.health || "-";

      document.getElementById("positionSummary").textContent = `${position.side || "flat"} x${position.quantity || 0}`;
      document.getElementById("entryPrice").textContent = position.entry_price ? fmtMoney(position.entry_price, 1) : "-";
      document.getElementById("snapshotAt").textContent = fmtTs(position.snapshot_at);
      document.getElementById("managedTradeSummary").textContent = managed ? `${managed.side} x${managed.quantity} (${managed.status})` : "none";
      document.getElementById("totalPnl").textContent = `${(summary.total_pnl || 0) >= 0 ? "+" : ""}${fmtMoney(summary.total_pnl || 0)}`;
      document.getElementById("totalTrades").textContent = summary.total_trades ?? 0;

      document.getElementById("closeOnly").textContent = watchdog.close_only ? `yes: ${watchdog.close_only_reason || ""}` : "no";
      document.getElementById("watchdogReason").textContent = watchdog.watchdog_reason || "-";
      document.getElementById("protectiveSideQty").textContent = protective.side ? `${protective.side} x${protective.quantity}` : "-";
      document.getElementById("protectiveLevels").textContent = (protective.stop_loss || protective.take_profit)
        ? `${protective.stop_loss || "-"} / ${protective.take_profit || "-"}`
        : "-";
      document.getElementById("protectiveTrigger").textContent = protective.trigger_price
        ? `${fmtMoney(protective.trigger_price, 1)} (${protective.trigger_reason || "-"})`
        : "-";
      document.getElementById("protectiveSubmit").textContent = protective.submit_price
        ? `${fmtMoney(protective.submit_price, 1)} (${protective.execution_price_type || "-"})`
        : "-";

      document.getElementById("quoteAge").textContent = fmtAge(quote.age_seconds);
      document.getElementById("quotePrice").textContent = quote.last_tick_price ? fmtMoney(quote.last_tick_price, 1) : "-";
      document.getElementById("quoteStale").textContent = quote.is_stale ? `yes (${quote.stale_after_seconds || "-"}s)` : "no";
      document.getElementById("lastResult").textContent = runtime.last_cycle_result ? JSON.stringify(runtime.last_cycle_result) : "-";
      renderBrokerSummary(summary.brokers || []);
      renderManagedTrade(managed);
      renderTable(
        "protectiveTable",
        (data.recent_protective_events || []).map((event) => ([
          { text: fmtTs(event.triggered_at) },
          { text: event.status || "-" },
          { text: event.trigger_reason || "-" },
          { text: `${event.side || "-"} x${event.quantity || 0}` },
          { text: event.trigger_price ? fmtMoney(event.trigger_price, 1) : "-" },
          { text: event.submit_price ? `${fmtMoney(event.submit_price, 1)} (${event.execution_price_type || "-"})` : "-" },
          { text: event.fill_price ? fmtMoney(event.fill_price, 1) : "-" },
          { text: event.slippage_points || event.slippage_points === 0 ? fmtMoney(event.slippage_points, 1) : "-" },
        ])),
        8,
      );
      renderTable(
        "fillsTable",
        (data.recent_fills || []).map((fill) => ([
          { text: fmtTs(fill.filled_at) },
          { text: fill.broker || "-" },
          { text: fill.action || "-" },
          { text: String(fill.filled_qty ?? "-") },
          { text: fmtMoney(fill.fill_price, 1) },
          { text: fmtMoney(fill.pnl || 0) },
        ])),
        6,
      );
      renderTable(
        "riskTable",
        (data.recent_risk_events || []).map((event) => ([
          { text: fmtTs(event.created_at) },
          { text: event.event_type || "-" },
          { text: event.broker || "-" },
          { text: event.details || "-", className: "mono" },
        ])),
        4,
      );

      setNotice(data.storage_error ? `資料讀取異常: ${data.storage_error}` : "");
      document.getElementById("refreshedAt").textContent = `最後刷新 ${new Date().toLocaleTimeString("zh-TW")}`;
    }
    refresh().catch(err => {
      setNotice(err.message);
    });
    setInterval(() => {
      refresh().catch(err => {
        setNotice(err.message);
      });
    }, 5000);
  </script>
</body>
</html>"""
    html = html.replace("__CSRF_TOKEN__", csrf_token)
    response = HTMLResponse(content=html)
    if (request.cookies.get("paper_dashboard_admin_secret") or "") and request.cookies.get("paper_dashboard_csrf") != csrf_token:
        response.set_cookie(
            "paper_dashboard_csrf",
            csrf_token,
            httponly=False,
            samesite="strict",
            max_age=8 * 60 * 60,
        )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "paper_dashboard:app",
        host="127.0.0.1",
        port=8010,
        reload=False,
        log_level="warning",
    )
