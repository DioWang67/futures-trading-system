# Trading System Roadmap

This document tracks the main gaps between the current repository and a more production-ready TMF (Micro TAIEX Futures) trading system.

## Current Baseline

The project already has a usable simulation foundation:

- `TMF` is the default Shioaji futures symbol.
- Local protective exit logic exists with watchdog and `close-only` mode.
- Dashboard, SQLite event storage, and Telegram notification hooks are available.
- Fast simulation roundtrip testing is available through `run_sim_roundtrip_test.py`.
- Order/fill handling now supports Shioaji `FuturesDeal` callbacks.

That said, the system is still closer to a strong simulation / paper-trading stack than a full live-trading production stack.

## Highest Priority Gaps

### 1. Trading Calendar And Session Rules

Why it matters:

- TMF has day and night sessions.
- Cross-date handling is important for risk resets and reporting.
- Expiry day and holiday rules can break otherwise valid logic.

What to add:

- `calendar.py` or equivalent module.
- `is_trading_day(dt)`
- `get_session(dt)` returning `DAY`, `NIGHT`, or `CLOSED`
- holiday / make-up workday support
- expiry-day session overrides
- session-aware daily reset logic

### 2. Contract Rollover Management

Why it matters:

- Near-expiry behavior can distort both live execution and analytics.
- Long-running automation needs explicit roll rules.

What to add:

- active-contract resolver for TMF
- configurable rollover window
- stop-new-entry rules near expiry
- forced close / forced roll behavior
- contract-month tracking in logs, fills, and reports

### 3. More Realistic PnL And Costs

Why it matters:

- Current realized PnL is based on actual simulated fill prices, but it is still local math.
- It does not yet include the full trading-cost picture.

What to add:

- `gross_pnl`
- `fees`
- `taxes`
- `net_pnl`
- configurable TMF cost model
- dashboard display for gross vs net performance

### 4. Broker Reconciliation

Why it matters:

- Local state is useful, but broker state must remain the source of truth.
- Reconnects, duplicate events, and delayed callbacks can cause drift.

What to add:

- scheduled reconciliation job
- open-order sync after startup and reconnect
- broker position sync beyond initial boot
- account balance / margin snapshots
- mismatch detection that forces `close-only`
- manual intervention mode when local and broker state diverge

### 5. Native Or Broker-Hosted Protection Research

Why it matters:

- The current stop-loss / take-profit flow is client-side triggered.
- If the process, machine, or network fails, protection can fail too.

What to add:

- confirm whether Shioaji exposes native futures stop / conditional orders for this use case
- if supported, build a broker-native protective mode
- if not supported, keep the current client-side mode but harden it further with a separate watchdog process
- compare `MKT`, `MKP`, and protective-limit behavior in simulation

## Operational Hardening

### 6. Slippage Measurement And Execution Analytics

What to add:

- trigger-to-submit latency
- submit-to-fill latency
- slippage by session and market regime
- separate reports for stop-loss vs take-profit exits
- execution-quality dashboard panel

### 7. Market Data Health Controls

What to add:

- stronger stale-quote detection
- quote gap alarms
- reconnect backoff policy
- explicit startup warm-up rules before allowing entries
- optional bid/ask-based trigger mode instead of last-trade-only logic

### 8. Risk Controls For Futures-Specific Behavior

What to add:

- no-new-position window before session close
- volatility circuit for disabling new entries
- max adverse excursion guardrails
- margin-aware position checks
- max same-direction add-on count
- stricter overnight / night-session rules

## Testing Roadmap

### 9. Simulation Test Expansion

What to add:

- repeatable long / short roundtrip scenarios
- partial-fill simulations where possible
- reconnect and restart recovery tests
- stale-quote and watchdog failover tests
- cost-model regression tests

### 10. Pre-Live Validation Checklist

Before using real money, the system should have:

- simulation login and quote subscription verified
- entry, exit, cancel, and callback lifecycle verified
- protective events recorded end-to-end
- Telegram alerts verified on a real device
- reconciliation flow tested after restart
- close-only behavior tested during fault injection
- live-trading config isolated from simulation config

## Nice-To-Have Improvements

- richer dashboard filters and trade drill-down views
- daily report export for fills, PnL, slippage, and risk events
- strategy parameter versioning in stored events
- per-trade signal snapshot storage
- deployment scripts for service supervision and restart health checks

## Suggested Build Order

1. Trading calendar and session rules
2. Contract rollover handling
3. Gross/net PnL with fees and taxes
4. Broker reconciliation and mismatch handling
5. Native-protection research or watchdog split-process hardening
6. Slippage analytics and pre-live validation checklist

