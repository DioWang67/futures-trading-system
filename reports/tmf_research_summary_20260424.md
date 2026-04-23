# TMF Research Summary 2026-04-24

## Data

- Source: Shioaji TMF history fetched in batches
- Cached files:
  - `data/TMF_15min.parquet`
  - `data/TMF_60min.parquet`
- Available true-data range:
  - `2024-07-29 09:00:00`
  - `2026-04-23 23:45:00`

## Baseline TMF Backtest

- Capital: `100,000`
- Contracts: `2`
- Point value: `10`
- Params:
  - `swing_lookback=10`
  - `bos_min_move=35.0`
  - `ob_max_age=19`
  - `ob_body_ratio=0.55`
  - `fvg_min_gap=13.0`
  - `fvg_enabled=true`
  - `pin_bar_ratio=0.75`
  - `engulf_ratio=1.4`
  - `rr_ratio=1.25`
  - `use_structure_tp=true`
  - `sl_buffer=3.0`
  - `pa_confirm=false`
  - `adx_period=16`
  - `adx_threshold=32.5`
  - `adx_filter_enabled=false`
  - `atr_filter_enabled=true`
  - `atr_period=14`
  - `atr_min_points=20.0`
  - `blocked_entry_hours=[0,1,10,15,17,21]`
- Result:
  - trades: `199`
  - win rate: `61.81%`
  - profit factor: `2.37`
  - max drawdown: `10.48%`
  - total pnl: `188,808`
- Artifacts:
  - `reports/backtest_20260423_234113.html`
  - `reports/tmf_api_history_backtest_20260423.json`

## Walk-Forward Findings

### Lite

- Setup: `3` splits, `60` trials
- Result: `1/3` passed
- Artifacts:
  - `reports/walkforward_20260424_001536.html`
  - `reports/tmf_api_walkforward_lite_20260424.json`

### Constrained

- Setup:
  - fixed base strategy from baseline
  - optimize only:
    - `bos_min_move`
    - `fvg_min_gap`
    - `rr_ratio`
    - `sl_buffer`
    - `atr_min_points`
- Result: `1/3` passed
- Notes:
  - segment 2 had `0` trades
  - segment 3 had strong pnl but drawdown exceeded threshold
- Artifacts:
  - `reports/walkforward_20260424_002755.html`
  - `reports/tmf_api_walkforward_constrained_20260424.json`

### Relaxed

- Setup:
  - blocked hours reduced to `[0,1,21]`
  - `atr_min_points` fixed at `15.0`
  - optimize only:
    - `bos_min_move`
    - `fvg_min_gap`
    - `rr_ratio`
    - `sl_buffer`
- Result: `0/3` passed
- Artifacts:
  - `reports/walkforward_20260424_003349.html`
  - `reports/tmf_api_walkforward_relaxed_20260424.json`

## Conclusion

- Single-period TMF backtest looks strong.
- Walk-forward validation is not stable enough yet.
- Current strategy likely has edge, but performance is regime-sensitive.
- Next work should focus on improving walk-forward pass rate rather than maximizing in-sample pnl.
