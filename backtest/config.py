"""
Strategy parameters — mirrors Pine Script v4.9 inputs.
Modify these to match your TradingView settings for comparison.
"""

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    # ── 策略風控 ──
    trade_mode: str = "多空皆可"  # "多空皆可" | "只做多" | "只做空"
    exit_mode: str = "移動止盈 (SL+Trailing)"
    # Options: "固定止盈 (SL+TP)" | "移動止盈 (SL+Trailing)" | "分批出場 (1:1 + 2.5R)"

    sl_atr: float = 2.0           # 止損 ATR 倍數
    tp_ratio: float = 5.0         # 止盈 風險回報比
    tp1_ratio: float = 1.0        # 分批出場第一口 RR
    atr_period_sl: int = 14       # ATR 週期

    trail_mult: float = 1.8       # Trailing 啟動倍數
    trail_offset_mult: float = 0.05  # Trailing 偏移倍數

    max_sl_points: float = 100.0  # 最大允許止損點數 (0=關閉)

    use_time_filter: bool = True
    trading_session_start: str = "09:30"  # 紐約時間
    trading_session_end: str = "15:45"

    # ── SMC 邏輯 ──
    pivot_length: int = 5
    min_signal_distance: int = 2
    range_period: int = 30
    range_threshold: float = 0.38
    near_structure_atr: float = 2.0
    near_ema_atr: float = 1.5
    struct_lookback: int = 20

    # ── 過濾器 ──
    use_trend_filter: bool = True
    higher_tf: str = "1H"  # "15M" | "30M" | "1H" | "4H"
    restrict_repeated_signals: bool = True
    use_dual_htf: bool = False

    # ── 形態設定 ──
    pin_wick_ratio: float = 1.2
    use_engulfing: bool = True
    engulf_body_ratio: float = 1.3

    # ── 進場微調 ──
    volume_long_period: int = 20
    vol_multiplier: float = 1.1
    min_candle_atr: float = 0.5
    min_body_atr: float = 0.15
    env_atr_ratio: float = 0.7
    momentum_block_atr: float = 1.5

    # ── RSI (disabled per user request) ──
    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_long_max: int = 63
    rsi_short_min: int = 37

    # ── 結構保護 (disabled per user request) ──
    use_structure_filter: bool = False
    struct_swing_len: int = 3
    avoid_exit_reentry: bool = True
    reentry_cooldown: int = 5

    # ── 回測設定 ──
    initial_capital: float = 25000.0
    default_qty: int = 2
    commission_per_contract: float = 2.0
    slippage_points: float = 5.0

    # ── 合約規格 ──
    tick_size: float = 0.25       # NQ tick size
    point_value: float = 20.0     # NQ $20 per point (MNQ=$2)
    # 若用 MNQ: tick_size=0.25, point_value=2.0

    score_threshold: int = 4
