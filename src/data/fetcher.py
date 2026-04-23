"""
Shioaji 資料擷取模組

從永豐 API 拉取 MX 歷史 K 棒資料（60分 + 15分），
並快取到本地 data/ 目錄避免重複下載。
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _resolve_env_vars(obj):
    """遞迴替換 ${ENV_VAR} 為環境變數的值。"""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def load_config(config_path: Optional[str] = None) -> dict:
    """從 settings.yaml 讀取設定，自動從 .env 載入環境變數。"""
    project_root = Path(__file__).resolve().parent.parent.parent
    if config_path is None:
        config_path = str(project_root / "config" / "settings.yaml")

    # 載入 .env
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return _resolve_env_vars(config)


def get_cache_path(symbol: str, freq: str) -> Path:
    """取得快取檔案路徑。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{symbol}_{freq}.parquet"


def fetch_kbars_shioaji(
    api_key: str,
    secret_key: str,
    symbol: str = "MXF",
    freq: str = "60T",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    simulation: bool = True,
    request_timeout_ms: int = 90_000,
) -> pd.DataFrame:
    """透過 Shioaji API 拉取歷史 K 棒資料。

    Parameters
    ----------
    freq : str
        '60T' = 60 分鐘, '15T' = 15 分鐘
    start_date, end_date : str
        格式 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        columns: datetime, open, high, low, close, volume
    """
    try:
        import shioaji as sj
    except ImportError:
        raise ImportError(
            "shioaji 未安裝。請執行: pip install shioaji"
        )

    api = sj.Shioaji(simulation=simulation)
    api.login(api_key=api_key, secret_key=secret_key)

    try:
        # 取得合約（StreamMultiContract → 選取近月合約）
        contracts = api.Contracts.Futures[symbol]
        contract_list = list(contracts)
        # 優先找近月合約（code 以 R1 結尾），否則取第一個
        contract = next(
            (c for c in contract_list if c.code.endswith("R1")),
            contract_list[0],
        )
        logger.info(f"Using contract: {contract.code} ({contract.symbol} {contract.name})")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        # Shioaji kbars 一次最多拉取有限天數，需要分批
        all_bars = []
        current_start = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # 根據頻率決定每批天數
        batch_days = 30 if freq == "15T" else 60

        while current_start < end_dt:
            batch_end = min(current_start + timedelta(days=batch_days), end_dt)
            logger.info(
                f"Fetching {symbol} {freq} from {current_start.date()} to {batch_end.date()}"
            )

            kbars = api.kbars(
                contract=contract,
                start=current_start.strftime("%Y-%m-%d"),
                end=batch_end.strftime("%Y-%m-%d"),
                timeout=request_timeout_ms,
            )

            df_batch = pd.DataFrame({**kbars})
            if not df_batch.empty:
                all_bars.append(df_batch)

            current_start = batch_end + timedelta(days=1)

        if not all_bars:
            logger.warning(f"No data fetched for {symbol} {freq}")
            return pd.DataFrame()

        df = pd.concat(all_bars, ignore_index=True)
        df.rename(columns={"ts": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])

        df = df.set_index("datetime")
        if "Open" in df.columns:
            df.columns = [c.lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]]
        df = df.sort_index().drop_duplicates()

        # Shioaji 回傳 1 分 K 棒，重採樣到目標頻率
        if freq and freq != "1T":
            resample_rule = freq.replace("T", "min") if "T" in freq else freq
            ohlc_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            df = df.resample(resample_rule, label="right", closed="right").agg(ohlc_dict)
            df.dropna(inplace=True)
            logger.info(f"Resampled to {freq}: {len(df)} bars")

        return df.reset_index()

    finally:
        try:
            api.logout()
        except TimeoutError as exc:
            logger.warning(f"Shioaji logout timed out: {exc}")


def fetch_and_cache(
    config: dict,
    freq: str = "60T",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """拉取資料並快取到本地。

    如果本地有快取且 force_refresh=False，直接從本地讀取。
    """
    symbol = config.get("instrument", {}).get("symbol", "MXF")
    cache_path = get_cache_path(symbol, freq.replace("T", "min"))

    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    shioaji_cfg = config.get("shioaji", {})
    df = fetch_kbars_shioaji(
        api_key=shioaji_cfg["api_key"],
        secret_key=shioaji_cfg["secret_key"],
        symbol=symbol,
        freq=freq,
        simulation=shioaji_cfg.get("simulation", True),
        request_timeout_ms=int(shioaji_cfg.get("historical_timeout_ms", 90_000)),
    )

    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Data cached to {cache_path}")

    return df


def load_cached_data(symbol: str = "MXF", freq: str = "60min") -> Optional[pd.DataFrame]:
    """直接從快取讀取資料（不呼叫 API）。"""
    cache_path = get_cache_path(symbol, freq)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


def load_csv_data(csv_path: str, resample_freq: Optional[str] = None) -> pd.DataFrame:
    """從 CSV 載入 K 棒資料（相容 Back_Trader 格式）。

    支援 columns: ts/datetime, Open/open, High/high, Low/low, Close/close, Volume/volume
    可選擇性重新取樣（例如 '60T' 或 '60min'）。
    """
    df = pd.read_csv(csv_path)

    # 標準化時間欄位名稱
    dt_col = None
    for col in ["ts", "datetime", "date", "Date", "Datetime"]:
        if col in df.columns:
            dt_col = col
            break

    if dt_col is None and df.columns[0] not in ["open", "Open"]:
        dt_col = df.columns[0]

    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.rename(columns={dt_col: "datetime"})

    # 標準化 OHLCV 欄位名稱
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume") and col != lower:
            rename_map[col] = lower
    if rename_map:
        df = df.rename(columns=rename_map)

    # 確保有必要欄位
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV 缺少必要欄位。現有: {list(df.columns)}, 需要: {required}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    # 重新取樣
    if resample_freq and "datetime" in df.columns:
        df = df.set_index("datetime")
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df = df.resample(resample_freq, label="right", closed="right").agg(ohlc_dict)
        df.dropna(inplace=True)
        df = df.reset_index()

    # 排序 + 去重
    if "datetime" in df.columns:
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    cols = ["datetime", "open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]]


def generate_sample_data(
    n_bars: int = 5000,
    freq: str = "60T",
    start_price: float = 16000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """產生模擬 K 棒資料，用於測試和開發。"""
    import numpy as np
    rng = np.random.RandomState(seed)

    dates = pd.date_range(
        start="2021-01-04 08:45",
        periods=n_bars,
        freq="60min" if freq == "60T" else "15min",
    )

    prices = [start_price]
    for _ in range(n_bars - 1):
        change = rng.normal(0, 30)
        prices.append(prices[-1] + change)
    prices = np.array(prices)

    opens = prices
    # 產生隨機 high/low/close
    bar_range = np.abs(rng.normal(40, 20, n_bars))
    direction = rng.choice([-1, 1], n_bars)

    highs = opens + np.abs(rng.normal(20, 10, n_bars))
    lows = opens - np.abs(rng.normal(20, 10, n_bars))
    closes = opens + direction * rng.uniform(0, bar_range)

    # 確保 high >= max(open, close), low <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    volumes = rng.randint(100, 5000, n_bars).astype(float)

    df = pd.DataFrame({
        "datetime": dates[:n_bars],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    return df
