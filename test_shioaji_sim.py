from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent
SIM_RUNTIME_DIR = PROJECT_ROOT / ".tmp" / "shioaji_sim"


def prepare_shioaji_runtime() -> Path:
    """Isolate Shioaji's cache/log files for this test process.

    The upstream SDK stores contract cache under ``~/.shioaji`` and may hold a
    file lock there while refreshing contracts. Reusing the same location across
    runs can produce noisy lock timeouts even when the trade itself succeeds.
    """
    runtime_home = SIM_RUNTIME_DIR
    shioaji_home = runtime_home / ".shioaji"
    shioaji_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    os.environ["USERPROFILE"] = str(runtime_home)
    os.environ["SJ_LOG_PATH"] = str(runtime_home / "shioaji-sim.log")
    return shioaji_home


prepare_shioaji_runtime()

import shioaji as sj


def load_shioaji_credentials() -> tuple[str, str]:
    env_path = PROJECT_ROOT / ".env"
    values = dotenv_values(env_path) if env_path.exists() else {}
    api_key = str(values.get("SHIOAJI_API_KEY") or os.environ.get("SHIOAJI_API_KEY") or "")
    secret_key = str(values.get("SHIOAJI_SECRET_KEY") or os.environ.get("SHIOAJI_SECRET_KEY") or "")
    return api_key.strip(), secret_key.strip()


def resolve_contract(api: sj.Shioaji, symbol: str) -> Any:
    contracts = list(api.Contracts.Futures[symbol])
    if not contracts:
        raise RuntimeError(f"No futures contracts found for {symbol}")

    # Prefer the current front month rather than continuous placeholders.
    non_continuous = [c for c in contracts if not c.code.endswith(("R1", "R2"))]
    if non_continuous:
        return min(non_continuous, key=lambda c: getattr(c, "delivery_date", "9999/99/99"))
    return contracts[0]


def safe_update_status(api: sj.Shioaji) -> bool:
    try:
        api.update_status(api.futopt_account)
        return True
    except Exception as exc:
        print(f"update_status raised: {exc}", file=sys.stderr)
        return False


def get_positions(api: sj.Shioaji) -> list[Any]:
    try:
        return list(api.list_positions(api.futopt_account))
    except Exception as exc:
        print(f"list_positions raised: {exc}", file=sys.stderr)
        return []


def get_matching_positions(api: sj.Shioaji, contract: Any) -> list[Any]:
    contract_code = str(getattr(contract, "code", "")).upper()
    return [
        p for p in get_positions(api)
        if str(getattr(p, "code", "")).upper() == contract_code
    ]


def get_net_quantity(positions: list[Any]) -> int:
    net = 0
    for pos in positions:
        qty = int(getattr(pos, "quantity", 0) or 0)
        direction = str(getattr(pos, "direction", "")).lower()
        net += qty if "buy" in direction else -qty
    return net


def wait_for_position(
    api: sj.Shioaji,
    contract: Any,
    expected_net_qty: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        positions = get_matching_positions(api, contract)
        if get_net_quantity(positions) == expected_net_qty:
            return True
        time.sleep(0.5)
    return False


def place_ioc_market(
    api: sj.Shioaji,
    contract: Any,
    side: str,
    quantity: int,
) -> Any:
    action = sj.constant.Action.Buy if side.lower() == "buy" else sj.constant.Action.Sell
    order = api.Order(
        price=0,
        quantity=quantity,
        action=action,
        price_type=sj.constant.FuturesPriceType.MKT,
        order_type=sj.constant.OrderType.IOC,
        octype=sj.constant.FuturesOCType.Auto,
        account=api.futopt_account,
    )
    return api.place_order(contract, order)


def safe_place_ioc_market(
    api: sj.Shioaji,
    contract: Any,
    side: str,
    quantity: int,
    label: str,
) -> tuple[bool, Any]:
    try:
        result = place_ioc_market(api, contract, side, quantity)
        print(f"{label} result:", result)
        return True, result
    except Exception as exc:
        print(f"{label} API raised: {exc}", file=sys.stderr)
        return False, exc


def print_positions(api: sj.Shioaji) -> None:
    positions = get_positions(api)
    print("\nCurrent simulation positions:")
    print(positions if positions else "  (flat)")


def print_recent_trades(api: sj.Shioaji) -> None:
    trades = api.list_trades()
    print("\nRecent trades:")
    print(trades if trades else "  (none)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Sinopac Shioaji simulation futures orders")
    parser.add_argument("--symbol", default="MXF", help="Futures symbol, e.g. MXF or TXF")
    parser.add_argument("--quantity", type=int, default=1, help="Contracts per leg")
    parser.add_argument(
        "--entry-side",
        choices=("buy", "sell"),
        default="buy",
        help="First simulated trade direction",
    )
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        help="Leave the simulation position open after the first fill",
    )
    parser.add_argument(
        "--flatten-only",
        action="store_true",
        help="Do not open a new test trade; only flatten any existing simulation position",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait before refreshing trade/position status",
    )
    args = parser.parse_args()

    api_key, secret_key = load_shioaji_credentials()
    if not api_key or not secret_key:
        print("Missing SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY in .env", file=sys.stderr)
        return 1

    api = sj.Shioaji(simulation=True)
    try:
        accounts = api.login(
            api_key=api_key,
            secret_key=secret_key,
            fetch_contract=True,
            contracts_timeout=10000,
            subscribe_trade=True,
        )
        print("Login OK")
        print("Accounts:", accounts)
        print("Future account:", api.futopt_account)
        if not api.futopt_account:
            print("No futopt_account available. Check futures account/API permissions.", file=sys.stderr)
            return 2

        contract = resolve_contract(api, args.symbol.upper())
        print(f"Using contract: {contract.code} | {getattr(contract, 'name', '')}")

        if args.flatten_only:
            matching = get_matching_positions(api, contract)
            current_net_qty = get_net_quantity(matching)
            if current_net_qty == 0:
                print("No open simulation position to flatten.")
                return 0

            current_direction = str(getattr(matching[0], "direction", "")).lower()
            exit_side = "sell" if "buy" in current_direction else "buy"
            current_qty = abs(current_net_qty)
            print(f"\nFlatten-only: {exit_side.upper()} x{current_qty}")
            safe_place_ioc_market(api, contract, exit_side, current_qty, "Exit")
            time.sleep(args.wait_seconds)
            safe_update_status(api)
            wait_for_position(api, contract, 0, timeout_seconds=max(2.0, args.wait_seconds * 3))
            print_recent_trades(api)
            print_positions(api)
            return 0 if get_net_quantity(get_matching_positions(api, contract)) == 0 else 4

        print(f"\nSubmitting simulation entry: {args.entry_side.upper()} x{args.quantity}")
        entry_ok, _ = safe_place_ioc_market(
            api, contract, args.entry_side, args.quantity, "Entry"
        )

        time.sleep(args.wait_seconds)
        safe_update_status(api)
        expected_entry_qty = args.quantity if args.entry_side == "buy" else -args.quantity
        entry_position_ok = wait_for_position(
            api, contract, expected_entry_qty, timeout_seconds=max(2.0, args.wait_seconds * 3)
        )
        print_recent_trades(api)
        print_positions(api)

        if args.no_flatten:
            print("\nLeaving simulation position open (--no-flatten).")
            return 0

        current_net_qty = get_net_quantity(get_matching_positions(api, contract))
        if current_net_qty == 0:
            if entry_ok and entry_position_ok:
                print("\nNo open simulation position detected. Nothing to flatten.")
                return 0
            print("\nNo open simulation position detected after the API exception.")
            return 3

        exit_side = "sell" if current_net_qty > 0 else "buy"
        current_qty = abs(current_net_qty)
        print(f"\nFlattening simulation position: {exit_side.upper()} x{current_qty}")
        safe_place_ioc_market(api, contract, exit_side, current_qty, "Exit")

        time.sleep(args.wait_seconds)
        safe_update_status(api)
        wait_for_position(api, contract, 0, timeout_seconds=max(2.0, args.wait_seconds * 3))
        print_recent_trades(api)
        print_positions(api)
        return 0 if get_net_quantity(get_matching_positions(api, contract)) == 0 else 4
    except Exception as exc:
        print(f"Simulation test failed: {exc}", file=sys.stderr)
        return 3
    finally:
        try:
            api.logout()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
