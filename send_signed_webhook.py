from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values


def load_env() -> dict[str, str]:
    env_path = Path(__file__).resolve().parent / ".env"
    values = dotenv_values(env_path)
    return {str(k): str(v) for k, v in values.items() if k and v is not None}


def build_signature(secret: str, timestamp: str, body: bytes) -> str:
    payload = timestamp.encode("utf-8") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a correctly signed webhook to the local trading bot")
    parser.add_argument("--url", default="http://127.0.0.1:8000/webhook")
    parser.add_argument("--action", choices=("buy", "sell", "exit"), default="buy")
    parser.add_argument("--sentiment", choices=("long", "short", "flat"), default="long")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--ticker", default="TMF")
    parser.add_argument("--type", dest="signal_type", choices=("entry", "exit", "eod_close"), default="entry")
    args = parser.parse_args()

    env = load_env()
    secret = env.get("WEBHOOK_SECRET", "").strip()
    if not secret:
        raise SystemExit("WEBHOOK_SECRET is missing in .env")

    payload = {
        "action": args.action,
        "sentiment": args.sentiment,
        "quantity": args.quantity,
        "ticker": args.ticker,
        "type": args.signal_type,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    signature = build_signature(secret, timestamp, body)

    headers = {
        "Content-Type": "application/json",
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }

    resp = httpx.post(args.url, content=body, headers=headers, timeout=20)
    print("status:", resp.status_code)
    print(resp.text)
    return 0 if resp.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
