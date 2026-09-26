"""Read OpenRouter credit metadata without generating tokens or exposing keys."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests

BASE_URL = "https://openrouter.ai/api/v1"


def check_credits(required_usd: float = 0) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    responses = {}
    for endpoint in ("credits", "key"):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}",
                                    headers={"Authorization": f"Bearer {key}"}, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenRouter {endpoint}: {type(exc).__name__}") from None
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter {endpoint}: HTTP {response.status_code}")
        responses[endpoint] = response.json()["data"]
    account, current = responses["credits"], responses["key"]
    remaining = Decimal(str(account["total_credits"])) - Decimal(str(account["total_usage"]))
    limit_remaining = current.get("limit_remaining")
    if current.get("limit") is not None and limit_remaining is None:
        raise RuntimeError("Key has a spending limit but limit_remaining is unavailable")
    available = remaining if limit_remaining is None else min(remaining, Decimal(str(limit_remaining)))
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "account_total_credits_usd": float(account["total_credits"]),
            "account_total_usage_usd": float(account["total_usage"]),
            "account_remaining_usd": float(remaining),
            "key_limit_usd": current.get("limit"),
            "key_remaining_usd": limit_remaining,
            "key_limit_reset": current.get("limit_reset"),
            "effective_remaining_usd": float(available),
            "required_usd": required_usd,
            "sufficient": available > 0 and available >= Decimal(str(required_usd)),
            "generation_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-usd", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.required_usd < 0:
        parser.error("--required-usd must be non-negative")
    try:
        result = check_credits(args.required_usd)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["sufficient"] else 2)


if __name__ == "__main__":
    main()
