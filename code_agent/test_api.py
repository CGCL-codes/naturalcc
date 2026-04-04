import argparse
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

import requests

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 OpenRouter API 可用性，并输出额度/用量信息。"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API Key，默认读取环境变量 OPENROUTER_API_KEY。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"要检查的模型，默认 {DEFAULT_MODEL}。",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenRouter API Base URL，默认 {DEFAULT_BASE_URL}。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"请求超时秒数，默认 {DEFAULT_TIMEOUT}。",
    )
    parser.add_argument(
        "--generation-check",
        action="store_true",
        help="额外发起一次最小生成请求（max_tokens=1）做真实连通性验证，会产生极少量 token 消耗。",
    )
    return parser.parse_args()


def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def fmt_decimal(value: Optional[Decimal], digits: int = 4) -> str:
    if value is None:
        return "-"
    quant = Decimal("1") if digits == 0 else Decimal(f"1.{'0' * digits}")
    return f"{value.quantize(quant):,}"


def fmt_bool(value: Optional[bool]) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "UNKNOWN"


def fmt_text(value: Any) -> str:
    if value in (None, "", "None"):
        return "-"
    return str(value)


def mask_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "*" * len(api_key)
    return f"{api_key[:6]}...{api_key[-4:]}"


def print_rule(char: str = "=") -> None:
    print(char * 78)


def print_title(title: str) -> None:
    print_rule("=")
    print(title)
    print_rule("=")


def print_section(title: str, rows: Dict[str, Any]) -> None:
    print(f"\n[{title}]")
    width = max(len(label) for label in rows) if rows else 0
    for label, value in rows.items():
        print(f"  {label:<{width}} : {value}")


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "naturalcc-openrouter-check",
    }


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "status_code": None,
        "json": None,
        "text": "",
        "error": None,
    }
    try:
        response = session.request(method=method, url=url, timeout=timeout, **kwargs)
        result["status_code"] = response.status_code
        result["text"] = response.text
        try:
            result["json"] = response.json()
        except ValueError:
            result["json"] = None
        result["ok"] = response.ok
        return result
    except requests.exceptions.RequestException as exc:
        result["error"] = str(exc)
        return result


def find_model(models_payload: Dict[str, Any], target_model: str) -> Optional[Dict[str, Any]]:
    models = models_payload.get("data") or []
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("id") == target_model or model.get("canonical_slug") == target_model:
            return model
    return None


def summarize_error(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return result["error"]
    payload = result.get("json")
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
        message = payload.get("message")
        if message:
            return str(message)
    status = result.get("status_code")
    if status is not None:
        text = (result.get("text") or "").strip()
        return f"HTTP {status}" if not text else f"HTTP {status}: {text[:200]}"
    return "未知错误"


def run_generation_check(
    session: requests.Session,
    base_url: str,
    timeout: int,
    model: str,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    result = request_json(
        session,
        "POST",
        f"{base_url}/chat/completions",
        timeout=timeout,
        json=payload,
    )
    summary: Dict[str, Any] = {
        "requested": True,
        "ok": result["ok"],
        "status_code": result["status_code"],
        "error": None,
        "reply": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    if not result["ok"]:
        summary["error"] = summarize_error(result)
        return summary

    payload_json = result.get("json") or {}
    choices = payload_json.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        summary["reply"] = message.get("content")

    usage = payload_json.get("usage") or {}
    summary["prompt_tokens"] = usage.get("prompt_tokens")
    summary["completion_tokens"] = usage.get("completion_tokens")
    summary["total_tokens"] = usage.get("total_tokens")
    return summary


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("未提供 OpenRouter API Key。请设置 OPENROUTER_API_KEY 或使用 --api-key。", file=sys.stderr)
        return 2

    api_key = args.api_key.strip()
    base_url = args.base_url.rstrip("/")
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    session = requests.Session()
    session.headers.update(build_headers(api_key))

    key_result = request_json(session, "GET", f"{base_url}/key", timeout=args.timeout)
    credits_result = request_json(session, "GET", f"{base_url}/credits", timeout=args.timeout)
    models_result = request_json(session, "GET", f"{base_url}/models", timeout=args.timeout)

    key_data = ((key_result.get("json") or {}).get("data") or {}) if key_result["ok"] else {}
    credits_data = (
        ((credits_result.get("json") or {}).get("data") or {}) if credits_result["ok"] else {}
    )
    target_model = find_model(models_result.get("json") or {}, args.model) if models_result["ok"] else None

    total_credits = to_decimal(credits_data.get("total_credits"))
    total_usage = to_decimal(credits_data.get("total_usage"))
    total_remaining = (
        total_credits - total_usage if total_credits is not None and total_usage is not None else None
    )

    key_limit = to_decimal(key_data.get("limit"))
    key_usage = to_decimal(key_data.get("usage"))
    key_remaining = to_decimal(key_data.get("limit_remaining"))

    metadata_available = key_result["ok"] and models_result["ok"] and target_model is not None
    if key_remaining is not None:
        quota_ok = key_remaining > 0
    elif total_remaining is not None:
        quota_ok = total_remaining > 0
    else:
        quota_ok = None

    generation_summary = None
    if args.generation_check:
        generation_summary = run_generation_check(
            session=session,
            base_url=base_url,
            timeout=args.timeout,
            model=args.model,
        )

    print_title("OpenRouter API 检查报告")
    print_section(
        "基础信息",
        {
            "检查时间": now,
            "Base URL": base_url,
            "目标模型": args.model,
            "Key 掩码": mask_key(api_key),
            "检查模式": "metadata only (0 token)" if not args.generation_check else "metadata + generation",
        },
    )

    availability_rows: Dict[str, Any] = {
        "Key 接口可达": fmt_bool(key_result["ok"]),
        "模型列表可达": fmt_bool(models_result["ok"]),
        "目标模型存在": fmt_bool(target_model is not None if models_result["ok"] else None),
        "额度看起来足够": fmt_bool(quota_ok),
        "默认结论": "可用" if metadata_available and quota_ok is not False else "需人工确认",
    }
    if not key_result["ok"]:
        availability_rows["Key 错误"] = summarize_error(key_result)
    if not models_result["ok"]:
        availability_rows["模型错误"] = summarize_error(models_result)
    print_section("可用性", availability_rows)

    account_rows: Dict[str, Any] = {
        "总额度 (credits)": fmt_decimal(total_credits),
        "已使用 (usage)": fmt_decimal(total_usage),
        "剩余额度": fmt_decimal(total_remaining),
    }
    if credits_result["ok"]:
        account_rows["额度接口"] = "YES"
    else:
        account_rows["额度接口"] = "NO"
        account_rows["额度错误"] = summarize_error(credits_result)
    print_section("账户额度", account_rows)

    key_rows: Dict[str, Any] = {
        "当前 Key 标签": fmt_text(key_data.get("label")),
        "Key 限额": fmt_decimal(key_limit),
        "Key 已用": fmt_decimal(key_usage),
        "Key 剩余": fmt_decimal(key_remaining),
        "日用量": fmt_decimal(to_decimal(key_data.get("usage_daily"))),
        "周用量": fmt_decimal(to_decimal(key_data.get("usage_weekly"))),
        "月用量": fmt_decimal(to_decimal(key_data.get("usage_monthly"))),
        "免费层": fmt_bool(key_data.get("is_free_tier")),
        "管理 Key": fmt_bool(key_data.get("is_management_key")),
        "额度重置周期": fmt_text(key_data.get("limit_reset")),
        "到期时间": fmt_text(key_data.get("expires_at")),
    }
    print_section("当前 Key 用量", key_rows)

    model_rows: Dict[str, Any] = {
        "模型存在": fmt_bool(target_model is not None if models_result["ok"] else None),
        "模型名称": fmt_text((target_model or {}).get("name")),
        "上下文长度": fmt_text((target_model or {}).get("context_length")),
        "输入单价": fmt_text(((target_model or {}).get("pricing") or {}).get("prompt")),
        "输出单价": fmt_text(((target_model or {}).get("pricing") or {}).get("completion")),
        "请求单价": fmt_text(((target_model or {}).get("pricing") or {}).get("request")),
    }
    print_section("模型信息", model_rows)

    if generation_summary is not None:
        gen_rows: Dict[str, Any] = {
            "真实生成检查": fmt_bool(generation_summary["ok"]),
            "HTTP 状态": generation_summary["status_code"] or "-",
            "回复内容": generation_summary["reply"] or "-",
            "prompt_tokens": generation_summary["prompt_tokens"] or "-",
            "completion_tokens": generation_summary["completion_tokens"] or "-",
            "total_tokens": generation_summary["total_tokens"] or "-",
        }
        if generation_summary.get("error"):
            gen_rows["生成错误"] = generation_summary["error"]
        print_section("低成本生成验证", gen_rows)

    print("\n说明:")
    print("  1. 默认只查元数据，不会触发模型生成，因此通常不消耗 token。")
    print("  2. “默认结论=可用”基于 key、模型、额度元数据推断；最严格验证请加 --generation-check。")
    print("  3. 额度接口展示的是账户级额度；当前 Key 用量展示的是该 key 自身的限制和消耗。")

    if generation_summary is not None:
        return 0 if generation_summary["ok"] else 1
    return 0 if metadata_available and quota_ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
