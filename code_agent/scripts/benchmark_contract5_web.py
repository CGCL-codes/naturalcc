"""Resume the contract-5 evaluation through the same HTTP API as the Web UI.

Start agent_web_api.py first, export OPENROUTER_API_KEY, then run this script.
No direct inference, automatic model fallback, source execution or source edits.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from .check_openrouter_credits import check_credits
except ImportError:
    from check_openrouter_credits import check_credits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/contract5"
OUT = DATA / "web-sonnet45"
REPORT = ROOT / "合同-5-result.md"
MODEL = "anthropic/claude-sonnet-4.5"
CATEGORIES = {787: "缓冲区溢出", 362: "多线程竞争", 401: "内存泄漏", 78: "命令执行注入"}
TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
BUDGET = {"max_llm_calls": 8, "max_tool_calls": 20, "max_input_tokens": 120000,
          "max_output_tokens": 24000, "max_seconds": 600, "max_compaction_calls": 8}
PRODUCT_FILES = ("security_review.py", "plugins/vulnerability_detection.py", "agent_core/tools/pipeline.py")
GOAL = '''请审计当前工作区 main.c，只检测以下四类风险：缓冲区越界写/溢出(CWE-787)、
多线程竞争(CWE-362)、内存泄漏(CWE-401)、OS 命令注入(CWE-78)。
先用工作区读取工具查看完整源码，再根据数据流、边界、所有权和同步关系判断。
可使用只读漏洞扫描工具辅助，但需核实其证据和覆盖限制，不能把扫描无告警当作安全证明。
本次仅静态审计：不要修改文件、编译、运行程序或执行 shell，不调用需要 execute/write 授权的工具。
不要搜索外部数据集或答案。只报告有具体代码依据的上述风险，不把其他类型缺陷归入这四类。
最终回答只输出 JSON（可以使用 json 代码块），结构为：
{"findings":[{"cwe":787,"line":7,"reason":"具体证据及触发条件"}],"summary":"简要结论"}。
findings 中 cwe 只能为 787、362、401、78，line 为 main.c 的真实 1-based 行号。
确认没有这四类风险时 findings 为 []。以上 JSON 仅为格式示例，不表示该文件有漏洞。'''


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_text(text):
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return text.replace(key, "[REDACTED]") if key else text


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(safe_text(text), encoding="utf-8")
    temp.replace(path)


def save_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_samples():
    manifest = json.loads((DATA / "manifest.json").read_text())
    raw = (DATA / "dataset.json").read_bytes()
    if digest(raw) != manifest["dataset_sha256"]:
        raise ValueError("dataset.json SHA256 mismatch")
    original = {s["id"]: s for s in json.loads(raw)["tests"]}
    samples = {s["id"]: s for s in manifest["samples"]}
    # Four categories first, then negative controls; frozen before any inference.
    ordered = [samples[f"{cwe}-{n}"] for n in (1, 7, 2, 8, 3, 9, 4, 10, 5, 6)
               for cwe in CATEGORIES]
    if len(samples) != 40:
        raise ValueError("Expected exactly 40 selected samples")
    for sample in ordered:
        ref = original[sample["id"]]
        source = ref["code"].encode("utf-8")
        if digest(source) != sample["sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {sample['id']}")
        if any(sample[k] != ref[k] for k in ("cwe", "vulnerable", "lines")):
            raise ValueError(f"Label mismatch: {sample['id']}")
    return manifest, [{**s, "code": original[s["id"]]["code"]} for s in ordered]


def parse_answer(answer, line_count):
    text = answer.strip()
    fenced = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    if len(fenced) > 1:
        raise ValueError("Ambiguous multiple JSON answers")
    value = json.loads(fenced[0] if fenced else text)
    if not isinstance(value, dict) or not isinstance(value.get("findings"), list):
        raise ValueError("Missing findings array")
    if not isinstance(value.get("summary"), str):
        raise ValueError("Missing summary")
    for finding in value["findings"]:
        if not isinstance(finding, dict) or type(finding.get("cwe")) is not int:
            raise ValueError("Invalid finding/CWE")
        if finding["cwe"] not in CATEGORIES:
            raise ValueError("Out-of-scope CWE")
        if type(finding.get("line")) is not int or not 1 <= finding["line"] <= line_count:
            raise ValueError("Invalid source line")
        if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
            raise ValueError("Missing evidence")
    return value


def metrics(rows):
    tp = sum(s["vulnerable"] and bool(p["findings"]) for s, p in rows)
    tn = sum(not s["vulnerable"] and not p["findings"] for s, p in rows)
    fp = sum(not s["vulnerable"] and bool(p["findings"]) for s, p in rows)
    fn = sum(s["vulnerable"] and not p["findings"] for s, p in rows)
    target_hits = sum(s["vulnerable"] and any(f["cwe"] == s["cwe"] for f in p["findings"])
                      for s, p in rows)
    return {"n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "target_hits": target_hits, "strict_correct": target_hits + tn}


def rate(n, d):
    return f"{n / d:.2%} ({n}/{d})" if d else "N/A（分母为 0）"


def credit_failure(events):
    errors = [e["payload"].get("error", {}) for e in events if e["type"] == "run.failed"]
    return any(re.search(r"\b402\b|insufficient.*credit|not enough credit|credit.*insufficient",
                         json.dumps(e), re.I) for e in errors)


def credit_stop_reason(events):
    errors = [e["payload"].get("error", {}) for e in events if e["type"] == "run.failed"]
    if any("in_flight_budget_exhausted" in json.dumps(e) for e in errors):
        return "OpenRouter 402：在途请求额度占用，余额未必耗尽；已停止，待在途请求结算后可重试"
    return "OpenRouter 返回额度不足，已停止；充值后从此样本继续"


def refresh_scores(checkpoint, samples):
    """Reparse saved completed answers offline; never spend tokens on format repair."""
    for sample in samples:
        rec = checkpoint["samples"].get(sample["id"], {})
        if rec.get("status") != "unscored":
            continue
        evidence = OUT / "runs" / f"{rec['attempts'][-1]['run_id']}.json"
        state = json.loads(evidence.read_text())["state"]
        if state["status"] != "completed":
            continue
        try:
            rec["prediction"] = parse_answer(state["final_answer"], len(sample["code"].splitlines()))
            rec["status"] = "scored"
            rec.pop("error", None)
        except (ValueError, TypeError):
            pass


def report(checkpoint, manifest, samples):
    records = checkpoint["samples"]
    rows = [(s, records[s["id"]]["prediction"]) for s in samples
            if records.get(s["id"], {}).get("prediction") is not None]
    m = metrics(rows)
    attempted = sum(bool(r["attempts"]) for r in records.values())
    balance = checkpoint.get("credits", [])
    lines = ["# 合同第 5 项：Web Agent 实测结果", "",
             f"更新时间（UTC）：{now()}", "",
             f"整体文件级正确率：**{rate(m['tp'] + m['tn'], m['n'])}**。"
             f"已有效评分 **{m['n']}/40** 条，已尝试 {attempted} 条；"
             f"已尝试但未评分 {attempted - m['n']} 条。", "",
             "**性质：同一数据集上的调优后复测，按原始标签计分；不是独立泛化成绩或正式合同验收。**", "",
             f"状态：{checkpoint['stop_reason']}。", "",
             ("40 条样本已全部有效完成，以上正确率覆盖完整选定子集。" if m["n"] == len(samples)
              else "这是当前有效完成样本的阶段性结果，不是全部 40 条样本的得分。") +
             "API/额度错误、未完成 Run、输出格式错误不会当作无漏洞或正确结果。", "",
             "## 指标", "", "| 指标 | 当前结果 |", "|---|---|",
             f"| 文件级 Accuracy = (TP+TN)/N | {rate(m['tp']+m['tn'],m['n'])} |",
             f"| 告警精确率 Precision = TP/(TP+FP) | {rate(m['tp'],m['tp']+m['fp'])} |",
             f"| 文件级检出率 Recall = TP/(TP+FN) | {rate(m['tp'],m['tp']+m['fn'])} |",
             f"| 误报率 FPR = FP/(FP+TN) | {rate(m['fp'],m['fp']+m['tn'])} |",
             f"| 目标 CWE 正确率（正例须命中目标类别） | {rate(m['strict_correct'],m['n'])} |",
             f"| 目标 CWE 检出率 | {rate(m['target_hits'],m['tp']+m['fn'])} |",
             f"| TP / TN / FP / FN | {m['tp']} / {m['tn']} / {m['fp']} / {m['fn']} |", "",
             "计分单位为文件：findings 非空视为预测有风险；标签以冻结数据集为准。"
             "未逐条判定告警的真伪，也未把行号精确匹配计入正确率。"
             "因此文件级 Precision 不等同于逐条告警精确率。"
             "解析接受纯 JSON 或回答中唯一的 json 代码块；周围说明文字不参与评分，"
             "不使用另一模型改写或修复答案。", "",
             "| 类别 | 已评分 | 文件级正确率 | 目标 CWE 检出率 | 误报率 |",
             "|---|---:|---|---|---|"]
    for cwe, name in CATEGORIES.items():
        cm = metrics([(s, p) for s, p in rows if s["cwe"] == cwe])
        lines.append(f"| {name} CWE-{cwe} | {cm['n']}/10 | "
                     f"{rate(cm['tp']+cm['tn'],cm['n'])} | "
                     f"{rate(cm['target_hits'],cm['tp']+cm['fn'])} | {rate(cm['fp'],cm['fp']+cm['tn'])} |")
    if m["n"] == len(samples) and m["fp"] + m["tn"] and m["fp"] / (m["fp"] + m["tn"]) >= 0.08:
        lines += ["", "即使按本实验的文件级代理口径，误报率也仍未低于合同要求的 8%，"
                  "不能将本轮表述为合同第 5 项全部指标达标。操作系统注入场景等适用边界见文末。"]
    errors = [(s, p) for s, p in rows if bool(p["findings"]) != s["vulnerable"]]
    if errors:
        lines += ["", "## 与原标签不一致的样本", "",
                  "以下均保留在上述计分中。人工复核提示不是对标签的自动修订，也不产生另一套更高的主成绩。"]
        for sample, prediction in errors:
            lines += ["", f"- `{sample['id']}`：{'漏报' if sample['vulnerable'] else '按标签计为误报'}。"
                      f"模型结论：{prediction['summary']}",
                      f"  [原始数据集](artifacts/contract5/dataset.json) 中检索 ID `{sample['id']}` 可查看源码。"]
        error_ids = {s["id"] for s, _ in errors}
        if "362-10" in error_ids:
            lines += ["", "人工源码复核发现 `362-10` 存在标签争议：主线程把循环变量 `&t` "
                      "传入多个 pthread，工作线程通过 `*t` 读取，主线程继续修改该变量；"
                      "这些访问间没有同步，模型的竞争告警有代码依据。"
                      "原数据集说明只关注不同文件名，却未反映共享参数的问题。"
                      "本报告仍按原始负例标签计为 FP，建议独立复核并向数据集上游反馈，"
                      "不应为提高得分而让产品忽略这种风险。"]
        if "78-10" in error_ids:
            lines += ["", "`78-10` 的模型告警以长输入越界为依据，但其所述上界关系有误："
                      "`BUFSIZ-11` 小于 `BUFSIZ-9`，不是大于。"
                      "仍需改进跨函数、指针偏移与仅删除字符的过滤器的区间推理；"
                      "这里仅反驳该条具体长输入论证，不把所有异常输入路径宣告为安全。"]
    lines += ["", "## 测试链路、额度与证据", "",
              f"- 模型：OpenRouter / `{MODEL}`；无模型 fallback。",
              "- 链路：Web `/api/agent/threads` → 新消息 → Run `/step` → 读取状态和 events；"
              "与页面共用持久化 Agent 引擎，不是脚本直接询问模型，也不是浏览器点击自动化。",
              "- 每条样本一个独立对话、独立工作区，文件重命名为 main.c；原始代码不变。"
              "不向 Agent 提供 CWE 标签、正负例标签或参考行号。",
              "- 静态只读审计，不编译/执行样本、不授权写入或 execute 工具；"
              "允许 Agent 读取源码、调用只读扫描并分析。Cppcheck 执行工具不在本次授权范围。",
              "- 固定次序：样本编号 1、7、2、8、3、9、4、10、5、6，"
              "每个编号依次测试 CWE-787、362、401、78；不按模型表现筛样本。",
              f"- 每个 Run 预算：`{json.dumps(BUDGET)}`。",
              "- [完整预测、断点与配置](artifacts/contract5/web-sonnet45/checkpoint.json)；"
              "保留每条样本的结构化结论、Thread/Run ID、时间和冻结配置。",
              "- 完整 Run 日志与临时工作区不纳入版本管理，完成评分后可清理；"
              "未评分样本的 runs/ 答案须保留用于离线重解析。"
              "原始源码统一保存在 dataset.json，不再重复导出 samples/。",
              "- 本次清理未删除服务数据库中的对话历史；Web 对话标题为“合同5 · sample_NNN”。"
              "已清理工作区的历史源码文件不再可浏览，源码以数据集为准。"]
    lines += ["", "## 本轮模块优化与实验口径", "",
              "本轮在清除旧测试结果后从零运行。产品新增 `c-security-review-v1`："
              "字面量格式中无长度限制的 scanf 字符串/扫描集候选检测，以及缓冲区精确长度、"
              "命令源到执行点的数据流、同步失败路径、分配释放所有权的复核规则。"
              "规则随漏洞扫描工具返回，Web 用户也能使用；未把样本 ID、文件散列或答案表写入产品。", "",
              "**这 40 条样本已用于分析失败模式与调优，本轮是调优集复测，不是独立测试集成绩。**"
              "标签、源码、评分方法和样本顺序未修改；本轮固定模块版本后完整跑队列，"
              "不因答案错误重试，不剔除难例。独立泛化能力仍需新的未参与调优的样本验证。", "",
              "语义复核依据：[POSIX fgets](https://pubs.opengroup.org/onlinepubs/009696699/functions/fgets.html)、"
              "[格式化输出](https://pubs.opengroup.org/onlinepubs/009604499/functions/fprintf.html)、"
              "[互斥锁返回语义](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pthread_mutex_lock.html)。", "",
              "源码版本 SHA256（用于防止续跑时混入另一版检测器）：", ""]
    for name, value in checkpoint.get("protocol", {}).get("product_sha256", {}).items():
        lines.append(f"- `{name}`：`{value}`")
    if balance:
        lines += ["", "## 额度记录", "",
                  f"- 首次余额：${balance[0]['effective_remaining_usd']:.6f}；"
                  f"最近余额：${balance[-1]['effective_remaining_usd']:.6f}"
                  f"（查询时间 {balance[-1]['checked_utc']}）。",
                  f"- 两次查询之间账户余额净变化："
                  f"${balance[0]['effective_remaining_usd']-balance[-1]['effective_remaining_usd']:.6f}。"
                  "该差值可能包含期间其他请求，不等同于本实验精确账单。",
                  "- 余额入账可能延迟；在途占用或下一请求的 token 预留也可能触发 402。"
                  "停止不等于账户已扣至零，脚本不会降低预留预算或更换模型来强行扣完余额。"]
    lines += ["", "## 四个范例", "", "固定展示每类第一个有效完成样本，包含错误结论，不仅挑选成功案例。"]
    for cwe, name in CATEGORIES.items():
        examples = [(s, p) for s, p in rows if s["cwe"] == cwe]
        lines += ["", f"### {name}（CWE-{cwe}）", ""]
        if not examples:
            lines.append("尚无有效完成样本；充值续跑后自动填充，不编造示例结果。")
            continue
        sample, prediction = examples[0]
        record = records[sample["id"]]
        attempt = record["attempts"][-1]
        correct = bool(prediction["findings"]) == sample["vulnerable"]
        lines += [f"样本：`{sample['name']}`；数据集标签："
                  f"{'有漏洞' if sample['vulnerable'] else '无漏洞'}；"
                  f"文件级判定：{'正确' if correct else '错误'}。",
                  f"参考行：{sample['lines']}；Thread：`{attempt['thread_id']}`；Run：`{attempt['run_id']}`。", "",
                  f"[保存的预测与 Run 标识](artifacts/contract5/web-sonnet45/checkpoint.json)（样本 `{sample['id']}`）", "",
                  "```c", sample["code"].rstrip(), "```", "",
                  "Agent 最终结构化结论：", "", "```json",
                  json.dumps(prediction, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## 逐样本进度", "", "| 样本 | 标签 | 状态 | 预测类别 | 文件判定 |",
              "|---|---|---|---|---|"]
    for sample in samples:
        rec = records.get(sample["id"], {})
        prediction = rec.get("prediction")
        status = rec.get("status", "未测试")
        result = "—" if prediction is None else ("正确" if bool(prediction["findings"]) == sample["vulnerable"] else "错误")
        predicted = "—" if prediction is None else str(sorted({f["cwe"] for f in prediction["findings"]}))
        lines.append(f"| {sample['id']} | {'正例' if sample['vulnerable'] else '反例'} | {status} | {predicted} | {result} |")
    lines += ["", "## 数据来源与合同适用边界", "",
              "CASTLE C250 v1.2（本地快照元数据日期 2025-03-13），"
              f"[固定源版本](https://github.com/CASTLE-Benchmark/CASTLE-Benchmark/tree/{manifest['revision']})。"
              "所选四类每类 10 条，共 24 条正例、16 条反例。",
              f"dataset.json SHA256：`{manifest['dataset_sha256']}`；"
              "脚本启动时核对源码散列及其与原始 dataset.json 的标签、代码一致性。", "",
              "标签解释提示：362-1 原始标签为有竞争风险，源码虽有互斥锁，"
              "却未检查初始化/加锁返回值。是否将这些异常路径计为目标竞争风险，"
              "会影响人工判断；本报告不自行改标签，保留模型结论与原标签的分歧。", "",
              "合同要求在一个操作系统底层程序中每类注入 10 处缺陷，并满足预警准确率 >90%、误报率 <8%。"
              "本实验是独立 C 微型程序测试，每类仅 6 条正例加 4 条反例，"
              "不满足该注入场景与数量条件；合同的预警准确率还需确认是 Precision 还是其他口径。"
              "小样本与余额截断也限制外推，不能据此直接宣布验收达标。", "",
              "## 充值后继续", "", "保持 Web 服务使用相同模型启动，且环境中有 OPENROUTER_API_KEY：", "",
              "```bash", "uv run python scripts/benchmark_contract5_web.py", "```", "",
              "脚本默认续跑，保留已评分结果（包括错误），不会反复测到答对。"
              "仅额度失败的样本会在充值或在途额度释放后重开 Run；非额度失败保留为未评分。"
              "`--report-only` 可在不发起任何网络请求的情况下重新生成本报告。"]
    atomic_write(REPORT, "\n".join(lines) + "\n")


class WebClient:
    def __init__(self, base):
        url = urlparse(base)
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local HTTP Web backend is allowed")
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False  # Local API must not forward the key through a proxy.

    def call(self, method, path, **kwargs):
        response = self.session.request(method, self.base + path, timeout=660,
                                        allow_redirects=False, **kwargs)
        if response.status_code != 200:
            raise RuntimeError(f"Web {path}: HTTP {response.status_code}: {safe_text(response.text[:800])}")
        return response.json()


def run(checkpoint, manifest, samples, base):
    client = WebClient(base)
    bootstrap = client.call("GET", "/api/bootstrap")
    config = bootstrap["runtime_default_model_config"]
    if (config["provider"], config["model"], config["base_url"].rstrip("/")) != (
            "openrouter", MODEL, "https://openrouter.ai/api/v1") or config.get("fallback_models"):
        raise ValueError("Web runtime defaults do not match OpenRouter / Sonnet 4.5 / no fallback")
    signature = {"model_config": config, "budget": BUDGET, "goal": GOAL,
                 "manifest_sha256": digest((DATA / "manifest.json").read_bytes()),
                 "product_sha256": {name: digest((ROOT / name).read_bytes()) for name in PRODUCT_FILES},
                 "order": [s["id"] for s in samples], "base_url": base}
    if checkpoint.get("protocol", signature) != signature:
        raise ValueError("Evaluation protocol changed; do not mix results in the existing checkpoint")
    checkpoint["protocol"] = signature

    def persist():
        save_json(OUT / "checkpoint.json", checkpoint)
        report(checkpoint, manifest, samples)

    def credits():
        receipt = check_credits()
        checkpoint["credits"].append(receipt)
        persist()
        print(f"Available credit: ${receipt['effective_remaining_usd']:.6f}", flush=True)
        return receipt["effective_remaining_usd"]

    for index, sample in enumerate(samples, 1):
        rec = checkpoint["samples"].setdefault(sample["id"], {"attempts": []})
        if rec.get("prediction") is not None or rec.get("status") in {"unscored", "failed"}:
            continue
        if credits() <= 0:
            checkpoint["stop_reason"] = "额度不足，等待充值后续跑"
            persist()
            return
        workspace = OUT / "workspaces" / f"sample_{index:03d}"
        workspace.mkdir(parents=True, exist_ok=True)
        source = sample["code"].encode("utf-8")
        target = workspace / "main.c"
        if target.exists() and target.read_bytes() != source:
            raise ValueError(f"Workspace source changed: {sample['id']}")
        if not target.exists():
            atomic_write(target, source.decode())
        attempt = rec["attempts"][-1] if rec["attempts"] else None
        if attempt is None or rec.get("status") == "credit_exhausted":
            thread = client.call("POST", "/api/agent/threads", json={
                "title": f"合同5 · sample_{index:03d}", "workspace": str(workspace),
                "runtime_mode": "agent", "runtime_model_config": config, "budget": BUDGET})
            attempt = {"thread_id": thread["thread_id"], "created_utc": now()}
            rec["attempts"].append(attempt)
            rec["status"] = "created"
            persist()
        if "run_id" not in attempt:
            # Do not duplicate a message whose HTTP response was lost.
            if attempt.get("message_pending"):
                raise RuntimeError("Message response lost; inspect saved Thread before retrying to avoid duplicate billing")
            attempt["message_pending"] = True
            persist()
            created = client.call("POST", f"/api/agent/threads/{attempt['thread_id']}/messages", json={
                "content": GOAL, "target_files": ["main.c"],
                "context_items": [{"path": "main.c", "absolute_path": str(target), "external": False, "type": "file"}],
                "budget": BUDGET, "api_key": os.environ["OPENROUTER_API_KEY"]})
            attempt["run_id"] = created["run_id"]
            attempt.pop("message_pending")
            persist()
        run_path = f"/api/agent/runs/{attempt['run_id']}"
        state = client.call("GET", run_path)
        if state["runtime_model_config"] != config:
            raise ValueError("Run model snapshot does not match the frozen configuration")
        while state["status"] not in TERMINAL:
            print(f"[{index}/40] {sample['id']} {state['status']} LLM={state.get('llm_calls',0)}", flush=True)
            if state["status"] == "waiting_approval":
                state = client.call("POST", run_path + "/reject")
            elif state["status"] == "paused":
                raise RuntimeError(f"Run paused: {attempt['run_id']}; review it in the Web UI")
            else:
                state = client.call("POST", run_path + "/step")
            events = client.call("GET", run_path + "/events")["events"]
            save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
            rec["status"] = state["status"]
            persist()
        events = client.call("GET", run_path + "/events")["events"]
        save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
        attempt["status"] = state["status"]
        attempt["finished_utc"] = now()
        if credit_failure(events):
            rec["status"] = "credit_exhausted"
            checkpoint["stop_reason"] = credit_stop_reason(events)
            persist()
            credits()
            return
        if state["status"] == "completed":
            try:
                rec["prediction"] = parse_answer(state["final_answer"], len(source.decode().splitlines()))
                rec["status"] = "scored"
            except (ValueError, TypeError) as exc:
                rec["status"] = "unscored"
                rec["error"] = str(exc)
        else:
            rec["status"] = "failed"
            checkpoint["stop_reason"] = f"非额度故障：{sample['id']} / {state['status']}；查看 Run 证据"
            persist()
            credits()
            return
        print(f"[{index}/40] {sample['id']}: {rec['status']}", flush=True)
        persist()
    checkpoint["stop_reason"] = "样本队列已处理完毕（未评分项见进度表）"
    credits()
    persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, samples = load_samples()
        path = OUT / "checkpoint.json"
        checkpoint = json.loads(path.read_text()) if path.exists() else {
            "created_utc": now(), "samples": {}, "credits": [], "stop_reason": "尚未开始"}
        refresh_scores(checkpoint, samples)
        if args.report_only:
            save_json(path, checkpoint)
            report(checkpoint, manifest, samples)
            return
        try:
            if not os.environ.get("OPENROUTER_API_KEY", "").strip():
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            checkpoint["stop_reason"] = "测试进行中"
            run(checkpoint, manifest, samples, args.base_url)
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint["stop_reason"] = safe_text(f"已停止：{type(exc).__name__}: {exc}")
            print(checkpoint["stop_reason"], flush=True)
            raise SystemExit(1) from None
        finally:
            save_json(path, checkpoint)
            report(checkpoint, manifest, samples)
            print(f"Report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
