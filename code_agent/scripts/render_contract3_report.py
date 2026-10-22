"""Render the single canonical Contract-3 report from result artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "contract3"
REPORT = ROOT / "合同-3-result.md"
NAMES = {"array_oob": "数组越界", "string_overflow": "字符串溢出", "null_pointer": "空指针调用"}


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def direct_table(result: dict[str, Any]) -> list[str]:
    lines = ["| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category, name in NAMES.items():
        value = result["metrics"][category]
        lines.append(f"| {name} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(value['detection_rate'])} | {percent(value['false_positive_rate'])} |")
    value = result["metrics"]["overall"]
    lines.append(f"| 总体 | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(value['detection_rate'])} | {percent(value['false_positive_rate'])} |")
    return lines


def direct_scope(result: dict[str, Any]) -> list[str]:
    return [f"- {item['name']}：`{item['commit']}`，{item['source_lines']:,} 行，{item['elapsed_seconds']:.2f} 秒，覆盖 `{item['coverage']['status']}`。" for item in result["projects"]]


def project_metrics(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    tp = sum(row["expected"] == "defect" and row["detected"] for row in rows)
    fn = sum(row["expected"] == "defect" and not row["detected"] for row in rows)
    fp = sum(row["expected"] == "control" and row["detected"] for row in rows)
    tn = sum(row["expected"] == "control" and not row["detected"] for row in rows)
    return tp, fn, fp, tn


def direct_project_table(result: dict[str, Any]) -> list[str]:
    lines = ["| 工程 | TP | FN | FP | TN | 检出率 | 误报率 | 覆盖 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for project in result["projects"]:
        rows = [row for row in result["cases"] if row["project"] == project["id"]]
        tp, fn, fp, tn = project_metrics(rows)
        lines.append(f"| {project['name']} | {tp} | {fn} | {fp} | {tn} | {percent(tp / (tp + fn))} | {percent(fp / (fp + tn))} | {project['coverage']['status']} |")
    return lines


def web_metrics(checkpoint: dict[str, Any]) -> dict[str, int]:
    rows = [row for record in checkpoint.get("samples", {}).values() if record.get("status") == "scored" for row in (record.get("rows") or [])]
    return {"n": len(rows), "tp": sum(row["expected"] == "defect" and row["detected"] for row in rows),
            "fn": sum(row["expected"] == "defect" and not row["detected"] for row in rows),
            "fp": sum(row["expected"] == "control" and row["detected"] for row in rows),
            "tn": sum(row["expected"] == "control" and not row["detected"] for row in rows)}


def web_section(title: str, checkpoint: dict[str, Any] | None, artifact: str) -> list[str]:
    lines = [f"### {title}", ""]
    if not checkpoint:
        return lines + ["尚未运行；启动 Web 服务后执行对应 `--suite`。"]
    value = web_metrics(checkpoint)
    scored = sum(record.get("status") == "scored" for record in checkpoint.get("samples", {}).values())
    return lines + ["Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。",
                    f"- 状态：{checkpoint.get('stop_reason', '未知')}。",
                    f"- 已有效评分：{scored}/9 组，{value['n']} 个函数；TP/FN/FP/TN：{value['tp']}/{value['fn']}/{value['fp']}/{value['tn']}。",
                    f"- 工件：`artifacts/contract3/{artifact}/`。"]


def web_summary(checkpoint: dict[str, Any] | None) -> tuple[str, str]:
    if not checkpoint:
        return "尚未运行", "—"
    value = web_metrics(checkpoint)
    scored = sum(record.get("status") == "scored" for record in checkpoint.get("samples", {}).values())
    detection = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else None
    fpr = value["fp"] / (value["fp"] + value["tn"]) if value["fp"] + value["tn"] else None
    return f"{scored}/9 组有效评分；{checkpoint.get('stop_reason', '未知')}", f"{percent(detection)} / {percent(fpr)}"


def web_progress(checkpoint: dict[str, Any] | None) -> list[str]:
    if not checkpoint:
        return []
    lines = ["", "| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |", "|---|---|---:|---|---|"]
    for sample_id, record in sorted(checkpoint.get("samples", {}).items()):
        rows = (record.get("rows") or []) if record.get("status") == "scored" else []
        tp, fn, fp, tn = project_metrics(rows)
        attempt = (record.get("attempts") or [{}])[-1]
        lines.append(f"| {sample_id} | {record.get('status', '未开始')} | {len(rows)} | {tp}/{fn}/{fp}/{tn} | `{attempt.get('run_id', '—')}` |")
    return lines


def web_rows(checkpoint: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not checkpoint:
        return []
    return [row for record in checkpoint.get("samples", {}).values() if record.get("status") == "scored"
            for row in (record.get("rows") or [])]


def counted_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp, fn, fp, tn = project_metrics(rows)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def web_category_table(checkpoint: dict[str, Any] | None) -> list[str]:
    rows = web_rows(checkpoint)
    lines = ["| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category, name in NAMES.items():
        value = counted_metrics([row for row in rows if row["category"] == category])
        lines.append(f"| {name} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(rate(value['tp'], value['tp'] + value['fn']))} | {percent(rate(value['fp'], value['fp'] + value['tn']))} |")
    value = counted_metrics(rows)
    lines.append(f"| 总体 | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(rate(value['tp'], value['tp'] + value['fn']))} | {percent(rate(value['fp'], value['fp'] + value['tn']))} |")
    return lines


def web_project_table(checkpoint: dict[str, Any] | None) -> list[str]:
    rows = web_rows(checkpoint)
    labels = {"freertos_kernel": "FreeRTOS Kernel", "zephyr": "Zephyr", "rt_thread": "RT-Thread"}
    lines = ["| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for project, name in labels.items():
        value = counted_metrics([row for row in rows if row["project"] == project])
        lines.append(f"| {name} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(rate(value['tp'], value['tp'] + value['fn']))} | {percent(rate(value['fp'], value['fp'] + value['tn']))} |")
    return lines


HARD_SUITES = {"balanced-hard": ("平衡验收集（加难版）", "加难验收链路"),
               "independent-hard": ("独立压力集（加难版）", "加难能力边界")}
HARD_INTRO = ("加难版与原两套测试集分开计分，不替换、平均或择优合并原结果。每类每工程仍为 20 正例 + 20 对照，"
              "但对照中加入“看起来危险、实际安全”的近邻写法（守卫后访问、恰好容纳的 strcpy/strcat、"
              "按 sizeof 截断的 snprintf/strncpy/memcpy、判空后返回、重新赋值或条件返回非空的指针），"
              "正例加入差一错误（`>`/`>=` 守卫、`<=` 循环、`sizeof` 计数下标、错误的 strncpy/snprintf 长度、判空后未返回）。"
              "40 个函数按固定种子打乱顺序，不能再按位置推断标签。平衡加难版为 10 个原平衡形态 + 10 个加难形态；"
              "压力加难版为 5 个原压力形态 + 15 个加难形态。所有加难形态的正例/对照均已用 GCC ASan/UBSan 单独编译运行核对："
              "正例触发越界、溢出或空指针错误，对照运行无报错。"
              "加难版要求每条 Finding 给出所在函数名并按函数名计分：首轮仅按行号计分时，模型自行数行的偏移随文件位置累积，"
              "在打乱顺序后把正例告警落到相邻对照上（67.22% / 21.67%），该数值反映的是行号定位误差而非误判，"
              "已归档于 `artifacts/contract3/archive-web-sonnet45-balanced-hard-line-scored/`，不作为加难版成绩。")


def hard_variant_table(checkpoint: dict[str, Any]) -> list[str]:
    rows = [row for row in web_rows(checkpoint) if row.get("variant")]
    lines = ["| 类别 | 形态 | 正例 TP/总数 | 对照 FP/总数 |", "|---|---|---:|---:|"]
    for category, name in NAMES.items():
        variants = sorted({row["variant"] for row in rows if row["category"] == category})
        for variant in variants:
            group = [row for row in rows if row["category"] == category and row["variant"] == variant]
            positives = [row for row in group if row["expected"] == "defect"]
            controls = [row for row in group if row["expected"] == "control"]
            lines.append(f"| {name} | `{variant}` | {sum(row['detected'] for row in positives)}/{len(positives)} | {sum(row['detected'] for row in controls)}/{len(controls)} |")
    return lines


FLOW_GROUPS = {"宏/全局常量/全局函数控制流（09–14）": {"09", "10", "11", "12", "13", "14"},
               "跨函数静态/全局标志（21–22）": {"21", "22"},
               "跨函数传参/返回/函数指针/静态全局（41–45）": {"41", "42", "44", "45"},
               "跨文件数据流（51–68）": {"51", "52", "53", "54", "61", "63", "64", "65", "66", "67", "68"}}


JULIET_SETS = {"juliet": "Juliet 调优集", "juliet-holdout": "Juliet 留出集"}


def juliet_samples(name: str = "juliet") -> dict[str, dict[str, Any]]:
    dataset = load(DATA / name / "dataset.json") or {"samples": []}
    return {sample["id"]: {key: value for key, value in sample.items() if key != "files"} for sample in dataset["samples"]}


def juliet_rows(checkpoint: dict[str, Any], name: str = "juliet") -> list[dict[str, Any]]:
    samples = juliet_samples(name)
    return [{**samples[identifier], "detected": bool(record["prediction"]["findings"]), "record": record}
            for identifier, record in checkpoint.get("samples", {}).items() if record.get("status") == "scored"]


def juliet_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {"tp": sum(r["vulnerable"] and r["detected"] for r in rows), "fn": sum(r["vulnerable"] and not r["detected"] for r in rows),
            "fp": sum(not r["vulnerable"] and r["detected"] for r in rows), "tn": sum(not r["vulnerable"] and not r["detected"] for r in rows)}


def juliet_table(rows: list[dict[str, Any]], groups: list[tuple[str, Any]]) -> list[str]:
    lines = ["| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, keep in groups:
        v = juliet_counts([row for row in rows if keep(row)])
        lines.append(f"| {name} | {v['tp']} | {v['fn']} | {v['fp']} | {v['tn']} | {percent(rate(v['tp'], v['tp'] + v['fn']))} | {percent(rate(v['fp'], v['fp'] + v['tn']))} |")
    return lines


def juliet_summary(checkpoint: dict[str, Any], name: str = "juliet") -> tuple[str, str]:
    rows = juliet_rows(checkpoint, name)
    v = juliet_counts(rows)
    total = len(juliet_samples(name))
    return (f"{len(rows)}/{total} 条有效评分；{checkpoint.get('stop_reason', '未知')}",
            f"{percent(rate(v['tp'], v['tp'] + v['fn']))} / {percent(rate(v['fp'], v['fp'] + v['tn']))}")


def juliet_section(runs: list[tuple[str, str, dict[str, Any]]], manifest: dict[str, Any]) -> list[str]:
    lines = ["## Juliet 第三方测试集（跨函数、宏、依赖上下文）", "",
             "前述 RTOS 平衡/压力集与加难版都是本项目脚本生成的 fixture：样例形态由我们决定，分数可以被样例设计左右，"
             "因此不宜单独作为检测能力证据。本节参照合同 5 的做法改用第三方冻结数据集与标签：", "",
             "- 来源：[NIST SARD Juliet Test Suite for C/C++ v1.3](https://samate.nist.gov/SARD/test-suites/112)，"
             "zip SHA256 `ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb`；标签取自 Juliet 自带的 "
             "`OMITBAD`/`OMITGOOD` 划分：bad 路径为有缺陷样本，good 路径（goodG2B/goodB2G/good1 等）为安全样本。",
             "- 范围：数组越界取 CWE-121 的 CWE129 下标与 CWE805 int 循环族；字符串溢出取 CWE-121 的 char 型 strcpy/strcat/strncpy/"
             "strncat/snprintf 族；空指针调用取 CWE-476 C 族。只纳入依赖上下文的流变体："
             + "、".join(FLOW_GROUPS) + "。",
             f"- 选样：每类按 `sha256(\"{manifest['selection_salt']}:<测试用例名>\")` 排序取前 {manifest['per_category']} 个测试用例，"
             "每个用例生成 1 个缺陷样本 + 1 个安全样本；选样在任何模型运行前冻结，不按结果筛选或剔除难例。",
             "- 去标签：删除全部注释，解析掉 OMITBAD/OMITGOOD/INCLUDEMAIN 块，把含 CWE/bad/good/G2B/B2G 的标识符改为 `sym_NN`，"
             "文件改名为 `src_a.c`…，并删去支持代码中的 `good1()`/`bad1()` 空桩；每个样本均与支持代码一起通过 `gcc -fsyntax-only`。"
             "已知残余线索：安全样本通常比缺陷样本多一个调用多个子函数的入口函数。",
             "- 执行：与合同 5 相同的 Web `/api/agent/threads` → Run 链路、OpenRouter/Sonnet 4.5、无 fallback；每个样本独立 Thread 与工作区，"
             "固定次序，已评分（含错误）不重问。计分为文件级：最终 JSON 中出现目标类别 Finding 即判为有缺陷。",
             "- 版本：`c-security-review-v1` 为未调优基线；之后的版本按调优集失败模式修改产品复核规则（不改样本、标签、提示词和计分），"
             "**因此调优集上的新版本成绩是调优集复测，不是独立成绩**。留出集按同一哈希顺序取后续用例，单独冻结，不参与调优，"
             "用于检验泛化。",
             "- 出题基线：`*-exam-baseline` 是给学生做检测器优化题用的**故意削弱**版本（环境变量 `CODE_AGENT_REVIEW_PROFILE=exam-baseline`），"
             "在完整规则上追加两条“跳过某类检查”的范围限制（跳过 strncat、跳过与 NULL 比较的分支内解引用），"
             "并按检出率约 86%、误报率约 13% 校准。它不是产品能力成绩，也不能用于验收；完整规则（参考答案）见同名不带后缀的版本。", ""]
    for name, version, checkpoint in runs:
        lines += juliet_run(name, version, checkpoint)
    return lines


def juliet_run(name: str, version: str, checkpoint: dict[str, Any]) -> list[str]:
    rows = juliet_rows(checkpoint, name)
    v = juliet_counts(rows)
    total = len(juliet_samples(name))
    title = f"### {JULIET_SETS[name]} · `{version}`" + ("（出题基线，故意削弱）" if "-exam-baseline" in version else "")
    lines = [title, "",
             f"已有效评分 {len(rows)}/{total} 条；TP/FN/FP/TN：{v['tp']}/{v['fn']}/{v['fp']}/{v['tn']}；"
             f"状态：{checkpoint.get('stop_reason', '未知')}。工件：`artifacts/contract3/{name}/web-sonnet45-{version}/`。", ""]
    lines += juliet_table(rows, [(name, lambda row, c=c: row["category"] == c) for c, name in NAMES.items()] + [("总体", lambda row: True)]) + [""]
    lines += juliet_table(rows, [(name, lambda row, f=flows: row["flow"] in f) for name, flows in FLOW_GROUPS.items()]) + [""]
    wrong = [row for row in rows if row["vulnerable"] != row["detected"]]
    if wrong:
        lines += ["与 Juliet 标签不一致的样本（全部保留在计分中）：", ""]
        for row in sorted(wrong, key=lambda r: r["id"]):
            kind = "漏报" if row["vulnerable"] else "误报"
            lines.append(f"- `{row['id']}` `{row['testcase'].split('__', 1)[1]}`（{kind}）：{row['record']['prediction']['summary'][:160]}")
        lines.append("")
    return lines


def render() -> str:
    manifest = load(DATA / "manifest.json") or {}
    balanced = load(DATA / "balanced-cppcheck" / "result.json")
    independent = load(DATA / "independent-cppcheck" / "result.json")
    web_balanced = load(DATA / "web-sonnet45-balanced" / "checkpoint.json")
    web_independent = load(DATA / "web-sonnet45-independent" / "checkpoint.json")
    juliet_runs = [(path.parent.parent.name, path.parent.name.removeprefix("web-sonnet45-"), load(path))
                   for name in ("juliet", "juliet-holdout") for path in sorted((DATA / name).glob("web-sonnet45-*/checkpoint.json"))]
    web_hard = {suite: load(DATA / f"web-sonnet45-{suite}" / "checkpoint.json") for suite in HARD_SUITES}
    lines = ["# 合同指标 3：高频缺陷检测总报告", "", f"更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
             "本报告是唯一的指标 3 Markdown 入口。各测试集的真值、原始结果、模型 Run 和覆盖状态保留在各自工件目录；不得择优挑选或平均不同测试集的分数。", "",
             "## 统计口径与前后端字段契约", "",
             "统计单位为单个注入函数或安全对照函数。后端/脚本只产出原始数值，不产出 `passed`、`verdict` 或阈值结论；前端若需展示达标状态，应基于下列字段和其配置阈值自行计算。", "",
             "- `categories.array_oob`、`categories.string_overflow`、`categories.null_pointer`：每个对象均含 `tp`、`fn`、`fp`、`tn`、`detection_rate`、`false_positive_rate`。", "- `overall`：将三类目标样本合并后的同名字段；不得与独立压力集或不同运行批次平均。", "- `scored_groups`、`scored_functions`：有效评分组数和函数数；未完成、模型/API 失败、超时或无效 JSON 的组不进入 TP/FN/FP/TN。", "",
             "接口现状：上述 Web 基准统计由 `benchmark_contract3_web.py` 写入 checkpoint/本报告，尚未提供历史 Web 基准的聚合统计 HTTP 端点。单次漏洞扫描已可通过 `/api/run` 的 `artifacts.contract_statistics` 或 Agent 扫描工具的 `data.contract_statistics` 返回统计：需传入 `scan_type` 和 `ground_truth_file`，分类字段为 `by_category`，总体为 `overall`。这是独立的单次扫描接口，不能用其结果覆盖本报告的模型基准成绩；字段和用法见 [安全检测联调说明](SECURITY_ACCEPTANCE.md)。", "",
             "定义：TP=正例被至少一个目标类别 Finding 命中；FN=正例没有目标类别 Finding；FP=安全对照被目标类别 Finding 命中；TN=安全对照没有目标类别 Finding。检出率=`TP/(TP+FN)`；误报率=`FP/(FP+TN)`。本报告指标 3 表只展示检出率和误报率。单次扫描的统一结构还可返回 `warning_accuracy=TP/(TP+FP)`，供指标 5 的预警准确率展示使用，不能把它称为检出率。", "",
             "通俗示例：已注入数组越界且报告数组越界为 TP；已注入空指针调用但未报告为 FN；安全数组访问却报告数组越界为 FP；安全字符串复制且未报告字符串溢出为 TN。这里的“正/负”只针对当前测试类别和 Ground Truth，不表示文件中是否还有其他类型的问题。", "",
             "## Finding 过滤规则", "",
             "指标 3 只评分数组越界、字符串溢出、空指针调用三类目标 Finding。CWE-398、CWE-563 等代码质量诊断，以及目标函数范围以外的所有 Finding，可作为额外诊断保留，但绝不参与 TP/FN/FP/TN、检出率或误报率。", "",
             "例如：数组越界正例只报告 CWE-563 未使用变量，不算 TP；数组安全对照只报告 CWE-398 风格问题，不算 FP。只有类别匹配且命中函数行范围的 Finding 才能改变当前指标统计。指标 5 应采用同样原则，只把缓冲区溢出、多线程竞争、内存泄漏和命令执行漏洞纳入其统计。", "",
             "- Web：每个任务调用 `vulnerability_detection` 时固定传入 `scan_type:\"frequent_defects\"`；不传 ground truth，真值始终留在评分脚本侧。工具的 `contract_statistics.status=not_evaluated` 不影响脚本按模型最终返回的 `prediction.findings` 评分。每个任务只接受与该任务类别相同的 `prediction.findings[].category`。Finding 的 `line` 必须落在对应真值函数的 `[line_start, line_end]` 内。类别不符的项写入 `out_of_scope_findings`，不评分；每个函数最终的匹配证据写入 `rows[].matched_findings`。", "- 本地 Cppcheck：此历史基准按 `manifest.json` 中该类别的 `cppcheck_ids` 或 `cwes` 匹配，再限定文件与函数行范围；原始扫描产生的其他诊断不进入 `matched_findings`。新增单次扫描使用 `security_contracts.py` 的规则及源操作分类，两套映射不互换。", ""]
    lines += ["## 当前结果总览", "", "| 测试集 / 运行方式 | 完成状态 | 检出率 / 误报率 | 定位 |", "|---|---|---|---|"]
    balanced_local = "—" if not balanced else f"{percent(balanced['metrics']['overall']['detection_rate'])} / {percent(balanced['metrics']['overall']['false_positive_rate'])}"
    independent_local = "—" if not independent else f"{percent(independent['metrics']['overall']['detection_rate'])} / {percent(independent['metrics']['overall']['false_positive_rate'])}"
    balanced_web_status, balanced_web_rate = web_summary(web_balanced)
    independent_web_status, independent_web_rate = web_summary(web_independent)
    lines += [f"| 平衡验收集 Web | {balanced_web_status} | {balanced_web_rate} | Web API 验收链路 |",
              f"| 平衡验收集本地 | {'已完成' if balanced else '未保留'} | {balanced_local} | 推荐本地验收口径 |",
              f"| 独立压力集 Web | {independent_web_status} | {independent_web_rate} | 能力边界 Web 链路 |",
              f"| 独立压力集本地 | {'已完成' if independent else '未保留'} | {independent_local} | 能力边界对照 |"]
    for name, version, checkpoint in juliet_runs:
        status, value = juliet_summary(checkpoint, name)
        role = "出题基线（故意削弱，供学生优化）" if "-exam-baseline" in version else "跨函数/宏/上下文，第三方标签"
        lines.append(f"| {JULIET_SETS[name]} Web · `{version}` | {status} | {value} | {role} |")
    for suite, (name, role) in HARD_SUITES.items():
        if web_hard[suite]:
            status, value = web_summary(web_hard[suite])
            lines.append(f"| {name} Web | {status} | {value} | {role} |")
    lines.append("")
    if manifest:
        lines += ["## 冻结测试来源", ""]
        for source in manifest.get("sources", []):
            lines.append(f"- {source['name']}：`{source['url']}`，固定提交 `{source['commit']}`，扫描根目录 `{', '.join(source['scan_roots'])}`。")
        lines += ["", "测试 fixture 只写入临时上游 checkout，不修改或提交上游源码；真值 JSON 不进入 Web Agent 工作区。每套工程在运行时会验证扫描范围的 C/C++ 源码不少于 10,000 行。", ""]
    if balanced or web_balanced:
        lines += ["## 平衡验收集（推荐验收口径）", "", "该集混合计算下标、循环、格式化/复制和分支指针形态；每类每工程仍固定 20 正例 + 20 对照。它是推荐的本地检测指标证据，但不是独立泛化分数。", ""]
        if web_balanced:
            lines += web_section("平衡验收集 Web", web_balanced, "web-sonnet45-balanced") + web_progress(web_balanced) + [""]
            lines += web_category_table(web_balanced) + [""] + web_project_table(web_balanced) + [""]
            lines += ["上述表为本轮原始统计值；阈值比较由前端或验收方按其配置执行。", ""]
        if balanced:
            lines += ["### 平衡验收集本地", ""] + direct_scope(balanced) + [""] + direct_table(balanced) + [""] + direct_project_table(balanced) + ["", "FreeRTOS 与 RT-Thread 的 `partial` 表示默认 Cppcheck 配置有工程解析诊断；注入文件统计有效，但不能据此声称完整覆盖全部上游工程。", ""]
    if independent or web_independent:
        lines += ["## 独立复杂压力集（能力边界）", "", "该集用于暴露跨辅助函数、结构体指针、指针算术和复杂字符串流的边界；不能替换、平均或择优合并平衡验收集。", ""]
        if web_independent:
            lines += web_section("独立压力集 Web", web_independent, "web-sonnet45-independent") + web_progress(web_independent) + [""]
            lines += web_category_table(web_independent) + [""] + web_project_table(web_independent) + [""]
            lines += ["独立压力集仅用于记录复杂形态下的能力边界；表中保留原始统计值，不作阈值判定。", ""]
        if independent:
            lines += ["### 独立压力集本地", ""] + direct_table(independent) + [""] + direct_project_table(independent) + [""]
    if juliet_runs:
        lines += juliet_section(juliet_runs, load(DATA / "juliet" / "manifest.json"))
    if any(web_hard.values()):
        lines += ["## 加难版测试集", "", HARD_INTRO, ""]
        for suite, (name, _) in HARD_SUITES.items():
            if web_hard[suite]:
                lines += web_section(f"{name} Web", web_hard[suite], f"web-sonnet45-{suite}") + web_progress(web_hard[suite]) + [""]
                lines += web_category_table(web_hard[suite]) + [""] + web_project_table(web_hard[suite]) + [""]
                lines += hard_variant_table(web_hard[suite]) + [""]
    lines += ["## 测试工程与 Ground Truth 交付状态", "", "当前可复现工程由三套固定版本的公开 RTOS 源码加运行时注入 fixture 组成：每套工程的三类各 20 个正例和 20 个同类安全对照，因此 Web 平衡验收集共 3×3=9 个任务、360 个函数。fixture、函数名、文件、行范围及预期标签由脚本固定生成，而不是由模型提供。", "", "- Ground truth 字段：`case_id`、`project`、`category`、`expected`、`function`、`file`、`line_start`、`line_end`、`defect_line`。Web 最新一轮位于 `artifacts/contract3/web-sonnet45-balanced/checkpoint.json` 的 `samples.*.rows[]`；该对象还含 `detected` 和 `matched_findings`，可供前端逐例展示。", "- 本地运行会另外生成 `artifacts/contract3/balanced-cppcheck/ground_truth.json` 与 `result.json`，字段口径相同。当前仓库按“仅保留最新 Web 输出”的清理策略没有保留它们；运行下面的本地命令即可重新生成。", "- 这套冻结来源和生成器可用于本项目复现，但**不是** NaturalCC 上游或验收方正式提供的最终验收工程/ground truth 包。若最终验收必须与外部指定样本逐字节一致，仍需要验收方提供或书面确认该工程、版本、注入位置和真值清单。", "", "## 测试链路与适用边界", "", "- 本地：通过产品 `security_analysis.cppcheck_scan` 适配层调用 Cppcheck；不执行目标程序。", "- Web：与合同 5 共用本机 Web Agent API、线程、Run、events 和持久化引擎；固定 OpenRouter/Sonnet 配置，无模型 fallback。", "- Web 运行按固定顺序串行执行，避免并发的限流、超时重试及账单归因干扰；未完成、超时、API 故障或无效 JSON 不计为安全、正确或任一统计桶。", "- 平衡验收集证明规定注入形态下的检测统计；独立压力集展示未覆盖形态的边界。两者均不能单独证明完整上游工程覆盖、所有路径安全或增量分析性能。", "- 多线程竞争/TSan、缓冲区溢出、内存泄漏、命令执行漏洞及预警准确率均属于指标 5，不在本指标 3 的统计接口或 Ground Truth 范围内。", "", "## 运行与续跑", "", "```bash", "# 推荐本地验收：同时生成本地 result.json 与 ground_truth.json", "uv run python scripts/benchmark_contract3_independent.py --profile balanced --workspace /tmp/naturalcc-contract3-independent-sources", "", "# Web 模式：先按 README 启动 agent_web_api.py，再在另一终端执行", "uv run python scripts/benchmark_contract3_web.py --suite balanced", "uv run python scripts/benchmark_contract3_web.py --suite independent", "uv run python scripts/benchmark_contract3_web.py --suite balanced-hard", "uv run python scripts/benchmark_contract3_web.py --suite independent-hard", "", "# 不发模型请求，仅从已有原始工件刷新本报告", "uv run python scripts/render_contract3_report.py", "```", "",
              "## 工件索引", "", "- `artifacts/contract3/manifest.json`：固定上游版本、目标类别、Cppcheck ID/CWE 白名单。", "- `artifacts/contract3/web-sonnet45-balanced/checkpoint.json`：当前最新 Web 评分、逐例 Ground Truth、预测和 `matched_findings`。", "- `artifacts/contract3/*-cppcheck/ground_truth.json`、`result.json`：本地模式运行时生成的真值和结果；按当前清理策略不随仓库保留。", "- `artifacts/contract3/web-sonnet45-balanced/runs/`：本机原始 Run events，体积较大且含运行历史，已由 `.gitignore` 排除；checkpoint 保留最近 Run ID 用于关联复核。"]
    return "\n".join(lines) + "\n"


def write_report() -> Path:
    REPORT.write_text(render(), encoding="utf-8")
    return REPORT


if __name__ == "__main__":
    print(write_report())
