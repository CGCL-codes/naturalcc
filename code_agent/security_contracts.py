"""Versioned finding taxonomy and case-level evaluation, after detection only."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from code_agent.security_exam import switch_off
from code_agent.security_c_rules import call_parts, declared_name, parse_source, text, walk

TAXONOMY_VERSION = "contract-categories-v2"
CATEGORIES = {
    "array_oob": "数组越界", "string_overflow": "字符串溢出", "null_pointer": "空指针调用",
    "buffer_overflow": "缓冲区溢出", "data_race": "多线程竞争",
    "memory_leak": "内存泄漏", "command_execution": "命令执行漏洞",
}
SCAN_CATEGORIES = {
    "frequent_defects": ["array_oob", "string_overflow", "null_pointer"],
    "high_risk": ["buffer_overflow", "data_race", "memory_leak", "command_execution"],
}
FORMULAS = {
    "detection_rate": "TP / (TP + FN)",
    "warning_accuracy": "TP / (TP + FP)",
    "false_positive_rate": "FP / (FP + TN)",
    "false_discovery_rate": "FP / (TP + FP)",
    "accuracy": "(TP + TN) / (TP + FN + FP + TN)",
}
STRING_CALLS = {"strcpy", "strncpy", "strcat", "strncat", "sprintf", "snprintf", "gets", "scanf", "fscanf", "sscanf"}
BUFFER_CALLS = {"memcpy", "memmove", "memset", "read", "recv", "fread"}


def _pointer_offset_accesses(source, suffix):
    """Locations needing pointer-to-array normalization in the teaching classifier.

    This is a syntactic capability boundary, not a test-case allow/deny list.
    No GT, function names, capacities or vulnerability labels are consulted.
    Both defective accesses and their safe neighbours take the same path.
    """
    ranges = []
    for function in walk(parse_source(source, suffix)):
        if function.type != "function_definition":
            continue
        nodes = list(walk(function))
        arrays = {declared_name(n) for n in nodes if n.type == "array_declarator"}
        pointers = set()
        for node in nodes:
            if node.type != "init_declarator":
                continue
            target, value = node.child_by_field_name("declarator"), node.child_by_field_name("value")
            if target is None or target.type != "pointer_declarator" or value is None or value.type != "binary_expression":
                continue
            if text(value.child_by_field_name("operator")) in {"+", "-"} and text(value.child_by_field_name("left")) in arrays:
                pointers.add(declared_name(target))
        for node in nodes:
            argument = node.child_by_field_name("argument")
            if argument is None or text(argument) not in pointers:
                continue
            if node.type == "subscript_expression" or (node.type == "pointer_expression" and text(node.child_by_field_name("operator")) == "*"):
                ranges.append((node.start_point.row + 1, node.end_point.row + 1))
    return ranges


# [maint-note:7c991e1c6ac3]
# 业务边界：先识别检测引擎及诊断 ID，再按源操作区分缺陷类别。
# 相同 CWE 编号可能覆盖多种操作；只看编号不能证明具体缺陷。
# 字符串字面量参与的内存复制属于字符串容量检查的证据来源。
# 关闭教学开关 A 会留下归类缺口，供学生实现并比较参考答案。
# 提示性规则保留为人工复核线索，不自动取得 metric_eligible。
# 分类阶段不读取 GT；真值只能在后续评分阶段使用。
# 维护导航：classify_findings；下列为本函数源码中的接口与路径索引。
# 输入 findings：类型由调用方与函数体约束。
# 输入 project_dir：类型由调用方与函数体约束。
# 分支条件：engine == 'c-source'。
# 分支条件：engine == 'tsan-import'。
# 分支条件：engine in {'cppcheck', 'clang'}。
# 分支条件：engine == 'clang'。
# 分支条件：identifier == 'unix.Malloc' and finding.get('diagnostic_type') == 'Memory leak'。
# 分支条件：identifier in {'arrayIndexOutOfBounds', 'arrayIndexOutOfBoundsCond', 'negativeIndex'}。
# 分支条件：identifier in {'nullPointer', 'nullPointerRedundantCheck', 'nullPointerDefaultArg'}。
# 分支条件：identifier in {'memleak', 'memleakOnRealloc'}。
# 分支条件：identifier in {'bufferAccessOutOfBounds', 'bufferAccessOutOfBoundsCond', 'terminateStrncpy'}。
# 分支条件：file not in operations。
# 分支条件：n.type == 'call_expression'。
# 分支条件：string != buffer。
# 分支条件：finding.get('rule_id') in {'cwe-120', 'cwe-787', 'cwe-362'}。
# 遍历边界：finding ← findings。
# 遍历边界：n ← walk(parse_source(source, path.suffix.lower()))。
# 返回路径：findings。
# 写入目标：operations[file]。
# 直接调用：finding.get。
# 直接调用：{'shellStringFlow': 'command_execution', 'sharedGlobalRace': 'data_race', 'capacityStringWrite': 'string_overflow', 'nullArgumentDeref': 'null_pointer', 'nullInitDeref': 'null_pointer', 'shellCommandBuild': 'command_execution', 'lockScopeEs。
# 直接调用：bool。
# 直接调用：{'core.NullDereference': 'nullPointer', 'alpha.security.ArrayBoundV2': 'arrayIndexOutOfBounds', 'alpha.security.ArrayBound': 'arrayIndexOutOfBounds', 'alpha.unix.cstring.OutOfBounds': 'bufferAccessOutOfBounds'}.get。
# 直接调用：(Path(project_dir) / file).resolve。
# 直接调用：Path。
# 直接调用：path.relative_to。
# 直接调用：Path(project_dir).resolve。
# 直接调用：path.read_text。
# 直接调用：walk。
# 直接调用：parse_source。
# 直接调用：path.suffix.lower。
# 直接调用：call_parts。
# 直接调用：any。
# 直接调用：operations[file].append。
# 直接调用：switch_off。
# 直接调用：finding.update。
# 直接调用：CATEGORIES.get。
# 集合派生：a 来自 arguments。
# 集合派生：(start, end, call, literal) 来自 operations[file]；过滤 start <= finding['line'] <= end。
# 集合派生：(call, literal) 来自 at_line。
# 表达式分流：string；真值与假值分支分别计算，不同时执行。
# 表达式分流：finding['rule_id'] == 'cwe-362'；真值与假值分支分别计算，不同时执行。
# [/maint-note:7c991e1c6ac3]
def classify_findings(findings, project_dir):
    operations = {}
    pointer_accesses = {}
    for finding in findings:
        category, eligible, basis = None, False, "Unmapped diagnostic; CWE alone is insufficient"
        identifier = finding.get("analyzer_id", "")
        engine = finding.get("analyzer")
        if engine == "c-source":
            category = {"shellStringFlow": "command_execution", "sharedGlobalRace": "data_race",
                        "guardedArrayAccess": "array_oob", "capacityStringWrite": "string_overflow", "nullArgumentDeref": "null_pointer",
                        "nullInitDeref": "null_pointer",
                        "nativeArrayExtent": "array_oob", "nativeStringExtent": "string_overflow",
                        "nativeNullFlow": "null_pointer",
                        # RTOS-aware rules (security_rtos_rules), one per high-risk category.
                        "shellCommandBuild": "command_execution", "lockScopeEscape": "data_race",
                        "ownershipLeak": "memory_leak", "capacityWriteOverflow": "buffer_overflow"}.get(identifier)
            eligible, basis = bool(category), "Source flow/conflicting-access/capacity evidence"
        elif engine == "tsan-import":
            category, eligible, basis = "data_race", True, "Imported TSan race; log provenance is user supplied"
        elif engine in {"cppcheck", "clang"}:
            if engine == "clang":
                identifier = {"core.NullDereference": "nullPointer", "alpha.security.ArrayBoundV2": "arrayIndexOutOfBounds", "alpha.security.ArrayBound": "arrayIndexOutOfBounds",
                    "alpha.unix.cstring.OutOfBounds": "bufferAccessOutOfBounds"}.get(identifier, identifier)
                if identifier == "unix.Malloc" and finding.get("diagnostic_type") == "Memory leak":
                    identifier = "memleak"
            if identifier in {"arrayIndexOutOfBounds", "arrayIndexOutOfBoundsCond", "negativeIndex"}:
                category = "array_oob"
            elif identifier in {"nullPointer", "nullPointerRedundantCheck", "nullPointerDefaultArg"}:
                category = "null_pointer"
            elif identifier in {"memleak", "memleakOnRealloc"}:
                category = "memory_leak"
            elif identifier in {"bufferAccessOutOfBounds", "bufferAccessOutOfBoundsCond", "terminateStrncpy"}:
                file = finding["file"]
                if file not in operations:
                    path = (Path(project_dir) / file).resolve()
                    path.relative_to(Path(project_dir).resolve())
                    source = path.read_text(encoding="utf-8", errors="replace")
                    operations[file] = []
                    for n in walk(parse_source(source, path.suffix.lower())):
                        if n.type == "call_expression":
                            name, arguments = call_parts(n)
                            literal = any(a.type in {"string_literal", "concatenated_string"} for a in arguments)
                            operations[file].append((n.start_point.row + 1, n.end_point.row + 1, name, literal))
                at_line = [(call, literal) for start, end, call, literal in operations[file] if start <= finding["line"] <= end]
                # A raw-memory copy whose source is a string literal moves string data into a char buffer.
                # Contract-3 switch A (security_exam): memcpy/memmove of a string literal counts as string overflow.
                string = any(call in STRING_CALLS or (call in BUFFER_CALLS and literal and not switch_off("A"))
                             for call, literal in at_line)
                buffer = any(call in BUFFER_CALLS and not literal for call, literal in at_line)
                if string != buffer:
                    category = "string_overflow" if string else "buffer_overflow"
            eligible, basis = bool(category), "Analyzer diagnostic ID/type and operation at diagnostic location"
        elif finding.get("rule_id") in {"cwe-120", "cwe-787", "cwe-362"}:
            category = "data_race" if finding["rule_id"] == "cwe-362" else "string_overflow"
            basis = "Review hint only; API presence does not establish a defect"
        # G leaves pointer-derived array normalization as a student exercise.
        # Preserve the raw engine diagnostic; only the incomplete teaching
        # taxonomy omits this category. reset and all non-Contract-3 scans
        # retain the ordinary classification. Scoring itself is unchanged.
        if category == "array_oob" and switch_off("G"):
            file = finding["file"]
            if file not in pointer_accesses:
                path = (Path(project_dir) / file).resolve()
                path.relative_to(Path(project_dir).resolve())
                pointer_accesses[file] = _pointer_offset_accesses(path.read_text(encoding="utf-8", errors="replace"), path.suffix.lower())
            if any(start <= finding["line"] <= end for start, end in pointer_accesses[file]):
                category, eligible = None, False
                basis = "Teaching gap G: pointer-derived array access normalization is disabled"
        finding.update(category=category, category_label=CATEGORIES.get(category, "其他诊断"),
                       metric_eligible=eligible, classification_basis=basis,
                       taxonomy_version=TAXONOMY_VERSION)
    return findings


# [maint-note:5917987ece31]
# 业务边界：真值文件不是可信输入，路径、结构与源文件版本都要验证。
# 仅接受工作区相对路径，解析符号链接后再次约束在工作区内。
# 每个 case_id 唯一，同文件同类别的函数行范围不能重叠。
# SHA-256 将标签绑定到具体源码，改动测试文件后旧标签不能继续计分。
# 只有实际扫描过的文件可以进入分母，避免把未扫描文件当成漏报或安全。
# 文件体积上限约束加载成本；类别必须属于本次 scan_type。
# 维护导航：load_ground_truth；下列为本函数源码中的接口与路径索引。
# 输入 project_dir：类型由调用方与函数体约束。
# 输入 filename：类型由调用方与函数体约束。
# 输入 scan_type：类型由调用方与函数体约束。
# 输入 scanned_files：类型由调用方与函数体约束。
# 分支条件：Path(filename).is_absolute()。
# 分支条件：path.stat().st_size > 5000000。
# 分支条件：not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('scan_type') != scan_type。
# 分支条件：not isinstance(cases, list) or not cases or (not isinstance(hashes, dict))。
# 分支条件：not isinstance(case, dict)。
# 分支条件：not isinstance(case_id, str) or not case_id or case_id in ids。
# 分支条件：category not in SCAN_CATEGORIES[scan_type] or case.get('expected') not in {'defect', 'control'}。
# 分支条件：not isinstance(file, str) or not file or Path(file).is_absolute()。
# 分支条件：relative != file。
# 分支条件：source not in selected。
# 分支条件：file not in checked。
# 分支条件：hashes.get(file) != hashlib.sha256(content).hexdigest()。
# 分支条件：type(start) is not int or type(end) is not int or (not 1 <= start <= end <= checked[file])。
# 分支条件：any((start <= b and a <= end for (a, b) in ranges.get(key, [])))。
# 遍历边界：case ← cases。
# 返回路径：data。
# 显式抛出：ValueError('ground_truth_file must be workspace-relative')。
# 显式抛出：ValueError('ground truth exceeds 5 MB')。
# 显式抛出：ValueError('ground truth requires schema_version=1 and matching scan_type')。
# 显式抛出：ValueError('ground truth requires nonempty cases and source_sha256')。
# 显式抛出：ValueError('each ground truth case must be an object')。
# 显式抛出：ValueError('ground truth case_id must be nonempty and unique')。
# 显式抛出：ValueError(f'invalid category or expected value for {case_id}')。
# 显式抛出：ValueError('ground truth file must be workspace-relative')。
# 显式抛出：ValueError('ground truth file must be a normalized relative path')。
# 显式抛出：ValueError(f'ground truth source was not scanned: {file}')。
# 显式抛出：ValueError(f'ground truth source hash mismatch: {file}')。
# 显式抛出：ValueError(f'invalid ground truth line range for {case_id}')。
# 显式抛出：ValueError(f'overlapping ground truth ranges for {file}/{category}')。
# 写入目标：checked[file]。
# 直接调用：Path(project_dir).resolve。
# 直接调用：Path。
# 直接调用：Path(filename).is_absolute。
# 直接调用：ValueError。
# 直接调用：(root / filename).resolve。
# 直接调用：path.relative_to。
# 直接调用：path.stat。
# 直接调用：json.loads。
# 直接调用：path.read_text。
# 直接调用：isinstance。
# 直接调用：data.get。
# 直接调用：Path(p).resolve。
# 直接调用：set。
# 直接调用：case.get。
# 直接调用：ids.add。
# 直接调用：Path(file).is_absolute。
# 直接调用：(root / file).resolve。
# 直接调用：source.relative_to(root).as_posix。
# 直接调用：source.relative_to。
# 直接调用：source.read_bytes。
# 直接调用：hashes.get。
# 直接调用：hashlib.sha256(content).hexdigest。
# 直接调用：hashlib.sha256。
# 直接调用：len。
# 直接调用：content.splitlines。
# 直接调用：type。
# 直接调用：any。
# 直接调用：ranges.get。
# 直接调用：ranges.setdefault(key, []).append。
# 直接调用：ranges.setdefault。
# 集合派生：p 来自 scanned_files。
# 集合派生：(a, b) 来自 ranges.get(key, [])。
# [/maint-note:5917987ece31]
def load_ground_truth(project_dir, filename, scan_type, scanned_files):
    root = Path(project_dir).resolve()
    if Path(filename).is_absolute():
        raise ValueError("ground_truth_file must be workspace-relative")
    path = (root / filename).resolve()
    path.relative_to(root)
    if path.stat().st_size > 5_000_000:
        raise ValueError("ground truth exceeds 5 MB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("scan_type") != scan_type:
        raise ValueError("ground truth requires schema_version=1 and matching scan_type")
    cases, hashes = data.get("cases"), data.get("source_sha256")
    if not isinstance(cases, list) or not cases or not isinstance(hashes, dict):
        raise ValueError("ground truth requires nonempty cases and source_sha256")
    selected = {Path(p).resolve() for p in scanned_files}
    checked, ids, ranges = {}, set(), {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each ground truth case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("ground truth case_id must be nonempty and unique")
        ids.add(case_id)
        category, file = case.get("category"), case.get("file")
        if category not in SCAN_CATEGORIES[scan_type] or case.get("expected") not in {"defect", "control"}:
            raise ValueError(f"invalid category or expected value for {case_id}")
        if not isinstance(file, str) or not file or Path(file).is_absolute():
            raise ValueError("ground truth file must be workspace-relative")
        source = (root / file).resolve()
        relative = source.relative_to(root).as_posix()
        if relative != file:
            raise ValueError("ground truth file must be a normalized relative path")
        if source not in selected:
            raise ValueError(f"ground truth source was not scanned: {file}")
        if file not in checked:
            content = source.read_bytes()
            if hashes.get(file) != hashlib.sha256(content).hexdigest():
                raise ValueError(f"ground truth source hash mismatch: {file}")
            checked[file] = len(content.splitlines())
        start, end = case.get("line_start"), case.get("line_end")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= checked[file]:
            raise ValueError(f"invalid ground truth line range for {case_id}")
        key = (file, category)
        if any(start <= b and a <= end for a, b in ranges.get(key, [])):
            raise ValueError(f"overlapping ground truth ranges for {file}/{category}")
        ranges.setdefault(key, []).append((start, end))
    return data


# [maint-note:1828e23e4b0b]
# 统计口径：检出率的分母是正例 TP+FN，安全对照不进入该分母。
# 误报率的分母是安全对照 FP+TN，与误发现率 FP/(TP+FP) 不同。
# 分母为零返回 None；不能把没有样本解释为零错误或满分。
# 总体与各类别分别聚合，不能用总体超过阈值代替每类超过阈值。
# 维护导航：_metric；下列为本函数源码中的接口与路径索引。
# 输入 rows：类型由调用方与函数体约束。
# 返回路径：a / b if b else None。
# 返回路径：{'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'detection_rate': ratio(tp, tp + fn), 'warning_accuracy': ratio(tp, tp + fp), 'false_positive_rate': ratio(fp, fp + tn), 'false_discovery_rate': ratio(fp, tp + fp), 'accuracy': ratio(tp + tn, tp + fn。
# 直接调用：Counter。
# 直接调用：ratio。
# 集合派生：row 来自 rows。
# 集合派生：k 来自 ('tp', 'fn', 'fp', 'tn')。
# [/maint-note:1828e23e4b0b]
def _metric(rows):
    counts = Counter(row["outcome"] for row in rows)
    tp, fn, fp, tn = (counts[k] for k in ("tp", "fn", "fp", "tn"))
    # [maint-note:1e5138cb44a1]
    # 维护导航：ratio；下列为本函数源码中的接口与路径索引。
    # 输入 a：类型由调用方与函数体约束。
    # 输入 b：类型由调用方与函数体约束。
    # 返回路径：a / b if b else None。
    # 表达式分流：b；真值与假值分支分别计算，不同时执行。
    # [/maint-note:1e5138cb44a1]
    def ratio(a, b):
        return a / b if b else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "detection_rate": ratio(tp, tp + fn), "warning_accuracy": ratio(tp, tp + fp),
        "false_positive_rate": ratio(fp, fp + tn), "false_discovery_rate": ratio(fp, tp + fp),
        "accuracy": ratio(tp + tn, tp + fn + fp + tn)}


# [maint-note:395a16c88545]
# 评分边界：只使用类别匹配且有可计分证据的 Finding。
# 同一个函数被多个引擎命中仍只算一个 TP，不按告警条数累计。
# Finding 必须同时命中文件、类别和闭区间行范围。
# 未标注区域的告警单独报告，不据此推断整个工程的准确率。
# 没有 GT 或扫描不可用时返回未评估，不能伪造 TP/FN/FP/TN。
# 保留逐例 matched_findings，便于从汇总数字回溯到源位置。
# 维护导航：evaluate_contract；下列为本函数源码中的接口与路径索引。
# 输入 findings：类型由调用方与函数体约束。
# 输入 scan_type：类型由调用方与函数体约束。
# 输入 ground_truth：类型由调用方与函数体约束；缺省表达式 None。
# 输入 unavailable_reason：类型由调用方与函数体约束；缺省表达式 None。
# 分支条件：ground_truth is None or unavailable_reason。
# 遍历边界：case ← ground_truth['cases']。
# 返回路径：result。
# 写入目标：result['reason']。
# 直接调用：enumerate。
# 直接调用：sum。
# 直接调用：len。
# 直接调用：set。
# 直接调用：bool。
# 直接调用：rows.append。
# 直接调用：findings[i].get。
# 直接调用：used.update。
# 直接调用：result.update。
# 直接调用：_metric。
# 集合派生：(i, f) 来自 enumerate(findings)；过滤 f['metric_eligible'] and f['category'] in categories。
# 集合派生：f 来自 findings。
# 集合派生：(_, f) 来自 eligible。
# 集合派生：c 来自 categories。
# 集合派生：(i, f) 来自 eligible；过滤 f['file'] == case['file'] and f['category'] == case['category'] and (case['line_start'] <= f['line'] <= case['line_end'])。
# 表达式分流：case['expected'] == 'defect'；真值与假值分支分别计算，不同时执行。
# 表达式分流：detected；真值与假值分支分别计算，不同时执行。
# 集合派生：k 来自 ('file', 'line', 'analyzer', 'analyzer_id', 'rule_id')。
# 集合派生：i 来自 matches。
# 集合派生：r 来自 rows；过滤 r['category'] == c。
# [/maint-note:395a16c88545]
def evaluate_contract(findings, scan_type, ground_truth=None, *, unavailable_reason=None):
    categories = SCAN_CATEGORIES[scan_type]
    eligible = [(i, f) for i, f in enumerate(findings) if f["metric_eligible"] and f["category"] in categories]
    result = {"status": "not_evaluated", "taxonomy_version": TAXONOMY_VERSION,
        "statistical_unit": "ground_truth_case", "rate_scale": "0..1; undefined denominator -> null",
        "formulas": FORMULAS, "target_categories": categories,
        "target_candidate_count": sum(f["category"] in categories for f in findings),
        "category_counts": {c: {"label": CATEGORIES[c], "candidates": sum(f["category"] == c for f in findings),
            "eligible_findings": sum(f["category"] == c for _, f in eligible)} for c in categories},
        "eligible_finding_count": len(eligible), "excluded_finding_count": len(findings) - len(eligible)}
    if ground_truth is None or unavailable_reason:
        result["reason"] = unavailable_reason or (
            "A versioned ground truth file is required; finding-matching rules use file, category and line range.")
        return result
    rows, used = [], set()
    for case in ground_truth["cases"]:
        matches = [i for i, f in eligible if f["file"] == case["file"] and f["category"] == case["category"]
                   and case["line_start"] <= f["line"] <= case["line_end"]]
        detected = bool(matches)
        outcome = ("tp" if detected else "fn") if case["expected"] == "defect" else ("fp" if detected else "tn")
        rows.append({**case, "detected": detected, "outcome": outcome,
                     "matched_findings": [{k: findings[i].get(k) for k in ("file", "line", "analyzer", "analyzer_id", "rule_id")} for i in matches]})
        used.update(matches)
    result.update(status="evaluated", overall=_metric(rows),
        by_category={c: {"label": CATEGORIES[c], **_metric([r for r in rows if r["category"] == c])} for c in categories},
        cases=rows, unscored_target_finding_count=len(eligible) - len(used),
        scope="Only annotated cases are scored. Unannotated target findings are reported separately; this is not whole-project accuracy.")
    return result
