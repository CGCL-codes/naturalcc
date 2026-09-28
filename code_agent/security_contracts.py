"""Versioned finding taxonomy and case-level evaluation, after detection only."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from code_agent.security_c_rules import call_parts, parse_source, walk

TAXONOMY_VERSION = "contract-categories-v1"
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


def classify_findings(findings, project_dir):
    operations = {}
    for finding in findings:
        category, eligible, basis = None, False, "Unmapped diagnostic; CWE alone is insufficient"
        identifier = finding.get("analyzer_id", "")
        engine = finding.get("analyzer")
        if engine == "c-source":
            category = {"shellStringFlow": "command_execution", "sharedGlobalRace": "data_race"}.get(identifier)
            eligible, basis = bool(category), "Source flow/conflicting-access evidence"
        elif engine == "tsan-import":
            category, eligible, basis = "data_race", True, "Imported TSan race; log provenance is user supplied"
        elif engine in {"cppcheck", "clang"}:
            if engine == "clang":
                identifier = {"core.NullDereference": "nullPointer", "alpha.security.ArrayBoundV2": "arrayIndexOutOfBounds",
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
                    operations[file] = [(n.start_point.row + 1, n.end_point.row + 1, call_parts(n)[0])
                        for n in walk(parse_source(source, path.suffix.lower())) if n.type == "call_expression"]
                calls = {call for start, end, call in operations[file] if start <= finding["line"] <= end}
                string, buffer = bool(calls & STRING_CALLS), bool(calls & BUFFER_CALLS)
                if string != buffer:
                    category = "string_overflow" if string else "buffer_overflow"
            eligible, basis = bool(category), "Analyzer diagnostic ID/type and operation at diagnostic location"
        elif finding.get("rule_id") in {"cwe-120", "cwe-787", "cwe-362"}:
            category = "data_race" if finding["rule_id"] == "cwe-362" else "string_overflow"
            basis = "Review hint only; API presence does not establish a defect"
        finding.update(category=category, category_label=CATEGORIES.get(category, "其他诊断"),
                       metric_eligible=eligible, classification_basis=basis,
                       taxonomy_version=TAXONOMY_VERSION)
    return findings


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


def _metric(rows):
    counts = Counter(row["outcome"] for row in rows)
    tp, fn, fp, tn = (counts[k] for k in ("tp", "fn", "fp", "tn"))
    def ratio(a, b):
        return a / b if b else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "detection_rate": ratio(tp, tp + fn), "warning_accuracy": ratio(tp, tp + fp),
        "false_positive_rate": ratio(fp, fp + tn), "false_discovery_rate": ratio(fp, tp + fp),
        "accuracy": ratio(tp + tn, tp + fn + fp + tn)}


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
