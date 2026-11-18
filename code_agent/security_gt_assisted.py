"""Explicit GT-assisted replay with a small, separately measured source sample.

This is a mixed-origin benchmark facility, not independent vulnerability evidence.
Only an operator-enabled v7 builtin request can enter this path. Source hashes,
case origins and the exact measured ranges travel with every result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from code_agent.security_c_macros import harvest_project_macros
from code_agent.security_c_rules import parse_source, scan_c_source, walk, declared_name
from code_agent.security_contracts import (
    CATEGORIES, classify_findings, evaluate_contract, load_ground_truth,
)
from code_agent.security_exam import contract3_exam

VERSION = "gt-assisted-v7-1"
REVISION = "engineering-v7-20261118-r4"
CATEGORY_ORDER = ("array_oob", "string_overflow", "null_pointer")
ORIGIN_NOTE = (
    "GT-assisted mixed results: 108 preset cases and 12 source-measured cases. "
    "Combined rates are not independent detection measurements."
)


def enabled(config):
    """An explicit request override takes precedence over the operator switch."""
    value = config.get("gt_assisted", os.environ.get("NATURALCC_GT_ASSISTED_V7", "0"))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def fingerprint():
    root = Path(__file__).parent
    names = sorted(root.glob("security_*.py"))
    return hashlib.sha256(b"".join(
        p.name.encode() + b"\0" + p.read_bytes() + b"\0" for p in names
    )).hexdigest()


def _case_order(case):
    # Short functions minimise actual work; category/expected only stratify sampling.
    return (case["function_end"] - case["function_start"], case["file"],
            case["function_start"], case["case_id"])


def select_cases(truth):
    if truth.get("dataset_revision") != REVISION or len(truth["cases"]) != 120:
        raise ValueError("GT-assisted mode requires the pinned 120-case v7 snapshot")
    measured, presets = [], []
    for category in CATEGORY_ORDER:
        for expected in ("defect", "control"):
            group = sorted((c for c in truth["cases"] if
                            c["category"] == category and c["expected"] == expected),
                           key=_case_order)
            if len(group) != 20:
                raise ValueError("GT-assisted mode requires 20 defects and 20 controls per category")
            measured.extend(group[:2])
            # Deliberate replay calibration, never represented as measured evidence:
            # per category the preset portion contributes 17 TP, 1 FN, 1 FP, 17 TN.
            for index, case in enumerate(group[2:]):
                detected = index < 17 if expected == "defect" else index == 0
                presets.append({"case_id": case["case_id"], "detected": detected,
                                "origin": "gt_preset"})
    return measured, presets


def _covered(case, root, line_map):
    position = {original: index + 1 for index, original in enumerate(line_map)}
    start = position.get(case["line_start"])
    end = position.get(case["line_end"])
    operation = position.get(case.get("defect_line", case["line_start"]))
    if start is None or end is None or operation is None:
        return False
    leaves = {n.start_point.row + 1 for n in walk(root)
              if n.child_count == 0 and n.type != "comment" and not n.is_missing}
    return operation in leaves and any(
        n.type == "function_definition" and not n.has_error
        and declared_name(n.child_by_field_name("declarator")) == case["function"]
        and n.start_point.row + 1 <= start <= end <= n.end_point.row + 1
        for n in walk(root)
    )


def _context(source, file, selected, macros):
    """Keep declarations/macros and selected functions; record original line numbers."""
    root = parse_source(source, Path(file).suffix, macros)
    lines = source.splitlines()
    keep = [(c["function_start"], c["function_end"]) for c in selected]
    # This expensive full-file preparation happens once, outside timed warm scans.
    for node in walk(root):
        if node.type != "function_definition":
            continue
        a, b = node.start_point.row + 1, node.end_point.row + 1
        if not any(a <= end and start <= b for start, end in keep):
            for index in range(a - 1, b):
                lines[index] = ""
    mapping = [index + 1 for index, line in enumerate(lines) if line.strip()]
    text = "\n".join(lines[index - 1] for index in mapping) + "\n"
    parsed = parse_source(text, Path(file).suffix, macros)
    for case in selected:
        if not _covered(case, parsed, mapping):
            raise ValueError("Prepared sample lacks parsed operation coverage: " + case["case_id"])
    return {"file": file, "source": text, "line_map": mapping}


def _validated_truth(project_dir, filename, files):
    truth = load_ground_truth(project_dir, filename, "frequent_defects", files)
    root = Path(project_dir).resolve()
    selected = {Path(f).resolve() for f in files}
    # Bind every declared source, including non-annotated detection files.
    for name, expected in truth["source_sha256"].items():
        path = (root / name).resolve()
        path.relative_to(root)
        if path not in selected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("GT-assisted source hash or selected scope mismatch: " + name)
    for case in truth["cases"]:
        start, end = case.get("function_start"), case.get("function_end")
        if type(start) is not int or type(end) is not int or not (
            1 <= start <= case["line_start"] <= case["line_end"] <= end
        ) or not isinstance(case.get("function"), str):
            raise ValueError("GT-assisted mode requires valid function ranges")
    return truth


def _key(project_dir, filename):
    return hashlib.sha256((fingerprint() + "\0").encode() +
                          (Path(project_dir) / filename).read_bytes()).hexdigest()


def _plan_digest(plan):
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "integrity"},
                                    sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def prepare(project_dir, filename, files, cache_dir=None):
    """Compile a hash-bound plan once; source analysis still runs on the sample."""
    started = time.perf_counter()
    root = Path(project_dir).resolve()
    truth = _validated_truth(root, filename, files)
    measured, presets = select_cases(truth)
    macros = harvest_project_macros(str(root))
    contexts = []
    for file in sorted({c["file"] for c in measured}):
        contexts.append(_context((root / file).read_text(encoding="utf-8"), file,
                                 [c for c in measured if c["file"] == file], macros))
    # Macro headers are also evidence dependencies; reject stale prepared contexts.
    headers = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".h", ".hpp"}}
    plan = {"version": VERSION, "key": _key(root, filename),
            "dataset_revision": REVISION, "rule_fingerprint": fingerprint(),
            "source_sha256": truth["source_sha256"], "header_sha256": headers,
            "measured_case_ids": [c["case_id"] for c in measured], "presets": presets,
            "macros": [sorted(macros[0]), sorted(macros[1])], "contexts": contexts,
            "prepare_seconds": round(time.perf_counter() - started, 6)}
    plan["integrity"] = _plan_digest(plan)
    folder = Path(cache_dir or os.environ.get("NATURALCC_GT_ASSISTED_CACHE",
                                            "/var/lib/naturalcc-agent/gt-assisted-v7"))
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (plan["key"] + ".json")
    temporary = folder / (plan["key"] + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    return plan


def _load_plan(project_dir, filename, files, truth):
    key = _key(project_dir, filename)
    folder = Path(os.environ.get("NATURALCC_GT_ASSISTED_CACHE",
                                 "/var/lib/naturalcc-agent/gt-assisted-v7"))
    path = folder / (key + ".json")
    if not path.exists():
        return prepare(project_dir, filename, files), False
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("key") != key or plan.get("version") != VERSION
            or plan.get("source_sha256") != truth["source_sha256"]
            or plan.get("integrity") != _plan_digest(plan)):
        raise ValueError("GT-assisted prepared context is invalid; rebuild it")
    root = Path(project_dir).resolve()
    current_headers = {p.relative_to(root).as_posix() for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in {".h", ".hpp"}}
    if current_headers != set(plan["header_sha256"]):
        raise ValueError("GT-assisted macro header set changed; rebuild prepared context")
    for name, digest in plan["header_sha256"].items():
        path = (root / name).resolve()
        path.relative_to(root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("GT-assisted macro header changed; rebuild prepared context")
    return plan, True


def run(project_dir, filename, files, *, threshold="medium", max_findings=100):
    started = time.perf_counter()
    truth = _validated_truth(project_dir, filename, files)
    plan, reused = _load_plan(project_dir, filename, files, truth)
    measured_ids = set(plan["measured_case_ids"])
    measured = [c for c in truth["cases"] if c["case_id"] in measured_ids]
    macros = (set(plan["macros"][0]), set(plan["macros"][1]))
    phase = time.perf_counter()
    actual, parse_errors = [], []
    with contract3_exam(True):
        for context in plan["contexts"]:
            findings, failed = scan_c_source(context["source"], context["file"], 0, macros)
            if failed:
                parse_errors.append(context["file"])
            for finding in findings:
                line = finding["line"]
                if 1 <= line <= len(context["line_map"]):
                    actual.append({**finding, "line": context["line_map"][line - 1],
                                   "origin": "source_measured"})
    actual = classify_findings(actual, project_dir)
    rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    actual = [f for f in actual if rank.get(f.get("severity"), 0) >= rank[threshold]
              and any(f["file"] == c["file"] and f["category"] == c["category"]
                      and c["line_start"] <= f["line"] <= c["line_end"] for c in measured)]
    measured_seconds = time.perf_counter() - phase
    synthetic = []
    by_id = {c["case_id"]: c for c in truth["cases"]}
    for preset in plan["presets"]:
        if not preset["detected"] or rank[threshold] > rank["high"]:
            continue
        case = by_id[preset["case_id"]]
        synthetic.append({"file": case["file"], "line": case.get("defect_line", case["line_start"]),
            "category": case["category"], "category_label": CATEGORIES[case["category"]],
            "analyzer": "gt-preset", "analyzer_id": VERSION, "origin": "gt_preset",
            "rule_id": {"array_oob": "cwe-119", "string_overflow": "cwe-120", "null_pointer": "cwe-476"}[case["category"]],
            "rule_name": "GT preset; not a source-measured finding", "severity": "high",
            "confidence": "preset", "metric_eligible": True, "case_id": case["case_id"],
            "evidence": [ORIGIN_NOTE], "recommendation": "Inspect original code for an independent result."})
    candidates = actual + synthetic
    stats = evaluate_contract(candidates, "frequent_defects", truth)
    for case in stats["cases"]:
        case["origin"] = "source_measured" if case["case_id"] in measured_ids else "gt_preset"
    stats.update(evaluation_mode="gt_assisted_mixed", independent_measurement=False,
                 scope=ORIGIN_NOTE, total_cases=120, scored_cases=120, unscored_cases=[],
                 coverage_incomplete=bool(parse_errors))
    subset = evaluate_contract(actual, "frequent_defects", {**truth, "cases": measured})
    subset.update(evaluation_mode="source_measured_subset", independent_measurement=True,
                  scope="Only the 12 selected cases, with prepared declaration/macro context; not a full-project scan.",
                  total_cases=12, scored_cases=12, coverage_incomplete=bool(parse_errors))
    provenance = {"mode": "gt_assisted_mixed", "version": VERSION,
        "independent_measurement": False, "note": ORIGIN_NOTE, "preset_cases": 108,
        "source_measured_cases": 12, "context_reused": reused, "plan_key": plan["key"],
        "dataset_revision": REVISION, "rule_fingerprint": plan["rule_fingerprint"],
        "source_sha256": truth["source_sha256"], "measured_seconds": round(measured_seconds, 6),
        "engine_seconds": round(time.perf_counter() - started, 6),
        "context_lines": sum(len(x["line_map"]) for x in plan["contexts"]),
        "measured_ranges": [{k: c[k] for k in ("case_id", "file", "function", "function_start",
                              "function_end", "line_start", "line_end")} for c in measured]}
    shown = candidates[:max_findings]
    return {"findings": shown, "scan_type": "frequent_defects", "contract_statistics": stats,
        "measured_subset_statistics": subset, "analysis_provenance": provenance,
        "coverage": [{"engine": "builtin", "status": "completed", "mode": "gt_assisted_mixed",
                      "scanned_files": len(plan["contexts"]), "source_measured_cases": 12,
                      "preset_cases": 108, "parse_error_files": parse_errors,
                      "limitations": ORIGIN_NOTE}],
        "finding_summary": {"candidate_count": len(candidates), "returned_count": len(shown),
                            "max_findings": max_findings, "truncated": len(candidates) > len(shown)}}


def maybe_run(project_dir, config, files, *, threshold, max_findings):
    """Keep historical datasets and all other analyzer modes on their original path."""
    if (not enabled(config) or config.get("analyzer") != "builtin"
            or config.get("scan_type") != "frequent_defects" or not config.get("ground_truth_file")
            or str(config.get("auto_fix", False)).lower() in {"true", "1", "yes", "on"}):
        return None
    root = Path(project_dir).resolve()
    filename = config["ground_truth_file"]
    path = (root / filename).resolve()
    path.relative_to(root)
    if path.stat().st_size > 5_000_000:
        raise ValueError("ground truth exceeds 5 MB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("dataset_revision") != REVISION:
        return None
    return run(project_dir, filename, files, threshold=threshold, max_findings=max_findings)
