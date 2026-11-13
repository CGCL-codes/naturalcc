"""Build the Contract-5 whole-project dataset (RT-Thread v5.2.1 + injected high-risk defects).

The dataset is one OS-level C project (kernel, shell, file system, driver
framework, libc glue) copied from a pinned upstream commit, with 10 artificial
defects per contract category injected into existing functions:

    buffer_overflow (CWE-787/120)   data_race (CWE-362)
    memory_leak (CWE-401)           command_execution (CWE-78)

Every injected defect has a safe look-alike control, and several target files
stay untouched ("clean").  Ground truth lives in ``dataset/ground_truth.json``
next to the project, never inside the Agent workspace.

Inputs (committed):
    artifacts/contract5/rtthread/dataset_spec.json      upstream, copied paths, clean files
    artifacts/contract5/rtthread/injections/<cat>.json  edit anchors + labels per category
Outputs (committed so the client receives the full project without cloning):
    artifacts/contract5/rtthread/dataset/project/       the injected project (Agent workspace)
    artifacts/contract5/rtthread/dataset/ground_truth.json
    artifacts/contract5/rtthread/dataset/manifest.json  hashes for tamper checks

Usage:
    uv run python scripts/build_contract5_dataset.py --check            # validate anchors only
    uv run python scripts/build_contract5_dataset.py                    # clone/verify upstream, build
    uv run python scripts/build_contract5_dataset.py --archive          # also write contract5.zip
    uv run python scripts/build_contract5_dataset.py --verify           # compare committed output with manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "contract5" / "rtthread"
SPEC = DATA / "dataset_spec.json"
INJECTIONS = DATA / "injections"
OVERLAY = DATA / "overlay"
DATASET = DATA / "dataset"
PROJECT = DATASET / "project"
GROUND_TRUTH = DATASET / "ground_truth.json"
MANIFEST = DATASET / "manifest.json"
if (ROOT / "project").is_dir() and (ROOT / "ground_truth.json").is_file():
    # Running from an unpacked contract5.zip: everything sits next to scripts/.
    DATA = DATASET = ROOT
    SPEC, INJECTIONS, OVERLAY = ROOT / "dataset_spec.json", ROOT / "injections", ROOT / "overlay"
    PROJECT, GROUND_TRUTH, MANIFEST = ROOT / "project", ROOT / "ground_truth.json", ROOT / "manifest.json"
DEFAULT_UPSTREAM_ROOT = Path("/tmp/naturalcc-contract5-upstream")

CATEGORIES = {
    "buffer_overflow": {"name": "缓冲区溢出", "cwe": 787, "cwes": [787, 120, 121, 122, 119]},
    "data_race": {"name": "多线程竞争", "cwe": 362, "cwes": [362, 366, 367]},
    "memory_leak": {"name": "内存泄漏", "cwe": 401, "cwes": [401, 772]},
    "command_execution": {"name": "命令执行漏洞", "cwe": 78, "cwes": [78, 77]},
}
DIFFICULTIES = ("easy", "medium", "hard")
EXPECTED = ("defect", "control")
SOURCE_SUFFIXES = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".S", ".s"}
SKIP_NAMES = {".git", "__pycache__", ".DS_Store", ".code-agent"}
MIN_TARGET_LINES = 10_000
SITES_PER_CATEGORY = 10


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_spec() -> dict[str, Any]:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for key in ("upstream", "project_paths", "clean_files"):
        if key not in spec:
            raise ValueError(f"dataset_spec.json is missing '{key}'")
    return spec


def load_injections(only: list[str] | None = None) -> dict[str, dict[str, Any]]:
    loaded = {}
    for category in only or CATEGORIES:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        path = INJECTIONS / f"{category}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing injection file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("category") != category:
            raise ValueError(f"{path.name}: category must be {category}")
        loaded[category] = value
    return loaded


def clone_or_verify(upstream: dict[str, Any], root: Path) -> Path:
    checkout = root / upstream["id"]
    if not checkout.exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", upstream["ref"], upstream["url"], str(checkout)], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != upstream["commit"]:
        raise ValueError(f"upstream revision mismatch: expected {upstream['commit']}, got {actual}")
    return checkout


def project_files(upstream_root: Path, project_paths: list[str]) -> list[str]:
    """Relative paths of every file copied into the dataset project."""
    files: list[str] = []
    for relative in project_paths:
        node = upstream_root / relative
        if node.is_file():
            files.append(relative)
            continue
        if not node.is_dir():
            raise FileNotFoundError(f"project path missing upstream: {relative}")
        for path in sorted(node.rglob("*")):
            if path.is_dir() or any(part in SKIP_NAMES for part in path.relative_to(upstream_root).parts):
                continue
            if path.is_symlink():
                continue
            files.append(path.relative_to(upstream_root).as_posix())
    unique = sorted(set(files))
    if len(unique) != len(files):
        raise ValueError("project_paths overlap; list each subtree once")
    return unique


_DEFINITION_CACHE: dict[tuple[int, str], list[tuple[int, int, int]]] = {}


def function_ranges(text: str, name: str) -> list[tuple[int, int, int]]:
    """Return (definition_line, body_open_line, body_close_line), 1-based, for column-0 C definitions."""
    key = (hash(text), name)
    if key in _DEFINITION_CACHE:
        return _DEFINITION_CACHE[key]
    lines = text.splitlines()
    pattern = re.compile(r"^[A-Za-z_][^;{}=]*?\b" + re.escape(name) + r"\s*\(")
    found: list[tuple[int, int, int]] = []
    for index, line in enumerate(lines):
        if not pattern.match(line) or line.rstrip().endswith(";") or line.startswith(("#", "typedef", "return", "RTM_EXPORT", "MSH_CMD", "FINSH_", "INIT_")):
            continue
        # Locate the body: first column-0 '{' after the signature, then the matching column-0 '}'.
        open_line = None
        for j in range(index, min(index + 40, len(lines))):
            if lines[j].startswith("{"):
                open_line = j
                break
            if lines[j].rstrip().endswith(";") and j > index:
                break
        if open_line is None:
            continue
        close_line = None
        for j in range(open_line + 1, len(lines)):
            if lines[j].startswith("}"):
                close_line = j
                break
        if close_line is None:
            continue
        found.append((index + 1, open_line + 1, close_line + 1))
    _DEFINITION_CACHE[key] = found
    return found


def apply_edits(text: str, site: dict[str, Any], label: str, errors: list[str]) -> str:
    for number, edit in enumerate(site.get("edits", []), 1):
        old, new = edit.get("old"), edit.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            errors.append(f"{label}: edit {number} needs non-empty 'old' and string 'new'")
            continue
        count = text.count(old)
        if count != 1:
            errors.append(f"{label}: edit {number} anchor occurs {count} times (must be exactly 1)")
            continue
        if old == new:
            errors.append(f"{label}: edit {number} changes nothing")
            continue
        text = text.replace(old, new, 1)
    return text


def resolve(upstream_root: Path, spec: dict[str, Any], injections: dict[str, dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
    """Apply every edit in memory and resolve function ranges and defect lines.

    Returns (patched_contents_by_relative_path, resolved_sites, errors).
    """
    errors: list[str] = []
    contents: dict[str, str] = {}
    originals: dict[str, str] = {}
    files = set(project_files(upstream_root, spec["project_paths"]))

    def text_of(relative: str) -> str | None:
        if relative not in files:
            errors.append(f"{relative}: not part of project_paths")
            return None
        if relative not in contents:
            raw = (upstream_root / relative).read_text(encoding="utf-8", errors="strict")
            originals[relative] = raw
            contents[relative] = raw
        return contents[relative]

    sites: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_functions: dict[tuple[str, str], str] = {}
    for category, value in injections.items():
        raw_sites = value.get("sites", [])
        for site in raw_sites:
            label = f"{category}/{site.get('id', '?')}"
            for key in ("id", "file", "function", "expected", "difficulty", "marker", "description"):
                if not site.get(key):
                    errors.append(f"{label}: missing '{key}'")
            if site.get("expected") not in EXPECTED or site.get("difficulty") not in DIFFICULTIES:
                errors.append(f"{label}: expected must be defect|control and difficulty easy|medium|hard")
            if site.get("id") in seen_ids:
                errors.append(f"{label}: duplicate id")
            seen_ids.add(site.get("id", ""))
            if site.get("expected") == "defect" and not (site.get("trigger") and site.get("fix")):
                errors.append(f"{label}: defects need 'trigger' and 'fix'")
            key2 = (site.get("file", ""), site.get("function", ""))
            if key2 in seen_functions:
                errors.append(f"{label}: function already used by {seen_functions[key2]}; one site per function")
            seen_functions[key2] = label
            text = text_of(site.get("file", ""))
            if text is None:
                continue
            contents[site["file"]] = apply_edits(text, site, label, errors)
        if sum(s.get("expected") == "defect" for s in raw_sites) != SITES_PER_CATEGORY:
            errors.append(f"{category}: need exactly {SITES_PER_CATEGORY} defects")
        if sum(s.get("expected") == "control" for s in raw_sites) < SITES_PER_CATEGORY:
            errors.append(f"{category}: need at least {SITES_PER_CATEGORY} controls")
        sites.extend({**s, "category": category} for s in raw_sites)

    # Add neutral API documentation after injections, before resolving line numbers.
    for relative, notes in spec.get("function_comments", {}).items():
        text = text_of(relative)
        if text is None:
            continue
        for name, note in notes.items():
            ranges = function_ranges(text, name)
            if len(ranges) != 1:
                errors.append(f"{relative}: comment target {name} has {len(ranges)} definitions")
                continue
            lines = text.splitlines(keepends=True)
            lines.insert(ranges[0][0] - 1, f"/* {note} */\n")
            text = "".join(lines)
        contents[relative] = text

    # Second pass: ranges and markers are resolved on the fully patched text.
    resolved: list[dict[str, Any]] = []
    for site in sites:
        label = f"{site['category']}/{site.get('id', '?')}"
        text = contents.get(site.get("file", ""))
        if text is None:
            continue
        ranges = function_ranges(text, site["function"])
        if len(ranges) != 1:
            errors.append(f"{label}: found {len(ranges)} column-0 definitions of {site['function']}() in {site['file']}")
            continue
        definition, _, close = ranges[0]
        marker_lines = [i + 1 for i, line in enumerate(text.splitlines()) if site["marker"] in line]
        if len(marker_lines) != 1:
            errors.append(f"{label}: marker occurs on {len(marker_lines)} lines (must be exactly 1)")
            continue
        line = marker_lines[0]
        if not definition <= line <= close:
            errors.append(f"{label}: marker line {line} is outside {site['function']}() lines {definition}-{close}")
            continue
        if site["expected"] == "control" and site.get("edits") and contents[site["file"]] == originals[site["file"]]:
            errors.append(f"{label}: control declares edits but file is unchanged")
        resolved.append({
            "id": site["id"], "category": site["category"], "cwe": CATEGORIES[site["category"]]["cwe"],
            "file": site["file"], "function": site["function"], "line_start": definition, "line_end": close,
            "line": line, "expected": site["expected"], "difficulty": site["difficulty"],
            "modified": bool(site.get("edits")), "description": site["description"],
            "trigger": site.get("trigger", ""), "fix": site.get("fix", ""),
        })
    for relative in spec["clean_files"]:
        if relative not in files:
            errors.append(f"clean file not in project: {relative}")
        if any(s["file"] == relative for s in sites):
            errors.append(f"clean file has injected sites: {relative}")
    return contents, resolved, errors


def target_files(spec: dict[str, Any], sites: list[dict[str, Any]], contents: dict[str, str], upstream_root: Path) -> list[dict[str, Any]]:
    names = sorted({s["file"] for s in sites} | set(spec["clean_files"]))
    rows = []
    for relative in names:
        text = contents.get(relative)
        if text is None:
            text = (upstream_root / relative).read_text(encoding="utf-8")
        file_sites = [s for s in sites if s["file"] == relative]
        defects = [s for s in file_sites if s["expected"] == "defect"]
        rows.append({
            "file": relative, "lines": len(text.splitlines()), "sha256": digest(text.encode("utf-8")),
            "verdict": "vulnerable" if defects else "clean",
            "defect_count": len(defects), "control_count": len(file_sites) - len(defects),
            "categories": sorted({s["category"] for s in defects}),
            "site_ids": [s["id"] for s in file_sites],
        })
    return rows


def ground_truth_document(spec: dict[str, Any], sites: list[dict[str, Any]], files: list[dict[str, Any]], upstream_root: Path, contents: dict[str, str]) -> dict[str, Any]:
    """Emit the ground truth in the schema the product's loader consumes.

    Top level is ``schema_version`` / ``scan_type`` / ``source_sha256`` / flat
    ``cases``, so the file can be handed straight to ``vulnerability_detection``
    as ``ground_truth_file`` with ``scan_type="high_risk"`` and scored by
    ``security_contracts.evaluate_contract``. The per-case extras
    (``defect_line``, ``defect_code``, ``difficulty``, ``description``,
    ``trigger``, ``fix``) follow the Contract-3 export and are ignored by the loader.
    """
    all_files = project_files(upstream_root, spec["project_paths"])
    source_lines = 0
    for relative in all_files:
        if Path(relative).suffix in SOURCE_SUFFIXES:
            text = contents.get(relative)
            if text is None:
                text = (upstream_root / relative).read_text(encoding="utf-8", errors="replace")
            source_lines += len(text.splitlines())

    def source_line(site: dict[str, Any]) -> str:
        text = contents.get(site["file"])
        if text is None:
            text = (upstream_root / site["file"]).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return lines[site["line"] - 1].strip() if 0 < site["line"] <= len(lines) else ""

    project_id = spec["upstream"]["id"]
    counters: dict[tuple[str, str], int] = {}
    cases = []
    for site in sorted(sites, key=lambda s: (s["category"], s["expected"] != "defect", s["file"], s["line_start"])):
        key = (site["category"], site["expected"])
        counters[key] = counters.get(key, 0) + 1
        cases.append({
            "case_id": f"{project_id}-{site['category']}-{site['expected']}-{counters[key]:02d}",
            "site_id": site["id"], "project": project_id, "category": site["category"], "cwe": site["cwe"],
            "expected": site["expected"], "function": site["function"], "file": site["file"],
            "line_start": site["line_start"], "line_end": site["line_end"],
            "defect_line": site["line"], "defect_code": source_line(site),
            "difficulty": site["difficulty"], "description": site["description"],
            "trigger": site["trigger"], "fix": site["fix"],
        })
    detection_files = [{"path": row["file"], "lines": row["lines"], "sha256": row["sha256"],
                        "verdict": row["verdict"], "defects": row["defect_count"],
                        "controls": row["control_count"], "categories": row["categories"]}
                       for row in files]
    by_category = {}
    for category, meta in CATEGORIES.items():
        rows = [s for s in sites if s["category"] == category]
        by_category[category] = {
            "name": meta["name"], "cwe": meta["cwe"], "accepted_cwes": meta["cwes"],
            "defects": sum(s["expected"] == "defect" for s in rows),
            "controls": sum(s["expected"] == "control" for s in rows),
            "difficulty": {d: sum(s["difficulty"] == d and s["expected"] == "defect" for s in rows) for d in DIFFICULTIES},
        }
    return {
        "schema_version": 1,
        "scan_type": "high_risk",
        "title": "合同指标 5：操作系统底层程序高危代码检测 ground truth（RT-Thread v5.2.1 注入版）",
        "generated_utc": now(),
        "line_rule": "detection_files 是要检测的文件清单，合计不少于 10000 行；行数按文件实际行数统计。",
        "scoring_rule": ("只有 cases 中的函数参与计分：expected=defect 为注入缺陷、expected=control 为同类安全对照。"
                         "告警的 category 需与 case 相同，且函数同名或行号落在 [line_start, line_end] 内才算命中。"
                         "detection_files 中没有 case 的文件为无缺陷文件，其同类告警计为误报。"),
        "metrics": {
            "warning_accuracy": "TP / (TP + FP)  —— 合同“预警准确率”，要求 > 90%",
            "false_positive_rate": "FP / (FP + TN)  —— 合同“误报率”，要求 < 8%",
            "detection_rate": "TP / (TP + FN)",
            "accuracy": "(TP + TN) / (TP + TN + FP + FN)  —— 甲方所说的“正确率”",
        },
        "project": {
            **spec["upstream"], "root": "project", "project_paths": spec["project_paths"],
            "file_count": len(all_files), "source_lines": source_lines,
        },
        "line_totals": {"detection_files": sum(f["lines"] for f in files), "project_source": source_lines},
        "categories": by_category,
        "detection_files": detection_files,
        "source_sha256": {row["file"]: row["sha256"] for row in files},
        "cases": cases,
    }


def write_project(upstream_root: Path, spec: dict[str, Any], contents: dict[str, str]) -> list[dict[str, Any]]:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    rows = []
    for relative in project_files(upstream_root, spec["project_paths"]):
        target = PROJECT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in contents:
            data = contents[relative].encode("utf-8")
            target.write_bytes(data)
        else:
            data = (upstream_root / relative).read_bytes()
            shutil.copyfile(upstream_root / relative, target)
        rows.append({"file": relative, "sha256": digest(data), "modified": relative in contents and contents[relative] != (upstream_root / relative).read_text(encoding="utf-8")})
    return rows


def compile_database(spec: dict[str, Any], project: Path) -> list[dict[str, Any]]:
    """compile_commands.json entries for every C source, following dataset_spec's build_context.

    ``directory`` is relative ("."), resolved against the database's own folder by
    NaturalCC, so the file works wherever the project is unpacked.
    """
    context = spec["build_context"]
    entries = []
    for source in sorted(p for p in project.rglob("*.c") if p.is_file()
                         and not any(part in SKIP_NAMES for part in p.relative_to(project).parts)):
        relative = source.relative_to(project).as_posix()
        local = [d for prefix, values in context.get("local_defines", {}).items() if relative.startswith(prefix) for d in values]
        arguments = ["cc", *(f"-I{d}" for d in context["include_dirs"]),
                     *(flag for d in context.get("after_dirs", []) for flag in ("-idirafter", d)),
                     *(f"-D{d}" for d in [*context.get("defines", []), *local]), "-c", relative]
        entries.append({"directory": ".", "file": relative, "arguments": arguments})
    return entries


def apply_build_context(spec: dict[str, Any], project: Path = PROJECT) -> list[dict[str, Any]]:
    """Copy the overlay (rtconfig.h) into the project and write compile_commands.json.

    Returns manifest rows for the files it wrote.
    """
    rows = []
    for name in spec.get("build_context", {}).get("overlay", []):
        data = (OVERLAY / name).read_bytes()
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        (project / name).write_bytes(data)
        rows.append({"file": name, "sha256": digest(data), "modified": False, "origin": "overlay"})
    if "build_context" in spec:
        data = (json.dumps(compile_database(spec, project), indent=2) + "\n").encode("utf-8")
        (project / "compile_commands.json").write_bytes(data)
        rows.append({"file": "compile_commands.json", "sha256": digest(data), "modified": False, "origin": "generated"})
    return rows


def compile_check(gt: dict[str, Any], project: Path = PROJECT) -> tuple[list[str], list[str]]:
    """Compile every detection file with its database flags; list failures and GT cases
    the configuration compiles out (their function emits no line after preprocessing)."""
    import shlex
    import subprocess
    database = {entry["file"]: entry for entry in json.loads((project / "compile_commands.json").read_text(encoding="utf-8"))}
    marker = re.compile(r'^# (\d+) "([^"]+)"')
    failures, inactive = [], []
    for row in gt["detection_files"]:
        path = row["path"]
        entry = database.get(path)
        if entry is None:
            failures.append(f"{path}: no compile_commands entry")
            continue
        flags = [a for a in (entry.get("arguments") or shlex.split(entry["command"]))[1:] if a not in {"-c", path}]
        result = subprocess.run(["clang", "-fsyntax-only", "-Wno-everything", *flags, path], cwd=project,
                                capture_output=True, text=True)
        if result.returncode != 0:
            errors = [line for line in result.stderr.splitlines() if " error: " in line]
            failures.append(f"{path}: {(errors or [result.stderr.strip()])[0][:200]}")
            continue
        pre = subprocess.run(["clang", "-E", *flags, path], cwd=project, capture_output=True, text=True)
        emitted, current, number = set(), None, 0
        target = str((project / path).resolve())
        for line in pre.stdout.splitlines():
            found = marker.match(line)
            if found:
                number = int(found.group(1))
                name = found.group(2)
                current = name if name.startswith("<") else str((project / name).resolve())
                continue
            if current == target:
                if line.strip():
                    emitted.add(number)
                number += 1
        for case in gt["cases"]:
            if case["file"] == path and not any(case["line_start"] <= n <= case["line_end"] for n in emitted):
                inactive.append(f"{case['site_id']} {case['function']} ({path}:{case['defect_line']})")
    return failures, inactive


def refresh_build_context() -> list[dict[str, Any]]:
    """Offline: re-apply the compile context to the committed project and update the manifest."""
    spec = load_spec()
    rows = apply_build_context(spec)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    written = {row["file"] for row in rows}
    manifest["project_files"] = [row for row in manifest["project_files"] if row["file"] not in written] + rows
    manifest["project_files"].sort(key=lambda row: row["file"])
    manifest["dataset_spec_sha256"] = digest(SPEC.read_bytes())
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def validate_counts(gt: dict[str, Any], injections: dict[str, Any]) -> list[str]:
    problems = []
    detection_lines = gt["line_totals"]["detection_files"]
    if len(injections) == len(CATEGORIES) and detection_lines < MIN_TARGET_LINES:
        problems.append(f"detection files total {detection_lines} lines (< {MIN_TARGET_LINES})")
    for category, row in gt["categories"].items():
        if category not in injections:
            continue
        if row["defects"] != SITES_PER_CATEGORY:
            problems.append(f"{category}: {row['defects']} defects")
    return problems


def summary(gt: dict[str, Any]) -> str:
    lines = [f"项目：{gt['project']['name']} {gt['project']['ref']} ({gt['project']['commit'][:12]})，"
             f"{gt['project']['file_count']} 个文件，源码 {gt['project']['source_lines']} 行",
             f"待检文件：{len(gt['detection_files'])} 个，合计 {gt['line_totals']['detection_files']} 行；"
             f"有缺陷 {sum(f['verdict'] == 'vulnerable' for f in gt['detection_files'])} 个，"
             f"无缺陷 {sum(f['verdict'] == 'clean' for f in gt['detection_files'])} 个"]
    for category, row in gt["categories"].items():
        lines.append(f"  {row['name']} (CWE-{row['cwe']}): 缺陷 {row['defects']} 处 (easy/medium/hard = "
                     f"{row['difficulty']['easy']}/{row['difficulty']['medium']}/{row['difficulty']['hard']})，对照 {row['controls']} 处")
    return "\n".join(lines)


def verify_committed() -> list[str]:
    """Compare the committed project with manifest hashes (no upstream needed)."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for row in manifest["project_files"]:
        path = PROJECT / row["file"]
        if not path.exists():
            problems.append(f"missing: {row['file']}")
        elif digest(path.read_bytes()) != row["sha256"]:
            problems.append(f"hash mismatch: {row['file']}")
    if digest(GROUND_TRUTH.read_bytes()) != manifest["ground_truth_sha256"]:
        problems.append("ground_truth.json hash mismatch")
    present = {p.relative_to(PROJECT).as_posix() for p in PROJECT.rglob("*")
               if p.is_file() and not any(part in SKIP_NAMES for part in p.relative_to(PROJECT).parts)}
    extra = present - {r["file"] for r in manifest["project_files"]}
    problems.extend(f"unexpected file: {name}" for name in sorted(extra))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT, help="directory holding the rt-thread clone (created if missing)")
    parser.add_argument("--check", action="store_true", help="validate anchors/labels only; write nothing")
    parser.add_argument("--category", action="append", help="with --check: validate only these injection files (repeatable)")
    parser.add_argument("--verify", action="store_true", help="verify committed dataset against manifest; needs no upstream")
    parser.add_argument("--build-context", action="store_true",
                        help="re-apply rtconfig.h and compile_commands.json to the committed project; needs no upstream")
    parser.add_argument("--compile-check", action="store_true",
                        help="compile every detection file with clang and list GT cases the configuration compiles out")
    parser.add_argument("--archive", action="store_true", help="write the delivery package contract5.zip from the committed dataset (no upstream needed)")
    args = parser.parse_args()
    if args.build_context:
        for row in refresh_build_context():
            print(f"wrote {row['file']} ({row['origin']})")
        return
    if args.compile_check:
        failures, inactive = compile_check(json.loads(GROUND_TRUTH.read_text(encoding="utf-8")))
        total = len(json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["detection_files"])
        print(f"compiled {total - len(failures)}/{total} detection files")
        for line in failures:
            print(f"  FAIL {line}")
        print(f"GT cases compiled out by this configuration: {len(inactive)}")
        for line in inactive:
            print(f"  - {line}")
        raise SystemExit(1 if failures else 0)
    if args.verify:
        problems = verify_committed()
        print("\n".join(problems) if problems else "dataset matches manifest")
        raise SystemExit(1 if problems else 0)
    spec = load_spec()
    if args.archive and not args.check:
        # Packaging only reads what is already committed; no upstream checkout needed.
        print(f"archive: {write_archive()}")
        return
    if args.category and not args.check:
        raise SystemExit("--category is only valid with --check")
    injections = load_injections(args.category)
    upstream_root = clone_or_verify(spec["upstream"], args.upstream_root.resolve())
    contents, sites, errors = resolve(upstream_root, spec, injections)
    files = target_files(spec, sites, contents, upstream_root)
    gt = ground_truth_document(spec, sites, files, upstream_root, contents)
    errors.extend(validate_counts(gt, injections))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"{len(errors)} problem(s)", file=sys.stderr)
        raise SystemExit(1)
    print(summary(gt))
    if args.check:
        return
    DATASET.mkdir(parents=True, exist_ok=True)
    rows = write_project(upstream_root, spec, contents)
    rows = sorted(rows + apply_build_context(spec), key=lambda row: row["file"])
    GROUND_TRUTH.write_text(json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "generated_utc": gt["generated_utc"], "upstream": spec["upstream"],
        "dataset_spec_sha256": digest(SPEC.read_bytes()),
        "injection_sha256": {c: digest((INJECTIONS / f"{c}.json").read_bytes()) for c in CATEGORIES},
        "ground_truth_sha256": digest(GROUND_TRUTH.read_bytes()),
        "project_files": rows, "modified_files": sorted(r["file"] for r in rows if r["modified"]),
        "target_files": [f["file"] for f in files],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {PROJECT} ({len(rows)} files), {GROUND_TRUTH.name}, {MANIFEST.name}")
    if args.archive:
        print(f"archive: {write_archive()}")


def write_archive() -> Path:
    """Bundle the project, ground truth, prompt, injection sources and scripts as contract5.zip."""
    archive = DATA / "contract5.zip"
    extras = [(GROUND_TRUTH, "ground_truth.json"), (MANIFEST, "manifest.json"),
              (DATA / "prompt.md", "prompt.md"), (SPEC, "dataset_spec.json"),
              (DATA / "使用说明.md", "使用说明.md"),
              (DATA / "comment_coverage.json", "comment_coverage.json")]
    scripts = ("build_contract5_dataset.py", "benchmark_contract5_web.py", "measure_contract5_static.py",
               "measure_contract5_comments.py")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(p for p in PROJECT.rglob("*") if p.is_file()
                           and not any(part in SKIP_NAMES for part in p.relative_to(PROJECT).parts)):
            bundle.write(path, Path("contract5/project") / path.relative_to(PROJECT))
        for path, name in extras:
            if path.exists():
                bundle.write(path, f"contract5/{name}")
        for path in sorted(INJECTIONS.glob("*.json")):
            bundle.write(path, f"contract5/injections/{path.name}")
        for path in sorted(OVERLAY.glob("*")):
            if path.is_file():
                bundle.write(path, f"contract5/overlay/{path.name}")
        for name in scripts:
            path = ROOT / "scripts" / name
            if path.exists():
                bundle.write(path, f"contract5/scripts/{name}")
    return archive


if __name__ == "__main__":
    main()
