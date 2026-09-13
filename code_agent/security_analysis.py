"""Optional analyzer adapters. Findings are evidence, never proof of safety."""
from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

C_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}


def cppcheck_scan(files: list[str], project_dir: str, context_lines: int):
    targets = [str(Path(p).resolve()) for p in files if Path(p).suffix.lower() in C_EXTENSIONS]
    if not targets:
        return [], {"engine": "cppcheck", "status": "not_applicable"}
    executable = shutil.which("cppcheck")
    if executable is None:
        return [], {"engine": "cppcheck", "status": "unavailable", "message": "Install Cppcheck and add it to the backend service PATH; bounds/null/leak analysis was not run."}
    try:
        result = subprocess.run(
            [executable, "--xml", "--xml-version=2", "--enable=warning,style,performance,portability", "--inconclusive", *targets],
            cwd=project_dir, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        if result.returncode != 0:
            return [], {"engine": "cppcheck", "status": "failed", "message": result.stderr[-2000:]}
        document = ET.fromstring(result.stderr)
    except (OSError, subprocess.TimeoutExpired, ET.ParseError) as exc:
        return [], {"engine": "cppcheck", "status": "failed", "message": str(exc)}
    root = Path(project_dir).resolve()
    findings, diagnostics = [], []
    for error in document.findall(".//error"):
        identifier = error.get("id", "unknown")
        if identifier in {"syntaxError", "internalError", "cppcheckError", "missingInclude", "missingIncludeSystem"}:
            diagnostics.append(error.get("msg", identifier))
            continue
        for location in error.findall("location")[:1]:
            path = Path(location.get("file", ""))
            if not path.is_absolute():
                path = root / path
            try:
                relative = path.resolve().relative_to(root).as_posix()
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                line = max(1, int(location.get("line", "1")))
            except (ValueError, OSError):
                continue
            cwe = error.get("cwe")
            findings.append({
                "rule_id": f"cwe-{cwe}" if cwe and cwe != "0" else f"cppcheck-{identifier}",
                "rule_name": error.get("msg", identifier), "analyzer": "cppcheck", "analyzer_id": identifier,
                "severity": "high" if error.get("severity") == "error" else "medium",
                "confidence": "low" if error.get("inconclusive") == "true" else "medium",
                "file": relative, "line": line,
                "snippet": lines[line - 1].strip()[:300] if line <= len(lines) else "",
                "context": "".join(f"{i + 1}: {lines[i]}" for i in range(max(0, line - 1 - context_lines), min(len(lines), line + context_lines))),
                "pattern": identifier, "recommendation": error.get("verbose", error.get("msg", "")),
                "references": [f"https://cwe.mitre.org/data/definitions/{cwe}.html"] if cwe and cwe != "0" else [],
            })
    return findings, {"engine": "cppcheck", "status": "partial" if diagnostics else "completed", "diagnostics": diagnostics,
        "limitations": "Default compiler configuration; no compile database. Cross-file and build-specific coverage is limited. All selected C/C++ files are reanalyzed on every scan."}


def import_tsan_report(project_dir: str, report_path: str):
    root = Path(project_dir).resolve()
    path = (root / report_path).resolve()
    path.relative_to(root)
    if path.stat().st_size > 2_000_000:
        raise ValueError("sanitizer report exceeds 2 MB")
    text = path.read_text(encoding="utf-8", errors="replace")
    findings = []
    for block in text.split("WARNING: ThreadSanitizer: data race")[1:]:
        # Select a source stack frame in this workspace, never a runtime frame.
        for frame in block.splitlines():
            match = re.search(r"(?:^|\s)([^\s]+\.(?:c|cc|cpp|cxx|h|hpp)):(\d+)(?::\d+)?", frame)
            if not match:
                continue
            source = (root / match[1]).resolve()
            try:
                relative = source.relative_to(root).as_posix()
            except ValueError:
                continue
            findings.append({
                "rule_id": "cwe-362", "rule_name": "Data race reported by ThreadSanitizer",
                "severity": "high", "confidence": "medium", "file": relative, "line": int(match[2]),
                "snippet": frame.strip(), "context": block[:4000], "pattern": "ThreadSanitizer: data race",
                "recommendation": "Review conflicting accesses and synchronize shared state; rerun instrumented tests.",
                "references": ["https://clang.llvm.org/docs/ThreadSanitizer.html"], "analyzer": "tsan-import",
                "evidence": "User-supplied runtime log; source revision and authenticity are not verified.",
            })
            break
    return findings, {"engine": "tsan-import", "status": "imported", "findings": len(findings),
        "limitations": "Only imported logs are inspected. No instrumented build or program was executed; absence of findings does not establish race freedom."}
