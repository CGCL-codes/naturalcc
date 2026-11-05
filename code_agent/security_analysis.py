"""Optional analyzer adapters. Findings are evidence, never proof of safety."""
from __future__ import annotations

import re
import plistlib
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

C_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}


def _cppcheck_flags(flags: list[str]) -> list[str]:
    """Translate compile flags to Cppcheck's options.

    Cppcheck reads no system headers, so -isystem/-idirafter directories become
    ordinary -I paths after the project ones; -std=gnuXX becomes --std=cXX.
    """
    regular, later, other, index = [], [], [], 0
    while index < len(flags):
        flag = flags[index]
        if flag in {"-I", "-iquote"} and index + 1 < len(flags):
            regular += ["-I", flags[index + 1]]
            index += 2
        elif flag in {"-isystem", "-idirafter"} and index + 1 < len(flags):
            later += ["-I", flags[index + 1]]
            index += 2
        elif flag == "-include" and index + 1 < len(flags):
            other.append(f"--include={flags[index + 1]}")
            index += 2
        elif flag == "-imacros" and index + 1 < len(flags):
            index += 2
        elif flag.startswith(("-D", "-U")):
            other.append(flag)
            index += 1
        elif flag.startswith("-std="):
            standard = re.sub(r"^gnu", "c", flag[5:])
            if re.fullmatch(r"c(89|99|11|17|23)", standard):
                other.append(f"--std={standard}")
            index += 1
        else:
            index += 1
    return [*regular, *later, *other]


def cppcheck_scan(files: list[str], project_dir: str, context_lines: int, timeout_seconds: int = 120,
                  build_context=None):
    targets = [str(Path(p).resolve()) for p in files if Path(p).suffix.lower() in C_EXTENSIONS]
    if not targets:
        return [], {"engine": "cppcheck", "status": "not_applicable"}
    executable = shutil.which("cppcheck")
    if executable is None:
        return [], {"engine": "cppcheck", "status": "unavailable", "message": "Install Cppcheck and add it to the backend service PATH; bounds/null/leak analysis was not run."}
    root = Path(project_dir).resolve()
    # One invocation per distinct compile context: Cppcheck takes a single set of
    # macros per run, and RT-Thread defines differ per directory.
    groups: dict[tuple[str, ...], list[str]] = {}
    for target in targets:
        relative = Path(target).relative_to(root).as_posix()
        flags = build_context.flags_for(relative) if build_context is not None else []
        groups.setdefault(tuple(_cppcheck_flags(flags)), []).append(target)
    findings, diagnostics, failed_files, failures, version_text = [], [], set(), [], None
    used, pending = 0.0, len(targets)
    for flags, members in groups.items():
        left = timeout_seconds - used
        if left <= 1:
            failures.append("Cppcheck time budget exhausted")
            failed_files.update(Path(p).relative_to(root).as_posix() for p in members)
            continue
        # Split what is left of the budget by the share of files still to check; a
        # single invocation gets the whole budget.
        share = left * len(members) / pending
        pending -= len(members)
        started = time.monotonic()
        try:
            result = subprocess.run(
                [executable, "--xml", "--xml-version=2", "--enable=warning,style,performance,portability", "--inconclusive",
                 *flags, "-I", str(root), *members],
                cwd=project_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=min(left, max(share, 30.0)),
            )
            if result.returncode != 0:
                failures.append((result.stderr or result.stdout)[-2000:])
                failed_files.update(Path(p).relative_to(root).as_posix() for p in members)
                continue
            document = ET.fromstring(result.stderr)
        except (OSError, subprocess.TimeoutExpired, ET.ParseError) as exc:
            failures.append(str(exc))
            failed_files.update(Path(p).relative_to(root).as_posix() for p in members)
            continue
        finally:
            used += time.monotonic() - started
        version = document.find("cppcheck")
        version_text = version.get("version") if version is not None else version_text
        for error in document.findall(".//error"):
            identifier = error.get("id", "unknown")
            if identifier in {"syntaxError", "internalError", "cppcheckError", "missingInclude", "missingIncludeSystem"}:
                diagnostics.append(error.get("msg", identifier))
                if identifier in {"syntaxError", "internalError", "cppcheckError"}:
                    locations = error.findall("location")
                    for location in locations:
                        path = (root / location.get("file", "")).resolve()
                        try:
                            failed_files.add(path.relative_to(root).as_posix())
                        except ValueError:
                            continue
                    if not locations:
                        failed_files.update(Path(p).relative_to(root).as_posix() for p in members)
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
    all_failed = len(failures) == len(groups)
    status = "failed" if all_failed else "partial" if diagnostics or failures else "completed"
    described = build_context.describe() if build_context is not None else {"source": "default"}
    report = {"engine": "cppcheck", "status": status, "diagnostics": diagnostics, "failed_files": sorted(failed_files),
        "version": version_text, "invocations": len(groups), "build_context": described,
        "limitations": ("Compile context from " + described["source"] + ". " if described["source"] != "default"
                        else "Default compiler configuration; no compile database. ")
                       + "Cross-file and build-specific coverage is limited. All selected C/C++ files are reanalyzed on every scan."}
    if failures:
        report["message"] = failures[0][-2000:] if isinstance(failures[0], str) else str(failures[0])
    return findings, report


CLANG_EXPERIMENTAL = ["alpha.security.ArrayBoundV2", "alpha.security.ArrayBound", "alpha.unix.cstring.OutOfBounds",
                      "alpha.security.taint.TaintPropagation"]
CLANG_CWE = {"core.NullDereference": 476, "alpha.security.ArrayBoundV2": 119, "alpha.security.ArrayBound": 119,
             "alpha.unix.cstring.OutOfBounds": 787}


def _clang_analyze(executable: str, unit: Path, root: Path, output: Path, timeout: float, analyze_included: bool,
                   flags: tuple[str, ...] | list[str] = ()):
    output.unlink(missing_ok=True)
    # Project compile context first; the project root stays as the last fallback.
    result = subprocess.run([executable, "--analyze", "-Wno-everything", *flags, "-I", str(root),
        "-Xanalyzer", "-analyzer-checker=core,unix,cplusplus," + ",".join(CLANG_EXPERIMENTAL),
        # plist-multi-file keeps diagnostics whose path crosses translation-unit files.
        "-Xanalyzer", "-analyzer-output=plist-multi-file",
        *(["-Xanalyzer", "-analyzer-opt-analyze-headers"] if analyze_included else []),
        "-o", str(output), str(unit)],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(result.stderr[-1500:] or "Clang produced no report")
    return plistlib.loads(output.read_bytes())


def clang_scan(files: list[str], project_dir: str, context_lines: int, timeout_seconds: int = 120,
               build_context=None):
    """Opt-in comprehensive profile. Never compile/link/run the target program.

    Several C sources are analysed together as one unity translation unit so that
    data flow across files (callers, globals, constants) is visible; if that unit
    does not compile (for example duplicate static names), each file is analysed alone.
    """
    targets = [Path(p).resolve() for p in files if Path(p).suffix.lower() in C_EXTENSIONS]
    if not targets:
        return [], {"engine": "clang", "status": "not_applicable"}
    executable = shutil.which("clang")
    if executable is None:
        return [], {"engine": "clang", "status": "unavailable", "message": "Comprehensive analysis requires clang on the backend PATH (validated with LLVM 18)."}
    root, findings, failures, versions, seen = Path(project_dir).resolve(), [], [], set(), set()
    deadline = time.monotonic() + timeout_seconds
    sources = [t for t in targets if t.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}]

    def flags_of(path: Path) -> tuple[str, ...]:
        return tuple(build_context.flags_for(path.relative_to(root).as_posix())) if build_context is not None else ()

    mode = "per_file"
    with tempfile.TemporaryDirectory(prefix="naturalcc-clang-") as temporary:
        output = Path(temporary) / "findings.plist"
        documents = []
        # A unity unit needs one compile context for all of its sources.
        if len(sources) > 1 and len({t.suffix.lower() for t in sources}) == 1 and len({flags_of(t) for t in sources}) == 1:
            unity = Path(temporary) / f"naturalcc_unity{sources[0].suffix.lower()}"
            unity.write_text("".join(f'#include "{t}"\n' for t in sources), encoding="utf-8")
            try:
                documents.append(_clang_analyze(executable, unity, root, output, max(1, min(60, deadline - time.monotonic())), True,
                                                flags_of(sources[0])))
                mode = "unity"
            except (OSError, RuntimeError, subprocess.TimeoutExpired, plistlib.InvalidFileException):
                documents = []
        if mode == "per_file":
            # With full project headers a translation unit is much larger, so give each
            # file a fair share of the budget instead of a fixed 15 seconds.
            per_file = max(15.0, timeout_seconds / max(1, len(targets)) * 2)
            for index, target in enumerate(targets):
                relative = target.relative_to(root).as_posix()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failures.extend({"file": p.relative_to(root).as_posix(), "message": "Total Clang timeout exceeded"} for p in targets[index:])
                    break
                try:
                    documents.append(_clang_analyze(executable, target, root, output, min(per_file, remaining), False,
                                                    flags_of(target)))
                except (OSError, RuntimeError, subprocess.TimeoutExpired, plistlib.InvalidFileException) as exc:
                    failures.append({"file": relative, "message": str(exc)})
        for document in documents:
            versions.add(document.get("clang_version", "unknown"))
            for diagnostic in document.get("diagnostics", []):
                location = diagnostic["location"]
                source = Path(document["files"][location["file"]]).resolve()
                try:
                    file = source.relative_to(root).as_posix()
                    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
                except (ValueError, OSError):
                    continue
                line, checker = location["line"], diagnostic["check_name"]
                if (file, line, checker) in seen:
                    continue
                seen.add((file, line, checker))
                # unix.Malloc also reports double-free/UAF: use its diagnostic type.
                cwe = CLANG_CWE.get(checker)
                if checker == "unix.Malloc" and diagnostic["type"] == "Memory leak":
                    cwe = 401
                evidence = [step.get("message") for step in diagnostic.get("path", []) if step.get("kind") == "event"]
                finding = {"rule_id": f"cwe-{cwe}" if cwe else f"clang-{checker}",
                    "rule_name": diagnostic["description"], "analyzer": "clang", "analyzer_id": checker,
                    "diagnostic_type": diagnostic["type"], "severity": "high", "confidence": "medium",
                    "experimental": checker in CLANG_EXPERIMENTAL, "file": file, "line": line,
                    "snippet": lines[line - 1][:300] if line <= len(lines) else "",
                    "context": "\n".join(f"{i + 1}: {lines[i]}" for i in range(max(0, line - 1 - context_lines), min(len(lines), line + context_lines))),
                    "evidence": evidence, "pattern": checker, "recommendation": diagnostic["description"],
                    "references": ["https://clang.llvm.org/docs/analyzer/checkers.html"]}
                findings.append(finding)
                # When the fault surfaces inside a callee, also attribute it to the outermost call
                # that supplies the faulting value (the root frame of the reported path).
                calls = [step for step in diagnostic.get("path", []) if step.get("kind") == "event" and step.get("depth", 0) == 0
                         and str(step.get("message", "")).startswith("Calling ")]
                if calls and any(step.get("depth", 0) > 0 for step in diagnostic.get("path", []) if step.get("kind") == "event"):
                    call = calls[-1]["location"]
                    try:
                        call_source = Path(document["files"][call["file"]]).resolve()
                        call_file = call_source.relative_to(root).as_posix()
                        call_lines = call_source.read_text(encoding="utf-8", errors="replace").splitlines()
                    except (ValueError, OSError, IndexError):
                        call_file = None
                    if call_file and (call_file, call["line"], checker) not in seen:
                        seen.add((call_file, call["line"], checker))
                        call_line = call["line"]
                        findings.append({**finding, "file": call_file, "line": call_line, "attribution": "call_site",
                            "callee_location": {"file": file, "line": line},
                            "snippet": call_lines[call_line - 1][:300] if call_line <= len(call_lines) else "",
                            "context": "\n".join(f"{i + 1}: {call_lines[i]}" for i in range(max(0, call_line - 1 - context_lines), min(len(call_lines), call_line + context_lines)))})
    described = build_context.describe() if build_context is not None else {"source": "default"}
    status = "failed" if len(failures) == len(targets) else "partial" if failures else "completed"
    report = {"engine": "clang", "status": status,
        "mode": mode, "versions": sorted(versions), "failed_files": [f["file"] for f in failures], "diagnostics": failures[:50],
        "experimental_checkers": CLANG_EXPERIMENTAL, "build_context": described,
        "limitations": ("Compile context from " + described["source"] + ". " if described["source"] != "default"
                        else "Default compiler configuration; no compile database. ")
                       + "Multiple sources are analysed as one unity unit when they share a compile context and compile together. "
                         "Bounds and taint checkers are LLVM experimental rules and can produce false positives. No target program was executed."}
    missing = [f for f in failures if "file not found" in f["message"]]
    if missing:
        report["hint"] = ("Headers could not be found. Provide compile_commands.json (or compile_flags.txt) in the "
                          "project root, or pass include_dirs/defines in the feature config.")
    return findings, report


def import_tsan_report(project_dir: str, report_path: str):
    root = Path(project_dir).resolve()
    path = (root / report_path).resolve()
    path.relative_to(root)
    if path.stat().st_size > 2_000_000:
        raise ValueError("sanitizer report exceeds 2 MB")
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"FATAL: ThreadSanitizer|ThreadSanitizer: (?:CHECK failed|unexpected memory mapping)", text):
        return [], {"engine": "tsan-import", "status": "failed",
            "message": "ThreadSanitizer runtime failed; this log is not a clean race test.", "diagnostics": text[-2000:]}
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
    warnings = text.count("WARNING: ThreadSanitizer: data race")
    return findings, {"engine": "tsan-import", "status": "partial" if warnings > len(findings) else "imported", "findings": len(findings),
        "unmatched_race_blocks": warnings - len(findings),
        "limitations": "Only imported logs are inspected. No instrumented build or program was executed; absence of findings does not establish race freedom."}
