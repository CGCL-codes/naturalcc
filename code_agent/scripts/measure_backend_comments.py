"""Measure Python comment coverage without importing the application or its dependencies.

From code_agent/: python scripts/measure_backend_comments.py --json backend_comment_coverage.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tokenize


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIRS = {"agent_core", "api", "plugins", "rag"}


def is_backend_path(path: Path) -> bool:
    """Accept application modules and RAG support code, not delivery or test inputs."""
    if path.suffix != ".py" or path.name.startswith("test_"):
        return False
    if any(part in {"tests", "__pycache__", "vendor", "third_party"} for part in path.parts):
        return False
    return len(path.parts) == 1 or path.parts[0] in BACKEND_DIRS


def count_source(source: str) -> dict[str, int]:
    # Python's tokenizer distinguishes comments from hashes inside prompts, SQL,
    # regexes and ordinary strings. Docstrings are deliberately not counted.
    comment_lines = {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }
    lines = source.splitlines()
    return {
        "physical_lines": len(lines),
        "nonblank_lines": sum(bool(line.strip()) for line in lines),
        "comment_lines": len(comment_lines),
    }


def build_report() -> dict:
    # Git limits the inventory to maintained files; installed packages, caches and
    # untracked generated Python files cannot inflate the reported percentage.
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT
    ).decode("utf-8").split("\0")
    files = []
    for name in sorted(filter(None, names)):
        path = Path(name)
        if not is_backend_path(path):
            continue
        data = (ROOT / path).read_bytes()
        files.append({
            "path": path.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            **count_source(data.decode("utf-8")),
        })
    totals = {
        key: sum(row[key] for row in files)
        for key in ("physical_lines", "nonblank_lines", "comment_lines")
    }
    if not totals["nonblank_lines"]:
        raise ValueError("No nonempty tracked backend Python sources found")
    return {
        "scope": "Git-tracked code_agent/*.py (except test_*) and agent_core/, api/, plugins/, rag/ Python modules, including RAG support and offline utilities.",
        "exclusions": ["dependencies and vendored code", "datasets/artifacts", "tests", "frontend and VS Code JavaScript", "maintenance scripts", "untracked/generated files"],
        "method": "Count each line containing a Python COMMENT token once, including inline comments. Docstrings and all other string literals are excluded. Aggregate across the entire inventory; not a per-file threshold.",
        "formula": "comment_lines / nonblank_lines * 100",
        "threshold_percent_exclusive": 20,
        "file_count": len(files),
        "totals": totals,
        "comment_percent": round(100 * totals["comment_lines"] / totals["nonblank_lines"], 4),
        "comment_percent_including_blank_lines": round(100 * totals["comment_lines"] / totals["physical_lines"], 4),
        "passes": totals["comment_lines"] * 5 > totals["nonblank_lines"],
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the full file inventory and hashes")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['file_count']} backend Python files: "
          f"{report['comment_percent']:.2f}% comments / nonblank lines; "
          f"{report['comment_percent_including_blank_lines']:.2f}% including blank lines")
    if not report["passes"]:
        raise SystemExit("Backend comment coverage must be greater than 20%")


if __name__ == "__main__":
    main()
