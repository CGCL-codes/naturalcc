"""Report comment coverage of Contract-5 detection sources, excluding libc glue.

Run from the repository or the scripts/ directory in the delivery archive.
Only lexical C comments count; comment-like strings and blank lines do not.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from build_contract5_dataset import DATA, GROUND_TRUTH, PROJECT

TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*[\s\S]*?\*/|//[^\n]*')


def count_comments(source: str) -> dict:
    lines = source.splitlines()
    comment_lines = set()
    for match in TOKENS.finditer(source):
        if not match.group().startswith(("/*", "//")):
            continue
        first = source.count("\n", 0, match.start())
        for offset, line in enumerate(match.group().splitlines()):
            if line.strip():
                comment_lines.add(first + offset)
    return {"comment_lines": len(comment_lines),
            "nonblank_lines": sum(bool(line.strip()) for line in lines),
            "physical_lines": len(lines)}


def main() -> None:
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    rows, excluded = [], []
    for item in gt["detection_files"]:
        path = item["path"]
        if path.startswith("components/libc/"):
            excluded.append(path)
            continue
        rows.append({"path": path, **count_comments((PROJECT / path).read_text(encoding="utf-8"))})
    totals = {key: sum(row[key] for row in rows)
              for key in ("comment_lines", "nonblank_lines", "physical_lines")}
    report = {
        "scope": "核心检测源码：ground_truth.detection_files 排除 components/libc/。RT-Thread 是被测主体；不将其核心内核与组件视为外部依赖。仅作为编译上下文的头文件、libcpu、外部文件系统库及未列入检测清单的源码不计入。",
        "method": "按 C 词法识别 /* */ 与 // 注释，字符串中的注释符不计。每个非空注释物理行计一次，包含块注释界定符及原有文件头；行尾注释与独占行注释均计。按全部核心文件汇总，不要求单文件达到 20%。",
        "formula": "comment_lines / nonblank_lines * 100",
        "file_count": len(rows), "excluded_detection_files": excluded,
        "totals": totals,
        "comment_percent": round(totals["comment_lines"] / totals["nonblank_lines"] * 100, 4),
        "comment_percent_including_blank_lines": round(totals["comment_lines"] / totals["physical_lines"] * 100, 4),
        "files": rows,
    }
    destination = DATA / "comment_coverage.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} core files: {report['comment_percent']:.2f}% of nonblank lines; "
          f"{report['comment_percent_including_blank_lines']:.2f}% including blank lines")
    print(destination)
    if report["comment_percent"] < 20:
        raise SystemExit("Comment coverage is below 20%")


if __name__ == "__main__":
    main()
