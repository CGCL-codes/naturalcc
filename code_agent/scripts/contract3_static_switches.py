"""Toggle the seven Contract-3 static-analysis switches (A-G) for exercise design.

The switches apply only while Contract 3 is evaluated: the vulnerability_detection
plugin with scan_type="frequent_defects" and scripts/measure_contract3_static.py.
Every other scan (Contract 5's high_risk, plain scans) always runs the full rules.
This script edits the one line ``CONTRACT3_OFF = frozenset({...})`` in
security_exam.py; no rule code is rewritten. See 合同-3-静态分析难度调节指南.md for
what each switch does and the measured rates.

  uv run python scripts/contract3_static_switches.py status
  uv run python scripts/contract3_static_switches.py only A C E F    # off during Contract 3: A, C, E, F
  uv run python scripts/contract3_static_switches.py off B           # also turn B off
  uv run python scripts/contract3_static_switches.py on C            # turn C back on
  uv run python scripts/contract3_static_switches.py reset           # all on: full rules for Contract 3 too
  uv run python scripts/contract3_static_switches.py only A --measure

Restart the Web backend after switching; it loads the rules at start-up.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAM = ROOT / "security_exam.py"
LINE = re.compile(r"^CONTRACT3_OFF = frozenset\((?:\{(?P<names>[^}]*)\})?\)$", re.M)

# name: (file holding the switch, description)
SWITCHES = {
    "G": ("security_contracts.py", "classify pointer-offset dereferences/subscripts as array bounds findings"),
    "A": ("security_contracts.py", "memcpy/memmove with a string-literal source counts as string overflow"),
    "B": ("security_analysis.py", "Clang: also report a callee fault at the outermost call site"),
    "C": ("security_c_rules.py", "capacity rule: report a callee overflow at the call site"),
    "D": ("security_c_rules.py", "capacity rule itself (string write longer than destination capacity)"),
    "E": ("security_c_rules.py", "NULL argument reaching an unchecked dereference in a same-file callee"),
    "F": ("security_c_rules.py", "path-sensitive NULL guard (off: NULL-initialised pointers stay NULL after a conditional reassignment)"),
}


def read_off() -> set[str]:
    matches = LINE.findall(EXAM.read_text(encoding="utf-8"))
    match = LINE.search(EXAM.read_text(encoding="utf-8"))
    if len(matches) != 1 or match is None:
        raise SystemExit("security_exam.py: expected exactly one line 'CONTRACT3_OFF = frozenset({...})'")
    return set(re.findall(r'"([A-G])"', match.group("names") or ""))


def write_off(names: set[str]) -> None:
    source = EXAM.read_text(encoding="utf-8")
    body = ", ".join(f'"{name}"' for name in sorted(names))
    EXAM.write_text(LINE.sub(f"CONTRACT3_OFF = frozenset({{{body}}})" if names else "CONTRACT3_OFF = frozenset()", source, count=1),
                    encoding="utf-8")


def state(name: str) -> str:
    return "off" if name in read_off() else "on"


def print_status() -> None:
    off = read_off()
    for name, (file, description) in SWITCHES.items():
        print(f"  {name}  {'off' if name in off else 'on':3}  {file:22} {description}")
    print("  (applies only to Contract-3 evaluation; other scans use the full rules)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "on", "off", "only", "reset"))
    parser.add_argument("switches", nargs="*", type=str.upper, help="Switch names A-G")
    parser.add_argument("--measure", action="store_true", help="Run scripts/measure_contract3_static.py on the four RTOS suites afterwards")
    args = parser.parse_args()
    unknown = [name for name in args.switches if name not in SWITCHES]
    if unknown:
        raise SystemExit(f"Unknown switch: {', '.join(unknown)} (use A-G)")
    if args.command in {"on", "off", "only"} and not args.switches:
        raise SystemExit(f"'{args.command}' needs at least one switch name")
    before = read_off()
    if args.command == "reset":
        after = set()
    elif args.command == "only":
        after = set(args.switches)
    elif args.command == "off":
        after = before | set(args.switches)
    elif args.command == "on":
        after = before - set(args.switches)
    else:
        after = before
    if after != before:
        write_off(after)
    changed = sorted(before ^ after)
    print(("Changed: " + ", ".join(changed)) if changed else "No change.")
    print_status()
    if changed:
        print("Restart the Web backend so the running service loads the new rules.")
    if args.measure:
        missing = [tool for tool in ("cppcheck", "clang") if shutil.which(tool) is None]
        if missing:
            raise SystemExit(f"{' and '.join(missing)} not on PATH; cannot measure")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "measure_contract3_static.py")], check=True)


if __name__ == "__main__":
    main()
