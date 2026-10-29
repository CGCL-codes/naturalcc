"""Toggle the five Contract-3 static-analysis switches (A-E) for exercise design.

Each switch is an exact source replacement marked with "# [contract3-switch X off]",
so the product stays unchanged until a switch is turned off and `reset` restores it
byte for byte. See 合同-3-静态分析难度调节指南.md for what each switch does and the
measured detection rates.

  uv run python scripts/contract3_static_switches.py status
  uv run python scripts/contract3_static_switches.py only A          # all on except A
  uv run python scripts/contract3_static_switches.py off A C E       # turn these off, keep the rest
  uv run python scripts/contract3_static_switches.py on C            # turn C back on
  uv run python scripts/contract3_static_switches.py reset           # all on (full rules)
  uv run python scripts/contract3_static_switches.py only A --measure

Restart the Web backend after switching; it loads the rules at start-up.
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name: (file, enabled source, disabled source, description)
SWITCHES = {
    "A": ("security_contracts.py",
          "                string = any(call in STRING_CALLS or (call in BUFFER_CALLS and literal) for call, literal in at_line)\n",
          "                string = any(call in STRING_CALLS for call, literal in at_line)  # [contract3-switch A off]\n",
          "memcpy/memmove with a string-literal source counts as string overflow"),
    "B": ("security_analysis.py",
          "                if calls and any(step.get(\"depth\", 0) > 0 for step in diagnostic.get(\"path\", []) if step.get(\"kind\") == \"event\"):\n",
          "                if False and calls and any(step.get(\"depth\", 0) > 0 for step in diagnostic.get(\"path\", []) if step.get(\"kind\") == \"event\"):  # [contract3-switch B off]\n",
          "Clang: also report a callee fault at the outermost call site"),
    "C": ("security_c_rules.py",
          "                        results.append((node, [f\"via call to {name}(): \" + item for item in evidence]))\n",
          "                        pass  # [contract3-switch C off] results.append((node, [f\"via call to {name}(): \" + item for item in evidence]))\n",
          "capacity rule: report a callee overflow at the call site"),
    "D": ("security_c_rules.py",
          "        for node, evidence in _string_writes(function, functions):\n",
          "        for node, evidence in []:  # [contract3-switch D off] _string_writes(function, functions)\n",
          "capacity rule itself (string write longer than destination capacity)"),
    "E": ("security_c_rules.py",
          "        for node, evidence in _null_arguments(function, functions):\n",
          "        for node, evidence in []:  # [contract3-switch E off] _null_arguments(function, functions)\n",
          "NULL argument reaching an unchecked dereference in a same-file callee"),
}


def state(name: str) -> str:
    file, enabled, disabled, _ = SWITCHES[name]
    source = (ROOT / file).read_text(encoding="utf-8")
    on, off = source.count(enabled), source.count(disabled)
    if (on, off) == (1, 0):
        return "on"
    if (on, off) == (0, 1):
        return "off"
    raise SystemExit(f"Switch {name}: expected source not found exactly once in {file} "
                     f"(on={on}, off={off}); the code changed, update SWITCHES")


def set_switch(name: str, enable: bool) -> bool:
    current = state(name)
    if current == ("on" if enable else "off"):
        return False
    file, enabled, disabled, _ = SWITCHES[name]
    path = ROOT / file
    source = path.read_text(encoding="utf-8")
    path.write_text(source.replace(disabled, enabled) if enable else source.replace(enabled, disabled), encoding="utf-8")
    return True


def print_status() -> None:
    for name, (file, _, _, description) in SWITCHES.items():
        print(f"  {name}  {state(name):3}  {file:22} {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "on", "off", "only", "reset"))
    parser.add_argument("switches", nargs="*", type=str.upper, help="Switch names A-E")
    parser.add_argument("--measure", action="store_true", help="Run scripts/measure_contract3_static.py on the four RTOS suites afterwards")
    args = parser.parse_args()
    unknown = [name for name in args.switches if name not in SWITCHES]
    if unknown:
        raise SystemExit(f"Unknown switch: {', '.join(unknown)} (use A-E)")
    if args.command in {"on", "off", "only"} and not args.switches:
        raise SystemExit(f"'{args.command}' needs at least one switch name")
    for name in SWITCHES:
        state(name)  # validate every anchor before editing anything
    if args.command == "reset":
        targets = {name: True for name in SWITCHES}
    elif args.command == "only":
        targets = {name: name not in args.switches for name in SWITCHES}
    elif args.command in {"on", "off"}:
        targets = {name: args.command == "on" for name in args.switches}
    else:
        targets = {}
    changed = [name for name, enable in targets.items() if set_switch(name, enable)]
    for file in {SWITCHES[name][0] for name in SWITCHES}:
        py_compile.compile(str(ROOT / file), doraise=True)
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
