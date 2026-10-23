"""Read-only preflight for official evaluation of the saved NaturalCC run."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RUN_DIR = PROJECT_ROOT / "test" / "benchmark_runs" / "easy-batch-20261023-163713-off"
SECREPO = HERE / "SecRepoBench"
DEFECTS4C = HERE / "Defects4C"


def line(label: str, result: str) -> None:
    print(f"{label}: {result}")


def docker_status() -> str:
    executable = shutil.which("docker")
    if not executable:
        return "unavailable (docker CLI not found)"
    try:
        proc = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc})"
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return "unavailable" + (f" ({detail[0]})" if detail else "")
    return f"available (server {proc.stdout.strip()})"


def defects4c_status() -> str:
    url = "https://defects4c.wj2ai.com/list_defects_bugid"
    request = urllib.request.Request(url, headers={"User-Agent": "NaturalCC-Benchmark-Preflight/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return f"available (HTTP 200; status={payload.get('status', 'unknown')})"
    except urllib.error.HTTPError as exc:
        return f"unavailable (HTTP {exc.code})"
    except Exception as exc:  # network/DNS/TLS errors are useful preflight data
        return f"unavailable ({type(exc).__name__}: {exc})"


def main() -> int:
    if not RUN_DIR.is_dir():
        line("Run directory", f"not found: {RUN_DIR}")
        return 2

    results_path = RUN_DIR / "results.jsonl"
    rows = [json.loads(text) for text in results_path.read_text(encoding="utf-8").splitlines() if text.strip()]
    defects = [row for row in rows if row.get("case_id", "").startswith(("5edf", "fe9ed", "eafe", "772245"))]
    sec_repo = [row for row in rows if row not in defects]

    line("Saved samples", f"{len(rows)} total ({len(defects)} Defects4C, {len(sec_repo)} SecRepoBench)")
    line("SecRepoBench upstream", "present" if (SECREPO / "run_eval.py").is_file() else "missing")
    line("Defects4C upstream", "present" if (DEFECTS4C / "http_tutorial.py").is_file() else "missing")
    line("Docker daemon", docker_status())
    line("Defects4C remote verifier", defects4c_status())

    try:
        import shutil as disk_shutil

        usage = disk_shutil.disk_usage(PROJECT_ROOT.drive + "\\")
        line("D: free space", f"{usage.free / (1024**3):.1f} GiB")
    except OSError as exc:
        line("D: free space", f"unavailable ({exc})")

    print("Official scores: not calculated by this preflight.")
    print("This script does not invoke the model, pull images, modify files, or submit patches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
