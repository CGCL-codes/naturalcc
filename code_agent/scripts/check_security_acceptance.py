"""Scan the prepared cases through the same plugin as /api/run; no LLM required."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from code_agent.plugins.base import ExecutionContext, PluginResult
from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
from code_agent.scripts.prepare_security_acceptance import DEFAULT_OUTPUT, write_json


def check(workspace, output, whole_project=False, api_url=None):
    suite = json.loads((workspace / "suite.json").read_text(encoding="utf-8"))
    results = []
    for project in suite["projects"]:
        root = workspace / project["id"]
        truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
        context = ExecutionContext(str(root), sorted(truth["source_sha256"]), "", "unused", None,
            {"analyzer": "comprehensive", "scan_scope": "project" if whole_project else "targets", "scan_type": project["scan_type"],
             "ground_truth_file": "ground_truth.json", "max_findings": 1000, "rule_profile": "c_cpp"})
        start = time.perf_counter()
        if api_url:
            import httpx
            payload = {"project_dir": str(root), "target_files": context.target_files,
                "feature": "vulnerability_detection", "feature_config": context.feature_config}
            result = None
            with httpx.stream("POST", api_url.rstrip("/") + "/api/run", json=payload, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "error":
                        raise RuntimeError(event.get("log", "API scan failed"))
                    if event.get("type") == "done":
                        result = PluginResult(success=event["status"] == "success", message="API scan",
                            artifacts=event.get("artifacts", {}))
            if result is None:
                raise RuntimeError("API stream ended without a done event")
        else:
            result = next(r for r in VulnerabilityDetectionPlugin().execute(context) if isinstance(r, PluginResult))
        row = {"project": project["id"], "success": result.success, "elapsed_seconds": round(time.perf_counter() - start, 3),
               "scan_scope": context.feature_config["scan_scope"], **result.artifacts}
        results.append(row)
        print(json.dumps({"project": project["id"], "success": result.success,
            "metrics": result.artifacts.get("contract_statistics", {}).get("overall")}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "http_api" if api_url else "plugin", "analyzer": "comprehensive", "suite": suite, "results": results})
    if any(not row["success"] or row.get("contract_statistics", {}).get("status") != "evaluated" for row in results):
        raise SystemExit("Some analyses did not complete; inspect coverage in the result file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/security-acceptance/result.json")
    parser.add_argument("--whole-project", action="store_true", help="Also scan OS sources; default measures only labeled fixtures")
    parser.add_argument("--api-url", help="Call a running backend via /api/run instead of invoking the plugin locally")
    args = parser.parse_args()
    check(args.workspace.resolve(), args.output.resolve(), args.whole_project, args.api_url)
