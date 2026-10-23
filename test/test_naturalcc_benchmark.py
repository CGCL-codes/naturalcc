import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("naturalcc_benchmark", ROOT / "naturalcc_benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(bench)

REPORT_SPEC = importlib.util.spec_from_file_location("summarize_naturalcc_benchmark", ROOT / "summarize_naturalcc_benchmark.py")
report = importlib.util.module_from_spec(REPORT_SPEC)
assert REPORT_SPEC and REPORT_SPEC.loader
REPORT_SPEC.loader.exec_module(report)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_relative_cli_paths_resolve_from_repository_root(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                expected = Path(bench.__file__).resolve().parents[1] / "test"
                self.assertEqual(bench.windows_to_wsl_path("test"), expected.resolve())
            finally:
                os.chdir(original_cwd)

    def test_compaction_budget_defaults_to_sixteen_and_is_overridable(self):
        default_args = bench.make_parser().parse_args(["--dataset-dir", "x"])
        custom_args = bench.make_parser().parse_args(
            ["--dataset-dir", "x", "--max-compaction-calls", "24"]
        )

        self.assertEqual(default_args.max_compaction_calls, 16)
        self.assertEqual(custom_args.max_compaction_calls, 24)

    def test_discovers_only_complete_case_cards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "case-good"
            (good / "workspace").mkdir(parents=True)
            (good / "TASK.md").write_text("task", encoding="utf-8")
            (root / "case-incomplete").mkdir()
            self.assertEqual(bench.discover_cases(root), [good])

    def test_rejects_target_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "case"
            workspace = case / "workspace"
            workspace.mkdir(parents=True)
            with self.assertRaises(ValueError):
                bench.select_targets(case, workspace, ["../outside.c"])

    def test_single_file_workspace_is_automatic_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "case"
            workspace = case / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "buggy_function.c").write_text("int f(void) { return 0; }\n", encoding="utf-8")
            self.assertEqual(bench.select_targets(case, workspace), ["buggy_function.c"])

    def test_multifile_workspace_requires_explicit_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "case"
            workspace = case / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "a.c").write_text("int a;\n", encoding="utf-8")
            (workspace / "b.h").write_text("int b;\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                bench.select_targets(case, workspace)

    def test_detects_common_test_paths(self):
        self.assertTrue(bench.looks_like_test("tests/test_api.py"))
        self.assertTrue(bench.looks_like_test("src/foo.spec.ts"))
        self.assertFalse(bench.looks_like_test("src/foo.c"))

    def test_auto_approval_is_explicit_and_includes_current_risk_types(self):
        self.assertFalse(bench.make_parser().parse_args(["--dataset-dir", "x"]).auto_approve_all)
        self.assertEqual(
            bench.approval_request({"pending_approval": {"risk": "write", "tool_call": {"id": "call-1"}}}),
            {"risk": "write", "tool_call_id": "call-1"},
        )
        for risk in ("execute", "network", "git_write"):
            self.assertEqual(
                bench.approval_request({"pending_approval": {"risk": risk}})["risk"],
                risk,
            )
        with self.assertRaises(ValueError):
            bench.approval_request({"pending_approval": {"risk": "read"}})

    def test_paired_codegraph_summary_uses_only_official_boolean(self):
        result = report.summarize([
            {"case_id": "x", "codegraph": "off", "status": "completed", "official_pass": False},
            {"case_id": "x", "codegraph": "on", "status": "completed", "official_pass": True},
            {"case_id": "y", "codegraph": "off", "status": "completed", "official_pass": None},
        ])
        self.assertEqual(result["paired_official"]["on_only_pass"], 1)
        self.assertEqual(result["paired_official"]["paired_cases"], 1)

    def test_security_density_is_calculated_from_explicit_scan_results(self):
        result = report.summarize(
            [{"case_id": "x", "codegraph": "off", "status": "completed"}],
            [{"case_id": "x", "codegraph": "off", "generated_nonblank_loc": 100,
              "self_check_vulnerability_findings": 2}],
        )
        row = result["by_codegraph"]["off"]
        self.assertEqual(row["security_density_findings_per_100_loc"], 2)
        self.assertEqual(row["security_cases_le_2_findings_per_100_loc"], 1)


if __name__ == "__main__":
    unittest.main()
