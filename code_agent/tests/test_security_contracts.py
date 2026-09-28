import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from code_agent.agent_web_api import app
from code_agent.security_contracts import classify_findings, evaluate_contract, load_ground_truth


def finding(category, line=1, file="a.c", **extra):
    return {"category": category, "metric_eligible": True, "file": file, "line": line, **extra}


def case(key, category, expected, line=1, file="a.c"):
    return {"case_id": key, "category": category, "expected": expected, "file": file,
            "line_start": line, "line_end": line}


def test_case_metrics_dedupe_and_exclude_unrelated_and_unlabeled_findings():
    truth = {"cases": [case("a", "array_oob", "defect"), case("b", "array_oob", "defect", 2),
        case("c", "array_oob", "control", 3), case("d", "array_oob", "control", 4)]}
    findings = [finding("array_oob"), finding("array_oob"), finding("array_oob", 3),
        finding("memory_leak", 2), finding("array_oob", 100), finding(None)]
    result = evaluate_contract(findings, "frequent_defects", truth)
    assert result["overall"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1, "detection_rate": .5,
        "warning_accuracy": .5, "false_positive_rate": .5, "false_discovery_rate": .5, "accuracy": .5}
    assert result["unscored_target_finding_count"] == 1
    assert result["excluded_finding_count"] == 2
    assert result["by_category"]["null_pointer"]["detection_rate"] is None


def test_classification_uses_operation_and_engine_not_cwe(tmp_path):
    (tmp_path / "a.c").write_text('void f(){\nstrcpy(a,b);\nmemcpy(a,b,42);\nint x=a[42];\n}', encoding="utf-8")
    rows = [{"file": "a.c", "line": line, "analyzer": "cppcheck", "analyzer_id": identifier, "rule_id": "cwe-788"}
        for line, identifier in [(2, "bufferAccessOutOfBounds"), (3, "bufferAccessOutOfBounds"), (4, "arrayIndexOutOfBounds"), (4, "unreadVariable")]]
    rows.append({"rule_id": "cwe-120", "file": "a.c", "line": 2})
    classify_findings(rows, tmp_path)
    assert [r["category"] for r in rows] == ["string_overflow", "buffer_overflow", "array_oob", None, "string_overflow"]
    assert [r["metric_eligible"] for r in rows] == [True, True, True, False, False]


def manifest(tmp_path):
    source = b"void f(char *input){system(input);}\nvoid g(void){system(\"date\");}\n"
    (tmp_path / "a.c").write_bytes(source)
    data = {"schema_version": 1, "scan_type": "high_risk", "source_sha256": {"a.c": hashlib.sha256(source).hexdigest()},
        "cases": [case("positive", "command_execution", "defect"), case("control", "command_execution", "control", 2)]}
    return data


@pytest.mark.parametrize("mutation", ["hash", "outside", "duplicate", "overlap", "range", "unscanned", "category"])
def test_ground_truth_rejects_stale_ambiguous_or_unauthorized_cases(tmp_path, mutation):
    data = manifest(tmp_path)
    scanned = [tmp_path / "a.c"]
    if mutation == "hash":
        data["source_sha256"]["a.c"] = "wrong"
    elif mutation == "outside":
        data["cases"][0]["file"] = "../a.c"
    elif mutation == "duplicate":
        data["cases"][1]["case_id"] = "positive"
    elif mutation == "overlap":
        data["cases"][1].update(line_start=1, line_end=1)
    elif mutation == "range":
        data["cases"][0]["line_end"] = 100
    elif mutation == "unscanned":
        scanned = []
    else:
        data["cases"][0]["category"] = "null_pointer"
    (tmp_path / "truth.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(tmp_path, "truth.json", "high_risk", scanned)


def test_api_returns_numeric_statistics_before_truncation(tmp_path):
    data = manifest(tmp_path)
    data["cases"][1].update(expected="defect")
    (tmp_path / "a.c").write_text("void f(char *input){system(input);}\nvoid g(char *input){system(input);}\n", encoding="utf-8")
    data["source_sha256"]["a.c"] = hashlib.sha256((tmp_path / "a.c").read_bytes()).hexdigest()
    (tmp_path / "truth.json").write_text(json.dumps(data), encoding="utf-8")
    payload = {"project_dir": str(tmp_path), "target_files": ["a.c"], "feature": "vulnerability_detection",
        "feature_config": {"analyzer": "builtin", "scan_type": "high_risk", "ground_truth_file": "truth.json", "max_findings": 1}}
    response = TestClient(app).post("/api/run", json=payload)
    done = json.loads(response.text.splitlines()[-1])
    assert done["status"] == "success"
    artifacts = done["artifacts"]
    assert artifacts["finding_summary"]["truncated"]
    assert artifacts["contract_statistics"]["overall"]["tp"] == 2
    assert artifacts["contract_statistics"]["overall"]["detection_rate"] == 1
    assert "passed" not in artifacts["contract_statistics"]


def test_incremental_metrics_include_candidates_beyond_old_internal_limit(tmp_path):
    from code_agent.plugins.base import ExecutionContext, PluginResult
    from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
    source = "void f(char *input){\n" + "system(input);\n" * 1002 + "}\n"
    (tmp_path / "a.c").write_text(source, encoding="utf-8")
    data = {"schema_version": 1, "scan_type": "high_risk", "source_sha256": {"a.c": hashlib.sha256(source.encode()).hexdigest()},
        "cases": [case(str(i), "command_execution", "defect", i + 2) for i in range(1002)]}
    (tmp_path / "truth.json").write_text(json.dumps(data), encoding="utf-8")
    context = ExecutionContext(str(tmp_path), ["a.c"], "", "unused", None,
        {"analyzer": "builtin", "scan_type": "high_risk", "ground_truth_file": "truth.json", "max_findings": 1, "incremental": True})
    for _ in range(2):
        result = next(r for r in VulnerabilityDetectionPlugin().execute(context) if isinstance(r, PluginResult))
        assert result.artifacts["contract_statistics"]["overall"]["tp"] == 1002
        assert result.artifacts["finding_summary"]["returned_count"] == 1
    assert result.artifacts["coverage"][0]["reused_files"] == 1
