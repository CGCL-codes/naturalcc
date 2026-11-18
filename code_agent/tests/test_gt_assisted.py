"""Mixed-origin replay must stay explicit, hash-bound and distinct from measurement."""
import copy
import hashlib
import json

import pytest

from code_agent import security_gt_assisted as gt


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    lines = ["#define LIMIT 4"]
    cases = []
    for category in gt.CATEGORY_ORDER:
        for expected in ("defect", "control"):
            for index in range(20):
                name = f"sample_{category}_{expected}_{index:02}"
                bad = expected == "defect"
                body = {
                    "array_oob": f"int values[LIMIT]; values[{4 if bad else 1}]=1;",
                    "string_overflow": f"char text[LIMIT]; text[{4 if bad else 1}]=0;",
                    "null_pointer": "int *p=0; *p=1;" if bad else "int value=1; int *p=&value; *p=1;",
                }[category]
                lines.append(f"void {name}(void) {{ {body} }}")
                number = len(lines)
                cases.append(dict(case_id=name, file="sample.c", function=name,
                                  category=category, expected=expected,
                                  line_start=number, line_end=number, defect_line=number,
                                  function_start=number, function_end=number))
    source = "\n".join(lines) + "\n"
    (root / "sample.c").write_text(source)
    truth = dict(schema_version=1, dataset_revision=gt.REVISION,
                 scan_type="frequent_defects", cases=cases,
                 source_sha256={"sample.c": hashlib.sha256(source.encode()).hexdigest()})
    (root / "ground_truth.json").write_text(json.dumps(truth))
    monkeypatch.setenv("NATURALCC_GT_ASSISTED_CACHE", str(tmp_path / "cache"))
    return root, truth, [root / "sample.c"]


def test_partition_and_context_preserve_declarations_and_original_lines(snapshot):
    root, truth, files = snapshot
    plan = gt.prepare(root, "ground_truth.json", files)
    assert len(plan["measured_case_ids"]) == 12
    assert len(plan["presets"]) == 108
    assert not set(plan["measured_case_ids"]) & {p["case_id"] for p in plan["presets"]}
    context = plan["contexts"][0]
    assert "#define LIMIT 4" in context["source"]
    assert "sample_array_oob_defect_00" in context["source"]
    assert "sample_array_oob_defect_19" not in context["source"]
    assert context["line_map"][0] == 1
    assert len(context["line_map"]) == 13


@pytest.mark.parametrize("detect_all", [False, True])
def test_combined_rates_have_explicit_preset_provenance(snapshot, monkeypatch, detect_all):
    root, truth, files = snapshot
    plan = gt.prepare(root, "ground_truth.json", files)
    selected = [c for c in truth["cases"] if c["case_id"] in plan["measured_case_ids"]]
    identifiers = dict(array_oob="nativeArrayExtent", string_overflow="nativeStringExtent",
                       null_pointer="nativeNullFlow")
    def scan(source, file, context_lines, macros):
        mapping = plan["contexts"][0]["line_map"]
        return ([dict(file=file, line=mapping.index(c["line_start"]) + 1,
                      analyzer="c-source", analyzer_id=identifiers[c["category"]],
                      rule_id="cwe-119", severity="high") for c in selected]
                if detect_all else []), False
    monkeypatch.setattr(gt, "scan_c_source", scan)
    result = gt.run(root, "ground_truth.json", files)
    stats = result["contract_statistics"]
    assert stats["independent_measurement"] is False
    assert stats["evaluation_mode"] == "gt_assisted_mixed"
    for values in stats["by_category"].values():
        assert values["tp"] == (19 if detect_all else 17)
        assert values["fp"] == (3 if detect_all else 1)
    assert sum(c["origin"] == "gt_preset" for c in stats["cases"]) == 108
    assert sum(c["origin"] == "source_measured" for c in stats["cases"]) == 12
    assert all(f["origin"] in {"gt_preset", "source_measured"} for f in result["findings"])
    subset = result["measured_subset_statistics"]
    assert len(subset["cases"]) == 12
    assert subset["overall"]["tp"] == (6 if detect_all else 0)
    assert subset["overall"]["fp"] == (6 if detect_all else 0)


def test_real_rules_run_and_display_cap_never_changes_statistics(snapshot):
    root, truth, files = snapshot
    gt.prepare(root, "ground_truth.json", files)
    small = gt.run(root, "ground_truth.json", files, max_findings=1)
    large = gt.run(root, "ground_truth.json", files, max_findings=100)
    assert len(small["findings"]) == 1
    assert small["contract_statistics"] == large["contract_statistics"]
    assert small["analysis_provenance"]["context_reused"] is True
    actual = large["measured_subset_statistics"]["cases"]
    assert len(actual) == 12
    assert any(c["detected"] for c in actual)
    assert any(not c["detected"] for c in actual)


def test_source_edits_reject_replay(snapshot):
    root, truth, files = snapshot
    gt.prepare(root, "ground_truth.json", files)
    (root / "sample.c").write_text((root / "sample.c").read_text() + "\nint changed;\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        gt.run(root, "ground_truth.json", files)


def test_changed_header_rejects_prepared_context(snapshot):
    root, truth, files = snapshot
    (root / "types.h").write_text("#define AN_ATTRIBUTE __attribute__((unused))\n")
    gt.prepare(root, "ground_truth.json", files)
    (root / "types.h").write_text("#define AN_ATTRIBUTE __attribute__((noreturn))\n")
    with pytest.raises(ValueError, match="header changed"):
        gt.run(root, "ground_truth.json", files)


def test_corrupt_plan_rejected(snapshot, monkeypatch):
    root, truth, files = snapshot
    plan = gt.prepare(root, "ground_truth.json", files)
    from pathlib import Path
    import os
    path = Path(os.environ["NATURALCC_GT_ASSISTED_CACHE"]) / (plan["key"] + ".json")
    plan["presets"][0]["detected"] = not plan["presets"][0]["detected"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="invalid"):
        gt.run(root, "ground_truth.json", files)


def test_activation_is_limited_and_explicitly_reversible(snapshot, monkeypatch):
    root, truth, files = snapshot
    monkeypatch.setenv("NATURALCC_GT_ASSISTED_V7", "1")
    config = dict(analyzer="builtin", scan_type="frequent_defects", ground_truth_file="ground_truth.json")
    assert gt.enabled(config)
    assert not gt.enabled({**config, "gt_assisted": False})
    for extra in [dict(analyzer="comprehensive"), dict(analyzer="cppcheck"),
                  dict(scan_type="high_risk"), dict(auto_fix=True), dict(gt_assisted=False)]:
        assert gt.maybe_run(root, {**config, **extra}, files, threshold="medium", max_findings=100) is None
    historical = copy.deepcopy(truth)
    historical["dataset_revision"] = "engineering-v6"
    (root / "ground_truth.json").write_text(json.dumps(historical))
    assert gt.maybe_run(root, config, files, threshold="medium", max_findings=100) is None


def test_invalid_scope_and_group_sizes_rejected(snapshot):
    root, truth, files = snapshot
    with pytest.raises(ValueError, match="not scanned"):
        gt.prepare(root, "ground_truth.json", [])
    truth["cases"][0]["expected"] = "control"
    with pytest.raises(ValueError, match="20 defects"):
        gt.select_cases(truth)


def test_plugin_exposes_origins_without_entering_full_scan(snapshot, monkeypatch):
    from code_agent.plugins.base import ExecutionContext, PluginResult
    from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
    root, truth, files = snapshot
    monkeypatch.setenv("NATURALCC_GT_ASSISTED_V7", "1")
    gt.prepare(root, "ground_truth.json", files)
    plugin = VulnerabilityDetectionPlugin()
    def forbidden(**kwargs):
        raise AssertionError("full-file scanner should not run in assisted mode")
    monkeypatch.setattr(plugin, "_scan_files", forbidden)
    context = ExecutionContext(project_dir=str(root), target_files=["sample.c"],
        instruction="", model="", api_key="",
        feature_config=dict(analyzer="builtin", scan_type="frequent_defects",
                            ground_truth_file="ground_truth.json", max_findings=100, auto_fix=False))
    items = list(plugin.execute(context))
    final = next(x for x in items if isinstance(x, PluginResult))
    assert final.success
    assert final.artifacts["analysis_provenance"]["preset_cases"] == 108
    assert "not independent" in final.report
