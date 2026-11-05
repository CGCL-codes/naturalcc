"""The committed Contract-3 ground truth must load with the product loader as-is."""
import json
import shutil
from pathlib import Path

import pytest

from code_agent.scripts import benchmark_contract3_independent as independent
from code_agent.scripts.export_contract3_ground_truth import SUITES
from code_agent.security_contracts import load_ground_truth

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "artifacts" / "contract3" / "ground_truth"
MANIFEST = json.loads((ROOT / "artifacts" / "contract3" / "manifest.json").read_text(encoding="utf-8"))
PROJECTS = [spec["id"] for spec in MANIFEST["sources"]]


@pytest.mark.parametrize("suite", SUITES)
@pytest.mark.parametrize("project", PROJECTS)
def test_ground_truth_loads_with_product_loader(tmp_path, suite, project):
    path = GT / suite / project / "ground_truth.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["scan_type"] == "frequent_defects"
    # Regenerate the fixtures exactly as the exporter and Web runner name them.
    for category in MANIFEST["categories"]:
        independent.inject_fixture(tmp_path, project, [category],
            fixture_name=f"contract3_{suite.replace('-', '_')}_web_{category['id']}.c", profile=suite)
    shutil.copy(path, tmp_path / "ground_truth.json")
    scanned = [str(tmp_path / file) for file in data["source_sha256"]]
    loaded = load_ground_truth(tmp_path, "ground_truth.json", "frequent_defects", scanned)
    cases = loaded["cases"]
    assert len(cases) == 120 and sum(case["expected"] == "defect" for case in cases) == 60
    assert {case["file"] for case in cases} == set(data["source_sha256"])
    assert data["line_totals"]["upstream_detection_files"] >= 10_000
    assert len(data["detection_files"]) == len({item["path"] for item in data["detection_files"]})
