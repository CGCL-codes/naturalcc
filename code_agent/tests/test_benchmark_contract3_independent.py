from code_agent.scripts import benchmark_contract3_cppcheck as base
from code_agent.scripts import benchmark_contract3_independent as independent


def test_independent_fixture_has_labels_only_in_truth_metadata(tmp_path):
    manifest = base.load_manifest()
    fixture, cases = independent.inject_fixture(tmp_path, "fixture", manifest["categories"])
    content = fixture.read_text(encoding="utf-8")
    assert len(cases) == 120
    assert "positive" not in content
    assert "control" not in content
    for category in manifest["categories"]:
        rows = [case for case in cases if case["category"] == category["id"]]
        assert sum(case["expected"] == "defect" for case in rows) == 20
        assert sum(case["expected"] == "control" for case in rows) == 20


def test_independent_examples_have_multiple_code_shapes():
    first = independent.case_lines("array_oob", 0, True)
    loop = independent.case_lines("array_oob", 2, True)
    helper = independent.case_lines("null_pointer", 2, True)
    assert "values[3]" in "\n".join(first)
    assert "for (" in "\n".join(loop)
    assert "contract3_read(0)" in "\n".join(helper)


def test_balanced_profile_has_eighteen_standard_and_two_harder_positive_shapes(tmp_path):
    manifest = base.load_manifest()
    _, cases = independent.inject_fixture(tmp_path, "fixture", manifest["categories"], profile="balanced")
    for category in manifest["categories"]:
        rows = [case for case in cases if case["category"] == category["id"] and case["expected"] == "defect"]
        assert len(rows) == 20
        assert independent.BALANCED_VARIANTS[category["id"]].count(4 if category["id"] != "null_pointer" else 2) >= 1
