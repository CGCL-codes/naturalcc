from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_existing_pipeline_route_is_preserved():
    source = (ROOT / "agent_web_api.py").read_text(encoding="utf-8")
    assert '@app.post("/api/run")' in source
    assert '@app.post("/api/prompt/preview")' in source


def test_aider_does_not_auto_commit():
    source = (ROOT / "aider_runner.py").read_text(encoding="utf-8")
    assert '"--no-auto-commits"' in source


def test_all_six_existing_plugins_remain_present():
    plugin_names = {
        "code_completion.py",
        "code_summary.py",
        "code_repair.py",
        "vulnerability_detection.py",
        "design_to_code.py",
        "knowledge_graph.py",
    }
    assert plugin_names.issubset({path.name for path in (ROOT / "plugins").glob("*.py")})
