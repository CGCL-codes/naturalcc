import json

import pytest

from code_agent.scripts import benchmark_contract5_web as bench


def test_frozen_dataset_and_order():
    _, samples = bench.load_samples()
    assert len(samples) == 40
    assert [s["id"] for s in samples[:8]] == [
        "787-1", "362-1", "401-1", "78-1", "787-7", "362-7", "401-7", "78-7"]
    assert sum(s["vulnerable"] for s in samples) == 24
    assert all(bench.digest(s["code"].encode("utf-8")) == s["sha256"] for s in samples)


def test_metrics_do_not_treat_wrong_category_as_target_hit():
    prediction = {"findings": [{"cwe": 78}]}
    rows = [({"vulnerable": True, "cwe": 787}, prediction),
            ({"vulnerable": False, "cwe": 78}, prediction),
            ({"vulnerable": True, "cwe": 401}, {"findings": []}),
            ({"vulnerable": False, "cwe": 362}, {"findings": []})]
    assert bench.metrics(rows) == dict(n=4, tp=1, tn=1, fp=1, fn=1,
                                      target_hits=0, strict_correct=1)
    assert bench.rate(0, 0).startswith("N/A")


def test_parser_never_defaults_bad_output_to_safe():
    assert bench.parse_answer('```json\n{"findings":[],"summary":"safe"}\n```', 10)["findings"] == []
    assert bench.parse_answer('Explanation\n```json\n{"findings":[],"summary":"safe"}\n```', 10)["findings"] == []
    for answer in ('no bugs', '{}', '{"findings":[],"summary":null}',
                   '{"findings":[{"cwe":787,"line":11,"reason":"x"}],"summary":"x"}'):
        with pytest.raises(ValueError):
            bench.parse_answer(answer, 10)


def test_offline_rescore_keeps_same_run(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/run.json").write_text(json.dumps({"state": {
        "status": "completed", "final_answer": 'Explanation\n```json\n{"findings":[],"summary":"safe"}\n```'}}))
    monkeypatch.setattr(bench, "DATA", tmp_path)
    monkeypatch.setattr(bench, "OUT", tmp_path)
    checkpoint = {"samples": {"a": {"status": "unscored", "attempts": [{"run_id": "run"}]}}}
    bench.refresh_scores(checkpoint, [{"id": "a", "name": "a.c", "code": "int main() {}"}])
    assert checkpoint["samples"]["a"]["status"] == "scored"
    assert checkpoint["samples"]["a"]["attempts"] == [{"run_id": "run"}]


def test_credit_failure_only_uses_runtime_errors():
    assert bench.credit_failure([{"type": "run.failed", "payload": {
        "error": {"message": "Error code: 402 - insufficient credits"}}}])
    assert not bench.credit_failure([{"type": "model.responded", "payload": {
        "error": {"message": "402"}}}])
    assert "在途请求" in bench.credit_stop_reason([{"type": "run.failed", "payload": {
        "error": {"message": "402 in_flight_budget_exhausted"}}}])


def test_resume_skips_scored_and_preserves_credit_failed_attempt(tmp_path, monkeypatch):
    sample = {"id": "787-1", "name": "a.c", "cwe": 787, "vulnerable": True,
              "code": "int main() {}\n"}
    (tmp_path / "manifest.json").write_text("{}")
    config = {"provider": "openrouter", "model": bench.MODEL,
              "base_url": "https://openrouter.ai/api/v1"}
    calls = []

    class FakeWeb:
        def __init__(self, base):
            pass

        def call(self, method, path, **kwargs):
            calls.append(path)
            if path == "/api/bootstrap":
                return {"runtime_default_model_config": config}
            if path == "/api/agent/threads":
                assert "vulnerable" not in json.dumps(kwargs)
                return {"thread_id": "new-thread"}
            if path.endswith("/messages"):
                return {"run_id": "new-run"}
            if path.endswith("/events"):
                return {"events": [{"type": "run.failed", "payload": {
                    "error": {"message": "402 insufficient credits"}}}]}
            return {"status": "failed", "runtime_model_config": config}

    monkeypatch.setattr(bench, "DATA", tmp_path)
    monkeypatch.setattr(bench, "OUT", tmp_path / "out")
    monkeypatch.setattr(bench, "WebClient", FakeWeb)
    monkeypatch.setattr(bench, "report", lambda *args: None)
    monkeypatch.setattr(bench, "check_credits", lambda: {"effective_remaining_usd": 0.01})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    checkpoint = {"samples": {sample["id"]: {"status": "credit_exhausted", "attempts": [
        {"thread_id": "old-thread", "run_id": "old-run"}]}}, "credits": []}
    bench.run(checkpoint, {}, [sample], "http://127.0.0.1:7860")
    rec = checkpoint["samples"][sample["id"]]
    assert len(rec["attempts"]) == 2
    assert rec["status"] == "credit_exhausted"
    assert "prediction" not in rec
    assert "test-secret" not in (tmp_path / "out/checkpoint.json").read_text()
    rec["prediction"] = {"findings": []}
    rec["status"] = "scored"
    calls.clear()
    bench.run(checkpoint, {}, [sample], "http://127.0.0.1:7860")
    assert calls == ["/api/bootstrap"]
    assert len(rec["attempts"]) == 2


def test_client_refuses_external_backend():
    with pytest.raises(ValueError):
        bench.WebClient("https://example.com")


def test_report_needs_no_exported_sources_or_run_logs(tmp_path, monkeypatch):
    for name in ("dataset.json", "manifest.json"):
        (tmp_path / name).write_bytes((bench.DATA / name).read_bytes())
    checkpoint = json.loads((bench.OUT / "checkpoint.json").read_text())
    original = json.dumps(checkpoint, sort_keys=True)
    monkeypatch.setattr(bench, "DATA", tmp_path)
    monkeypatch.setattr(bench, "OUT", tmp_path / "web-sonnet45")
    monkeypatch.setattr(bench, "REPORT", tmp_path / "report.md")
    manifest, samples = bench.load_samples()
    bench.refresh_scores(checkpoint, samples)
    bench.report(checkpoint, manifest, samples)
    report = bench.REPORT.read_text()
    assert "92.50% (37/40)" in report
    assert report.count("Agent 最终结构化结论：") == 4
    assert "artifacts/contract5/samples/" not in report
    assert "web-sonnet45/runs/" not in report
    assert json.dumps(checkpoint, sort_keys=True) == original
