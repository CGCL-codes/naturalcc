from pathlib import Path

import pytest

from code_agent.agent_core.compaction import (
    CompactionService,
    CompactionValidationError,
    FrozenCompactionSource,
    FrozenThreadCompactionSource,
)
from code_agent.agent_core.compaction_prompts import (
    ANALYZER_PROMPT_VERSION,
    CHECKPOINT_SCHEMA,
    COMPACTION_ANALYSIS_SCHEMA,
    SUMMARIZER_PROMPT_VERSION,
)
from code_agent.agent_core.contracts import ModelResponse
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import (
    DeepSeekRequestSerializer,
    ModelGateway,
    ScriptedModelGateway,
)
from code_agent.agent_core.token_budget import (
    DeepSeekModelProfile,
    DeepSeekTokenCounter,
)


def _analysis(covered_to: int = 1):
    return {
        "covered_range": {"from": 0, "to": covered_to},
        "active_goals": [],
        "user_constraints": [],
        "decisions": [],
        "completed_actions": [],
        "changed_files": [],
        "symbols_touched": [],
        "tool_findings": [],
        "failed_attempts": [],
        "verification_results": [],
        "unresolved_questions": [],
        "next_steps": [],
        "contradictions": [],
        "discarded_noise": [],
    }


def _checkpoint(covered_to: int = 1):
    return {
        "version": 1,
        "covered_range": {"from": 0, "to": covered_to},
        "task_objective": "Keep the API stable while fixing the bug.",
        "must_preserve": ["Public API compatibility"],
        "decisions": [],
        "current_repository_state": [],
        "completed_work": [],
        "failed_approaches": [],
        "verification_state": [],
        "open_work": ["Run tests"],
        "important_artifacts": [],
        "source_refs": ["message:0"],
        "repository_revision": None,
    }


def _service(tmp_path: Path, responses: list[ModelResponse]):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Compaction", workspace=tmp_path)
    store.create_run("run-1", str(tmp_path), "fix bug", {}, thread_id)
    messages = [
        {"role": "user", "content": "Keep the API stable."},
        {"role": "assistant", "content": "I will inspect it."},
        {"role": "user", "content": "Recent tail"},
    ]
    store.save_snapshot(
        "run-1",
        {
            "status": "running",
            "messages": messages,
            "active_run_checkpoint_id": None,
            "compacted_message_to": -1,
        },
        expected_version=0,
    )
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    gateway = ScriptedModelGateway(responses)
    service = CompactionService(
        store=store,
        gateway=gateway,
        counter=DeepSeekTokenCounter.from_directory(tokenizer_root),
        serializer=DeepSeekRequestSerializer(),
        profile=DeepSeekModelProfile(context_window_tokens=16_384),
    )
    source = FrozenCompactionSource.freeze(
        run_id="run-1",
        thread_id=thread_id,
        source_snapshot_version=0,
        covered_from=0,
        covered_to=1,
        recent_tail_from=2,
        messages=messages,
        working_state={"current_objective": "fix bug", "verification": {}},
        old_checkpoint=None,
    )
    return store, gateway, service, source


def test_prompt_contracts_are_versioned_and_require_coverage():
    assert ANALYZER_PROMPT_VERSION.endswith("v1")
    assert SUMMARIZER_PROMPT_VERSION.endswith("v1")
    assert "covered_range" in COMPACTION_ANALYSIS_SCHEMA["required"]
    assert {
        "task_objective",
        "must_preserve",
        "open_work",
        "source_refs",
        "repository_revision",
    } <= set(CHECKPOINT_SCHEMA["required"])


def test_compaction_service_runs_analysis_then_summary_and_commits(tmp_path: Path):
    store, gateway, service, source = _service(
        tmp_path,
        [
            ModelResponse(
                content=__import__("json").dumps(_analysis()),
                input_tokens=120,
                output_tokens=40,
            ),
            ModelResponse(
                content=__import__("json").dumps(_checkpoint()),
                input_tokens=80,
                output_tokens=30,
            ),
        ],
    )

    outcome = service.compact_run(source)

    assert [request.purpose for request in gateway.requests] == [
        "compaction_analysis",
        "compaction_summary",
    ]
    assert all(request.tools == [] for request in gateway.requests)
    assert all(
        request.response_format == {"type": "json_object"}
        for request in gateway.requests
    )
    assert gateway.requests[0].max_output_tokens == 4096
    assert gateway.requests[1].max_output_tokens == 2048
    assert outcome.maintenance_calls == 2
    assert outcome.input_tokens == 200
    assert outcome.output_tokens == 70
    assert outcome.checkpoint["task_objective"].startswith("Keep the API")
    assert store.get_compaction(outcome.compaction_id)["status"] == "committed"
    assert store.load_snapshot("run-1")["active_run_checkpoint_id"] == (
        outcome.compaction_id
    )


def test_invalid_analysis_json_gets_one_bounded_repair(tmp_path: Path):
    _, gateway, service, source = _service(
        tmp_path,
        [
            ModelResponse(content="not json"),
            ModelResponse(content=__import__("json").dumps(_analysis())),
            ModelResponse(content=__import__("json").dumps(_checkpoint())),
        ],
    )

    outcome = service.compact_run(source)

    assert outcome.maintenance_calls == 3
    assert [request.purpose for request in gateway.requests] == [
        "compaction_analysis",
        "compaction_json_repair",
        "compaction_summary",
    ]
    assert gateway.requests[1].tools == []


def test_compaction_never_exceeds_explicit_maintenance_call_allowance(
    tmp_path: Path,
):
    _, gateway, service, source = _service(
        tmp_path,
        [
            ModelResponse(content="not json"),
            ModelResponse(content=__import__("json").dumps(_analysis())),
            ModelResponse(content=__import__("json").dumps(_checkpoint())),
        ],
    )

    outcome = service.compact_run(source, max_maintenance_calls=2)

    assert outcome.maintenance_calls == 2
    assert outcome.fallback_used is True
    assert [request.purpose for request in gateway.requests] == [
        "compaction_analysis",
        "compaction_json_repair",
    ]


class _FailingGateway(ModelGateway):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        raise RuntimeError("provider unavailable")


def test_provider_failure_commits_deterministic_fallback(tmp_path: Path):
    store, _, service, source = _service(tmp_path, [])
    gateway = _FailingGateway()
    service.gateway = gateway

    outcome = service.compact_run(source)

    assert outcome.fallback_used is True
    assert outcome.maintenance_calls == 1
    assert outcome.checkpoint["task_objective"] == "fix bug"
    assert "message:0" in outcome.checkpoint["source_refs"]
    assert store.get_compaction(outcome.compaction_id)["status"] == "committed"


def test_run_fallback_preserves_previous_checkpoint_and_recent_facts(tmp_path: Path):
    store, _, service, source = _service(tmp_path, [])
    old_checkpoint = _checkpoint()
    old_checkpoint["must_preserve"] = [f"old-{index}" for index in range(20)]
    old_checkpoint["decisions"] = ["keep old decision"]
    source = FrozenCompactionSource.freeze(
        run_id=source.run_id,
        thread_id=source.thread_id,
        source_snapshot_version=source.source_snapshot_version,
        covered_from=source.covered_from,
        covered_to=source.covered_to,
        recent_tail_from=source.recent_tail_from,
        messages=source.covered_messages,
        working_state={
            **source.working_state,
            "decisions": ["new decision"],
        },
        old_checkpoint=old_checkpoint,
    )
    service.gateway = _FailingGateway()

    outcome = service.compact_run(source)

    assert outcome.checkpoint["version"] == 1
    assert "old-0" in outcome.checkpoint["must_preserve"]
    assert any("Keep the API stable" in item for item in outcome.checkpoint["must_preserve"])
    assert outcome.checkpoint["decisions"] == ["keep old decision", "new decision"]


class _MutatingGateway(ScriptedModelGateway):
    def __init__(self, store: EventStore):
        super().__init__(
            [
                ModelResponse(content=__import__("json").dumps(_analysis())),
                ModelResponse(content=__import__("json").dumps(_checkpoint())),
            ]
        )
        self.store = store

    def generate(self, request):
        response = super().generate(request)
        if request.purpose == "compaction_summary":
            snapshot = self.store.load_snapshot("run-1")
            snapshot["messages"][0]["content"] = "mutated during compaction"
            self.store.save_snapshot("run-1", snapshot, expected_version=0)
        return response


def test_frozen_source_mutation_rejects_checkpoint_commit(tmp_path: Path):
    store, _, service, source = _service(tmp_path, [])
    service.gateway = _MutatingGateway(store)

    with pytest.raises(CompactionValidationError, match="prefix changed"):
        service.compact_run(source)

    with store._connect() as connection:
        row = connection.execute(
            "SELECT status, error_json FROM compactions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert row["status"] == "failed"
    assert "SourceMutation" in row["error_json"]


def _incomplete_record(store: EventStore, source: FrozenCompactionSource):
    return store.create_compaction(
        scope="run",
        thread_id=source.thread_id,
        run_id=source.run_id,
        covered_from=source.covered_from,
        covered_to=source.covered_to,
        source_version=source.source_snapshot_version,
        source_hash=source.source_hash,
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version=ANALYZER_PROMPT_VERSION,
        summarizer_prompt_version=SUMMARIZER_PROMPT_VERSION,
    )


def test_resume_analyzing_compaction_reuses_record(tmp_path: Path):
    store, gateway, service, source = _service(
        tmp_path,
        [
            ModelResponse(content=__import__("json").dumps(_analysis())),
            ModelResponse(content=__import__("json").dumps(_checkpoint())),
        ],
    )
    record = _incomplete_record(store, source)

    outcome = service.resume_run(record["id"], source)

    assert outcome.compaction_id == record["id"]
    assert outcome.maintenance_calls == 2
    assert [request.purpose for request in gateway.requests] == [
        "compaction_analysis",
        "compaction_summary",
    ]
    assert store.get_compaction(record["id"])["status"] == "committed"


def test_resume_summarizing_skips_completed_analysis(tmp_path: Path):
    store, gateway, service, source = _service(
        tmp_path,
        [ModelResponse(content=__import__("json").dumps(_checkpoint()))],
    )
    record = _incomplete_record(store, source)
    store.update_compaction(
        record["id"], status="summarizing", analysis=_analysis()
    )

    outcome = service.resume_run(record["id"], source)

    assert outcome.maintenance_calls == 1
    assert [request.purpose for request in gateway.requests] == [
        "compaction_summary"
    ]
    assert store.get_compaction(record["id"])["status"] == "committed"


def test_resume_validating_commits_without_model_call(tmp_path: Path):
    store, gateway, service, source = _service(tmp_path, [])
    record = _incomplete_record(store, source)
    store.update_compaction(
        record["id"], status="summarizing", analysis=_analysis()
    )
    store.update_compaction(
        record["id"], status="validating", checkpoint=_checkpoint()
    )

    outcome = service.resume_run(record["id"], source)

    assert outcome.maintenance_calls == 0
    assert gateway.requests == []
    assert store.get_compaction(record["id"])["status"] == "committed"


def test_thread_compaction_commits_sequence_watermark(tmp_path: Path):
    analysis = _analysis(covered_to=2)
    analysis["covered_range"]["from"] = 1
    checkpoint = _checkpoint(covered_to=2)
    checkpoint["covered_range"]["from"] = 1
    store, gateway, service, _ = _service(
        tmp_path,
        [
            ModelResponse(content=__import__("json").dumps(analysis)),
            ModelResponse(content=__import__("json").dumps(checkpoint)),
        ],
    )
    thread_id = store.create_thread("Thread history", workspace=tmp_path)
    for index in range(1, 4):
        store.append_conversation_message(
            thread_id,
            role="user" if index % 2 else "assistant",
            content=f"conversation-{index}",
        )
    messages = store.list_conversation_messages(thread_id)
    source = FrozenThreadCompactionSource.freeze(
        thread_id=thread_id,
        source_thread_version=store.get_thread(thread_id)["version"],
        covered_messages=messages[:2],
        working_state={"current_objective": "continue"},
        old_checkpoint=None,
    )

    outcome = service.compact_thread(source)

    assert outcome.maintenance_calls == 2
    assert [request.purpose for request in gateway.requests] == [
        "compaction_analysis",
        "compaction_summary",
    ]
    thread = store.get_thread(thread_id)
    assert thread["active_thread_checkpoint_id"] == outcome.compaction_id
    assert thread["checkpoint_covered_sequence"] == 2


def test_thread_compaction_provider_failure_uses_fallback(tmp_path: Path):
    store, _, service, _ = _service(tmp_path, [])
    service.gateway = _FailingGateway()
    thread_id = store.create_thread("Thread fallback", workspace=tmp_path)
    store.append_conversation_message(
        thread_id,
        role="user",
        content="Preserve the public API.",
    )
    messages = store.list_conversation_messages(thread_id)
    source = FrozenThreadCompactionSource.freeze(
        thread_id=thread_id,
        source_thread_version=store.get_thread(thread_id)["version"],
        covered_messages=messages,
        working_state={"current_objective": "fix bug"},
        old_checkpoint=None,
    )

    outcome = service.compact_thread(source)

    assert outcome.fallback_used is True
    assert outcome.checkpoint["task_objective"] == "fix bug"
    assert "Preserve the public API." in outcome.checkpoint["must_preserve"]
    assert store.get_thread(thread_id)["checkpoint_covered_sequence"] == 1
