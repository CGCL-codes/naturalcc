from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Event

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import AgentEvent, ModelResponse, ToolCall, ToolResult
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.memory_proposals import (
    MemoryProposalGenerationError,
    MemoryProposalService,
    ProposalValidationError,
)
from code_agent.agent_core.memory_store import MemoryProposalConflict, MemoryStore
from code_agent.agent_core.memory_tool_recovery import find_tool_argument_recoveries
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.api.agent_routes import create_agent_router


def _analysis_response(message_id: str = "message-1") -> ModelResponse:
    return ModelResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "claim": "Keep existing REST APIs backward compatible.",
                        "evidence_refs": [f"conversation:{message_id}"],
                        "durable": True,
                    }
                ],
                "contradictions": [],
                "sensitive_findings": [],
                "recommended_scope": "project",
                "recommended_kind": "project_constraint",
                "abstain": False,
                "abstain_reason": None,
            }
        )
    )


def _proposal_response(message_id: str = "message-1", **overrides) -> ModelResponse:
    return ModelResponse(
        content=json.dumps(
            {
                "proposals": [
                    {
                        "schema_version": 1,
                        "operation": "create",
                        "target_memory_id": None,
                        "scope": "project",
                        "kind": "project_constraint",
                        "subject": "REST API compatibility",
                        "canonical_content": "Changes must preserve existing REST API behavior.",
                        "evidence_refs": [f"conversation:{message_id}"],
                        "verification": "user_asserted",
                        "durability": "long_term",
                        "expires_at": None,
                        "confidence": 0.98,
                        "conflicts_with": [],
                        "sensitive": False,
                        "abstained": False,
                        "abstain_reason": None,
                        **overrides,
                    }
                ]
            }
        )
    )


def _prepared_service(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Memory review", workspace=tmp_path)
    message = events.append_conversation_message(
        thread_id,
        role="user",
        content="All changes must preserve existing REST APIs.",
        message_id="message-1",
    )
    memories = MemoryStore(tmp_path / "agent.db")
    gateway = ScriptedModelGateway([_analysis_response(), _proposal_response()])
    service = MemoryProposalService(
        event_store=events,
        memory_store=memories,
        gateway=gateway,
    )
    return events, memories, gateway, service, thread_id, message


@pytest.mark.parametrize("mode", ["off", "review", "auto"])
def test_completed_run_extracts_memory_in_background(tmp_path: Path, mode: str):
    events = EventStore(tmp_path / "agent.db")
    memories = MemoryStore(events.db_path)
    thread_id = events.create_thread("Auto memory", workspace=tmp_path)
    events.append_conversation_message(thread_id, role="user", content="Earlier turn, out of scope.")
    analyzing = Event()
    finish_analysis = Event()

    class Gateway(ScriptedModelGateway):
        def generate(self, request):
            self.requests.append(request)
            if request.purpose == "memory_proposal":
                analyzing.set()
                assert finish_analysis.wait(5)
                payload = json.loads(request.messages[1]["content"])
                assert [item["role"] for item in payload["frozen_evidence"]] == ["user"]
                assert payload["analysis"]["mode"] == "explicit_user_guidance"
                assert "Earlier turn" not in payload["frozen_evidence"][0]["content"]
                assert request.metadata["api_key"] == "request-key"
                user_ref = payload["allowed_evidence_refs"][0]
                assert request.tools == []
                return _proposal_response(user_ref.removeprefix("conversation:"))
            return ModelResponse(content="Understood.")

    gateway = Gateway([])
    engine = RunEngine(events, build_default_registry(include_mutating=False), gateway, memory_store=memories)
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories, auto_memory_mode=mode))
    client = TestClient(app)
    created = client.post(
        f"/api/agent/threads/{thread_id}/messages",
        json={"content": "All changes must preserve existing REST APIs.", "api_key": "request-key"},
    )
    assert created.status_code == 200, created.text
    run_id = created.json()["run_id"]
    response = client.post(f"/api/agent/runs/{run_id}/run")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    outcome_url = f"/api/agent/runs/{run_id}/memory-extraction"
    project_id = str(tmp_path.resolve())
    if mode == "off":
        assert engine.auto_memory is None
        assert len(gateway.requests) == 1
        assert client.get(outcome_url).json()["status"] == "disabled"
        return

    try:
        assert analyzing.wait(5)
        assert client.get(outcome_url).json()["status"] == "running"
        assert memories.list_proposals(project_id=project_id) == []
    finally:
        finish_analysis.set()
    engine.auto_memory._pending.join()
    proposals = memories.list_proposals(project_id=project_id)
    assert len(proposals) == 1
    outcome = client.get(outcome_url).json()
    assert outcome["status"] == "completed"
    assert outcome["proposal_count"] == 1
    assert outcome["applied_count"] == int(mode == "auto")
    proposal = proposals[0]
    assert proposal["source_mode"] == "automatic_run"
    assert proposal["run_id"] == run_id
    if mode == "review":
        assert proposal["status"] == "review_ready"
        assert memories.list_pinned_for_context(project_id=project_id) == []
        response = client.post(
            f"/api/agent/memory-proposals/{proposal['id']}/approve",
            json={"expected_version": proposal["version"]},
        )
        assert response.status_code == 200, response.text
    else:
        assert proposal["status"] == "applied"
        assert memories.list_review_actions(proposal["id"])[0]["actor"] == "auto_memory"
    assert len(memories.list_pinned_for_context(project_id=project_id)) == 1
    assert memories.list_pinned_for_context(project_id=str(tmp_path / "other")) == []

    # Replaying a terminal Run does not re-extract. A second Run with the same
    # candidate is deduplicated against the active memory in SQLite.
    assert client.post(f"/api/agent/runs/{run_id}/step").status_code == 200
    assert len(gateway.requests) == 2
    duplicate = client.post(
        f"/api/agent/threads/{thread_id}/messages",
        json={"content": "All changes must preserve existing REST APIs.", "api_key": "request-key"},
    )
    assert duplicate.status_code == 200, duplicate.text
    assert client.post(f"/api/agent/runs/{duplicate.json()['run_id']}/run").status_code == 200
    engine.auto_memory._pending.join()
    assert len(gateway.requests) == 4
    assert len(memories.list_proposals(project_id=project_id)) == 1


@pytest.mark.parametrize(
    ("goal", "proposal_content", "expected_status", "expected_reason"),
    [
        ("更换编译环境后仍用了旧依赖，重试前要先处理哪个目录？", None, "skipped", "no_durable_evidence"),
        ("这个仓库的长期规则是什么？", None, "skipped", "no_durable_evidence"),
        ("请复述之前保存的长期发布规则。", None, "skipped", "no_durable_evidence"),
        ("请保存本仓库的长期规则。", '{"proposals":[]}', "abstained", "composer_abstained"),
        ("请记录本仓库的长期排障经验。", "invalid", "failed", "ProposalValidationError"),
    ],
)
def test_run_reports_extraction_outcome(tmp_path, goal, proposal_content, expected_status, expected_reason):
    events = EventStore(tmp_path / "agent.db")
    memories = MemoryStore(events.db_path)
    thread_id = events.create_thread("Memory audit", workspace=tmp_path)
    responses = [ModelResponse(content="已有记忆说 NCC-Q9V6，签名检查必须在打包前。")]
    if proposal_content is not None:
        responses.extend([ModelResponse(content=proposal_content)] * 2)
    gateway = ScriptedModelGateway(responses)
    engine = RunEngine(events, build_default_registry(include_mutating=False), gateway, memory_store=memories)
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories, auto_memory_mode="review"))
    client = TestClient(app)
    run = client.post(f"/api/agent/threads/{thread_id}/messages", json={"content": goal}).json()
    run_id = run["run_id"]
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "completed"
    engine.auto_memory._pending.join()
    outcome = client.get(f"/api/agent/runs/{run_id}/memory-extraction").json()
    assert outcome["status"] == expected_status
    assert outcome["reason"] == expected_reason
    assert outcome["proposal_count"] == 0
    assert not memories.list_proposals()
    if proposal_content is None:
        assert [request.purpose for request in gateway.requests] == ["agent"]


def test_verified_run_cannot_promote_a_question_as_new_evidence(tmp_path):
    events = EventStore(tmp_path / "agent.db")
    memories = MemoryStore(events.db_path)
    thread_id = events.create_thread("Verified repair", workspace=tmp_path)
    events.create_run("verified", str(tmp_path), "Find what changed", {}, thread_id)
    events.append_conversation_message(
        thread_id, role="user", content="这个仓库的发布代号是什么？",
        run_id="verified", message_id="question",
    )
    result = events.append_event(
        "verified", "verification.finished", {"passed": True, "tool": "tests.run"}
    )
    events.append_event("verified", "run.completed", {}, status="completed")
    unsupported = _proposal_response("question")
    gateway = ScriptedModelGateway([_analysis_response("question"), unsupported, unsupported])
    service = MemoryProposalService(event_store=events, memory_store=memories, gateway=gateway)
    with pytest.raises(MemoryProposalGenerationError, match="outside the frozen bundle"):
        service.extract_completed_run("verified", {"runtime_model_config": {}}, auto_apply=False)
    payload = json.loads(gateway.requests[1].messages[1]["content"])
    assert payload["allowed_evidence_refs"] == [f"event:verified:{result.sequence}"]
    assert not memories.list_proposals()


@pytest.mark.parametrize("mode", ["auto", "review"])
def test_ppt_run_learns_recovered_tool_error_without_memory_request(tmp_path, mode):
    events = EventStore(tmp_path / "agent.db")
    memories = MemoryStore(events.db_path)
    gateway = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("bad-1", "workspace.create_file", {})]),
        ModelResponse(tool_calls=[ToolCall("bad-2", "workspace.create_file", {})]),
        ModelResponse(tool_calls=[ToolCall("fixed", "workspace.create_file", {
            "path": "memory-slides.md", "content": "# Memory system",
        })]),
        ModelResponse(content="The slide outline is ready."),
        ModelResponse(content="Check path and content before calling workspace.create_file."),
    ])
    engine = RunEngine(events, build_default_registry(include_mutating=True), gateway, memory_store=memories)
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories, auto_memory_mode=mode))
    client = TestClient(app)

    def start(title, content):
        thread = client.post("/api/agent/threads", json={"title": title, "workspace": str(tmp_path)}).json()
        return client.post(f"/api/agent/threads/{thread['thread_id']}/messages", json={"content": content}).json()["run_id"]

    run_id = start("PPT regression", "帮我制作一份ppt,讲述你的memory系统，我希望你能熟悉本地工具")
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "waiting_approval"
    assert client.post(f"/api/agent/runs/{run_id}/approve", json={
        "risk": "write", "tool_call_id": "bad-1",
    }).status_code == 200
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "completed"
    assert (tmp_path / "memory-slides.md").is_file()
    finished = events.list_events(run_id, event_type="tool.finished")
    assert [event.payload["result"]["status"] for event in finished] == ["error", "error", "success"]
    verifications = events.list_events(run_id, event_type="verification.finished")
    assert [event.payload.get("kind") for event in verifications] == ["snapshot_or_diff"]
    engine.auto_memory._pending.join()
    outcome = client.get(f"/api/agent/runs/{run_id}/memory-extraction").json()
    assert outcome["status"] == "completed", outcome
    assert outcome["proposal_count"] == 1
    assert outcome["applied_count"] == int(mode == "auto")
    proposal, = memories.list_proposals(project_id=str(tmp_path.resolve()))
    assert proposal["kind"] == "successful_approach"
    assert proposal["verification"] == "tool_observed"
    assert {item["source_sequence"] for item in memories.get_proposal_evidence(proposal["id"])} == {
        event.sequence for event in finished
    }
    if mode == "review":
        assert memories.list(status="active") == []
        return

    memory, = memories.list(status="active", project_id=str(tmp_path.resolve()))
    assert "path" in memory["content"] and "content" in memory["content"]
    recall_run = start("Later session", "使用 workspace.create_file 前需要检查哪些参数？")
    assert client.post(f"/api/agent/runs/{recall_run}/run").json()["status"] == "completed"
    engine.auto_memory._pending.join()
    request, = events.list_events(recall_run, event_type="model.requested")
    assert request.payload["retrieved_memory_ids"] == [memory["id"]]
    assert len(memories.list_proposals()) == 1
    assert all(request.purpose == "agent" for request in gateway.requests)


@pytest.mark.parametrize(("failures", "recovery_tool", "recovery_args", "error_type"), [
    (1, "workspace.create_file", {"path": "x", "content": "y"}, "ValueError"),
    (2, None, {}, "ValueError"),
    (2, "workspace.read", {"path": "x", "content": "y"}, "ValueError"),
    (2, "workspace.create_file", {"path": "x"}, "ValueError"),
    (2, "workspace.create_file", {"path": "x", "content": "y"}, "CommandFailed"),
])
def test_unconfirmed_tool_recoveries_do_not_create_experiences(
    failures, recovery_tool, recovery_args, error_type
):
    events = []

    def add_call(tool, args, result):
        call_id = f"call-{len(events)}"
        events.append(AgentEvent("run", len(events) + 1, "tool.started", {
            "tool_call": ToolCall(call_id, tool, args).to_dict(),
        }, ""))
        events.append(AgentEvent("run", len(events) + 1, "tool.finished", {
            "tool_call_id": call_id, "tool": tool, "result": result.to_dict(),
        }, ""))

    for _ in range(failures):
        add_call("workspace.create_file", {}, ToolResult.failure(
            "missing required arguments: content, path", error_type
        ))
    if recovery_tool:
        add_call(recovery_tool, recovery_args, ToolResult.success("Finished"))
    assert find_tool_argument_recoveries(events) == []


def test_manual_evidence_becomes_visual_review_then_active_memory(tmp_path: Path):
    _, memories, gateway, service, thread_id, _ = _prepared_service(tmp_path)

    reviews = service.create_from_selection(
        thread_id=thread_id,
        project_id=str(tmp_path.resolve()),
        evidence_refs=[{"type": "conversation_message", "source_id": "message-1"}],
    )

    assert len(gateway.requests) == 2
    assert [request.purpose for request in gateway.requests] == [
        "memory_analysis",
        "memory_proposal",
    ]
    assert all(request.tools == [] for request in gateway.requests)
    for request in gateway.requests:
        if request.response_format == {"type": "json_object"}:
            prompt = "\n".join(str(message.get("content", "")) for message in request.messages)
            assert "json" in prompt.casefold()
    analyzer_payload = json.loads(gateway.requests[0].messages[1]["content"])
    composer_payload = json.loads(gateway.requests[1].messages[1]["content"])
    assert analyzer_payload["required_output_schema"]["required"]
    assert composer_payload["required_output_schema"]["required"] == ["proposals"]
    assert any(
        "model_inferred" in rule and "thread" in rule
        for rule in composer_payload["requirements"]["validation_rules"]
    )
    review = reviews[0]
    assert review["title"] == "REST API compatibility"
    assert review["scope_label"] == "仅当前项目"
    assert review["evidence"][0]["source_locator"] == "对话 #1"
    assert "proposal_json" not in review
    assert "raw_proposal" not in review
    assert memories.search("REST API", project_id=str(tmp_path.resolve())) == []

    edited = service.update_review(
        review["proposal_id"],
        expected_version=review["version"],
        subject="Public REST API compatibility",
    )
    assert edited["version"] == 1
    applied = service.approve(
        review["proposal_id"], expected_version=edited["version"]
    )
    assert applied["status"] == "applied"
    result = memories.search("REST API", project_id=str(tmp_path.resolve()))
    assert result[0]["subject"] == "Public REST API compatibility"
    assert result[0]["source_proposal_id"] == review["proposal_id"]
    assert [item["action"] for item in memories.list_review_actions(review["proposal_id"])] == [
        "edit",
        "approve",
    ]


def test_automatic_correction_preserves_conflict_review_and_expires_from_context(tmp_path: Path):
    events, memories, _, service, thread_id, _ = _prepared_service(tmp_path)
    project_id = str(tmp_path.resolve())
    original = service.create_from_selection(
        thread_id=thread_id,
        project_id=project_id,
        evidence_refs=[{"type": "conversation_message", "source_id": "message-1"}],
    )[0]
    service.approve(original["proposal_id"], expected_version=original["version"])
    previous = memories.list_pinned_for_context(project_id=project_id)[0]

    events.create_run("correction", project_id, "API migration policy", {}, thread_id)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="Correction to the standing policy: API behavior may change after an agreed deprecation period.",
        run_id="correction",
        message_id="correction-message",
    )
    events.append_event("correction", "run.completed", {}, status="completed")
    metadata = {"runtime_model_config": {"provider": "deepseek", "model": "deepseek-chat"}}
    revised = "REST API behavior may change after the agreed deprecation period."
    service.gateway = ScriptedModelGateway([
        _proposal_response("correction-message", canonical_content=revised),
        _proposal_response(
            "correction-message",
            operation="supersede",
            target_memory_id=previous["id"],
            canonical_content=revised,
        ),
    ])

    service.extract_completed_run("correction", metadata, auto_apply=True)
    pending = memories.list_proposals(status="review_ready", project_id=project_id)
    assert len(pending) == 1
    assert memories.get(previous["id"])["status"] == "active"
    # The existing proposal deduper treats a create and a targeted revision as
    # distinct decisions. Resolve the rejected alternative before retrying.
    service.reject(pending[0]["id"], expected_version=pending[0]["version"])
    service.extract_completed_run("correction", metadata, auto_apply=True)
    current = memories.list_pinned_for_context(project_id=project_id)[0]
    assert current["supersedes_id"] == previous["id"]
    assert memories.get(previous["id"])["status"] == "superseded"
    assert len(memories.search("REST", project_id=project_id)) == 1

    with sqlite3.connect(memories.db_path) as connection:
        connection.execute("UPDATE memories SET expires_at=? WHERE id=?", ("2000-01-01T00:00:00Z", current["id"]))
    assert memories.list_pinned_for_context(project_id=project_id) == []
    assert memories.search("REST", project_id=project_id) == []
    assert memories.list(status="active", project_id=project_id) == []
    assert current["id"] in {item["id"] for item in memories.list(status="expired")}


def test_approval_rejects_stale_evidence_and_version_replay(tmp_path: Path):
    events, _, _, service, thread_id, _ = _prepared_service(tmp_path)
    review = service.create_from_selection(
        thread_id=thread_id,
        project_id=str(tmp_path.resolve()),
        evidence_refs=[{"type": "conversation_message", "source_id": "message-1"}],
    )[0]
    with events._connect() as connection:
        connection.execute(
            "UPDATE conversation_messages SET content='changed' WHERE id='message-1'"
        )
        connection.commit()

    with pytest.raises(MemoryProposalConflict, match="evidence changed"):
        service.approve(review["proposal_id"], expected_version=review["version"])


def test_old_candidate_is_migrated_to_readable_review(tmp_path: Path):
    database = tmp_path / "agent.db"
    first = MemoryStore(database)
    memory_id = first.create_candidate(
        scope="project",
        kind="project_constraint",
        project_id="project-a",
        subject="Runtime",
        content="Keep Python 3.12",
        source_run_id=None,
        source_revision=None,
        confidence=0.8,
    )

    migrated = MemoryStore(database)
    proposals = migrated.list_proposals(status="review_ready", project_id="project-a")
    assert len(proposals) == 1
    assert proposals[0]["target_memory_id"] == memory_id
    assert proposals[0]["verification"] == "legacy_unverified"
    assert migrated.get_proposal_evidence(proposals[0]["id"])[0]["evidence_type"] == "legacy_memory"


def test_v5_memory_database_migrates_without_losing_candidate(tmp_path: Path):
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            kind TEXT NOT NULL,
            project_id TEXT,
            subject TEXT NOT NULL,
            content TEXT NOT NULL,
            source_run_id TEXT,
            source_revision TEXT,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            supersedes_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE memory_fts USING fts5(id UNINDEXED, subject, content);
        INSERT INTO memories VALUES(
            'legacy-candidate', 'project', 'project_constraint', 'project-v5',
            'Compatibility', 'Keep existing APIs', NULL, NULL, 0.9,
            'candidate', NULL, 'now', 'now'
        );
        PRAGMA user_version=5;
        """
    )
    connection.commit()
    connection.close()

    migrated = MemoryStore(database)

    assert migrated.get("legacy-candidate")["content"] == "Keep existing APIs"
    assert Path(str(database) + ".before-v6-memory-proposals.bak").is_file()
    proposal = migrated.list_proposals(status="review_ready")[0]
    assert proposal["target_memory_id"] == "legacy-candidate"
    with migrated._connect() as current:
        assert current.execute("PRAGMA user_version").fetchone()[0] == 6


def test_review_api_never_exposes_internal_model_json(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Memory review", workspace=tmp_path)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="All changes must preserve existing REST APIs.",
        message_id="message-api",
    )
    memories = MemoryStore(tmp_path / "agent.db")
    gateway = ScriptedModelGateway(
        [_analysis_response(), _proposal_response("message-api")]
    )
    engine = RunEngine(
        events,
        build_default_registry(include_mutating=False),
        gateway,
        memory_store=memories,
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories))
    client = TestClient(app)

    created = client.post(
        "/api/agent/memory-proposals/from-selection",
        json={
            "thread_id": thread_id,
            "project_id": str(tmp_path.resolve()),
            "evidence_refs": [
                {"type": "conversation_message", "source_id": "message-api"}
            ],
        },
    )
    assert created.status_code == 200, created.text
    review = created.json()["proposals"][0]
    serialized = json.dumps(review)
    assert "proposal_json" not in serialized
    assert "analysis_json" not in serialized
    assert "raw_proposal" not in serialized

    proposal_id = review["proposal_id"]
    approved = client.post(
        f"/api/agent/memory-proposals/{proposal_id}/approve",
        json={"expected_version": review["version"]},
    )
    assert approved.status_code == 200
    replay = client.post(
        f"/api/agent/memory-proposals/{proposal_id}/approve",
        json={"expected_version": review["version"]},
    )
    assert replay.status_code == 409
    assert client.get(
        "/api/agent/memories", params={"project_id": str(tmp_path.resolve())}
    ).json()["memories"][0]["status"] == "active"


def test_memory_model_failure_returns_readable_bad_gateway(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Memory failure", workspace=tmp_path)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="Remember this project constraint.",
        message_id="message-failure",
    )
    memories = MemoryStore(tmp_path / "agent.db")

    class FailingGateway(ScriptedModelGateway):
        def generate(self, request):
            raise RuntimeError("provider rejected the structured response request")

    gateway = FailingGateway([])
    engine = RunEngine(
        events,
        build_default_registry(include_mutating=False),
        gateway,
        memory_store=memories,
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories))
    client = TestClient(app)

    response = client.post(
        "/api/agent/memory-proposals/from-selection",
        json={
            "thread_id": thread_id,
            "project_id": str(tmp_path.resolve()),
            "evidence_refs": [
                {"type": "conversation_message", "source_id": "message-failure"}
            ],
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "memory_analysis model request failed: "
        "provider rejected the structured response request"
    )
    assert memories.list_proposals(project_id=str(tmp_path.resolve())) == []


def test_invalid_memory_analysis_is_repaired_once(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Repair analysis", workspace=tmp_path)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="Preserve the public REST API.",
        message_id="message-repair-analysis",
    )
    gateway = ScriptedModelGateway(
        [
            ModelResponse(content='{"claims": []}'),
            _analysis_response("message-repair-analysis"),
            _proposal_response("message-repair-analysis"),
        ]
    )
    service = MemoryProposalService(
        event_store=events,
        memory_store=MemoryStore(tmp_path / "agent.db"),
        gateway=gateway,
    )

    reviews = service.create_from_selection(
        thread_id=thread_id,
        project_id=str(tmp_path.resolve()),
        evidence_refs=[
            {"type": "conversation_message", "source_id": "message-repair-analysis"}
        ],
    )

    assert len(reviews) == 1
    assert [request.purpose for request in gateway.requests] == [
        "memory_analysis",
        "memory_analysis_repair",
        "memory_proposal",
    ]
    assert gateway.requests[1].tools == []
    assert "json" in gateway.requests[1].messages[0]["content"].casefold()


def test_invalid_memory_proposal_is_repaired_once(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Repair proposal", workspace=tmp_path)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="Preserve the public REST API.",
        message_id="message-repair-proposal",
    )
    invalid = json.loads(_proposal_response("message-repair-proposal").content)
    invalid["proposals"][0]["scope"] = "global"
    gateway = ScriptedModelGateway(
        [
            _analysis_response("message-repair-proposal"),
            ModelResponse(content=json.dumps(invalid)),
            _proposal_response("message-repair-proposal"),
        ]
    )
    service = MemoryProposalService(
        event_store=events,
        memory_store=MemoryStore(tmp_path / "agent.db"),
        gateway=gateway,
    )

    reviews = service.create_from_selection(
        thread_id=thread_id,
        project_id=str(tmp_path.resolve()),
        evidence_refs=[
            {"type": "conversation_message", "source_id": "message-repair-proposal"}
        ],
    )

    assert len(reviews) == 1
    assert [request.purpose for request in gateway.requests] == [
        "memory_analysis",
        "memory_proposal",
        "memory_proposal_repair",
    ]


def test_invalid_memory_analysis_returns_readable_bad_gateway(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Invalid memory analysis", workspace=tmp_path)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="Remember this project constraint.",
        message_id="message-invalid-analysis",
    )
    memories = MemoryStore(tmp_path / "agent.db")
    gateway = ScriptedModelGateway(
        [
            ModelResponse(content='{"claims": []}'),
            ModelResponse(content='{"claims": []}'),
        ]
    )
    engine = RunEngine(
        events,
        build_default_registry(include_mutating=False),
        gateway,
        memory_store=memories,
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories))
    client = TestClient(app)

    response = client.post(
        "/api/agent/memory-proposals/from-selection",
        json={
            "thread_id": thread_id,
            "project_id": str(tmp_path.resolve()),
            "evidence_refs": [
                {
                    "type": "conversation_message",
                    "source_id": "message-invalid-analysis",
                }
            ],
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "memory_analysis returned invalid output after one repair: "
        "analysis.contradictions must be an array"
    )
    assert memories.list_proposals(project_id=str(tmp_path.resolve())) == []


def test_model_cannot_smuggle_secret_into_proposal(tmp_path: Path):
    events = EventStore(tmp_path / "agent.db")
    thread_id = events.create_thread("Sensitive proposal", workspace=tmp_path)
    events.append_conversation_message(
        thread_id,
        role="user",
        content="Remember the deployment convention, but never credentials.",
        message_id="message-secret",
    )
    proposal = json.loads(_proposal_response("message-secret").content)
    proposal["proposals"][0]["canonical_content"] = (
        "Use api_key=" + "sk-" + "123456789012345678901234 for deployment"
    )
    service = MemoryProposalService(
        event_store=events,
        memory_store=MemoryStore(tmp_path / "agent.db"),
        gateway=ScriptedModelGateway(
            [
                _analysis_response(),
                ModelResponse(content=json.dumps(proposal)),
                ModelResponse(content=json.dumps(proposal)),
            ]
        ),
    )

    with pytest.raises(MemoryProposalGenerationError, match="secret"):
        service.create_from_selection(
            thread_id=thread_id,
            project_id=str(tmp_path.resolve()),
            evidence_refs=[
                {"type": "conversation_message", "source_id": "message-secret"}
            ],
        )
