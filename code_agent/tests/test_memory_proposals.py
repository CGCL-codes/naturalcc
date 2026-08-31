from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import ModelResponse
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.memory_proposals import (
    MemoryProposalGenerationError,
    MemoryProposalService,
    ProposalValidationError,
)
from code_agent.agent_core.memory_store import MemoryProposalConflict, MemoryStore
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


def _proposal_response(message_id: str = "message-1") -> ModelResponse:
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
