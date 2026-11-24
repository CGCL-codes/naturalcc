import json
import sqlite3
from pathlib import Path

from code_agent.agent_core.context_builder import ContextPlanner
from code_agent.agent_core.contracts import ModelResponse, RunBudget
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.memory_store import MemoryStore
from code_agent.agent_core.model_gateway import DeepSeekRequestSerializer, ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.token_budget import DeepSeekModelProfile, DeepSeekTokenCounter
from code_agent.agent_core.tool_registry import build_default_registry


def _note(store: MemoryStore, workspace: str, subject: str, content: str, *, activate: bool = True) -> str:
    memory_id = store.create_candidate(
        scope="project", kind="workflow", project_id=workspace,
        subject=subject, content=content, source_run_id=None, source_revision=None, confidence=0.9,
    )
    if activate:
        store.activate(memory_id)
    return memory_id


def test_rebuilds_legacy_search_index_and_ranks_cjk_and_exact_identifiers(tmp_path: Path):
    database = tmp_path / "agent.db"
    store = MemoryStore(database)
    local = _note(store, "p1", "离线测试配置", "检查 src/local_cache.py。")
    remote = _note(store, "p1", "在线验证", "检查 src/remote_cache.py。")
    _note(store, "p2", "离线测试配置", "检查 src/local_cache.py。")
    candidate = _note(store, "p1", "离线测试待审核", "检查 src/local_cache.py。", activate=False)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE memory_search_format")
        connection.execute("DELETE FROM memory_fts")
        connection.execute(
            "INSERT INTO memory_fts SELECT id, subject, content FROM memories WHERE status='active' OR id=?",
            (candidate,),
        )

    for _ in range(2):
        store = MemoryStore(database)
        chinese = store.search_with_quality("离线测试", project_id="p1")
        assert [item["id"] for item in chinese.strong_matches] == [local]
        mixed = store.search_with_quality("检查 SRC/LOCAL_CACHE.PY", project_id="p1")
        assert [item["id"] for item in mixed.strong_matches] == [local]
        assert {item["id"] for item in mixed.matches} == {local, remote}
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM memory_fts WHERE id=?", (local,)).fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM memory_fts WHERE id=?", (candidate,)).fetchone() == (0,)

    store.update(local, subject="离线验收配置", content="检查 src/build_cache.py。", confidence=0.9)
    assert not store.search_with_quality("src/local_cache.py", project_id="p1").strong_matches
    store.reject(local)
    assert not store.search_with_quality("src/build_cache.py", project_id="p1").strong_matches


def test_english_weak_match_is_reranked_within_the_run_budget(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    memory = MemoryStore(store.db_path)
    workspace = str(tmp_path.resolve())
    stale_hit = _note(memory, workspace, "formatter", "Apply formatting to Python code.")
    relevant = _note(memory, workspace, "Release prerequisite", "Check fixture compatibility before publishing.")
    foreign = _note(memory, "different-workspace", "formatter publish fixture", "Private project context.")
    model = ScriptedModelGateway([
        ModelResponse(content=json.dumps({"ids": [foreign, relevant]})),
        ModelResponse(content="done"),
        ModelResponse(content="one-call answer"),
        ModelResponse(content="direct answer"),
    ])
    tokenizer_root = Path(__file__).resolve().parents[1] / "resources" / "deepseek_v3_tokenizer"
    planner = ContextPlanner(
        DeepSeekTokenCounter.from_directory(tokenizer_root),
        DeepSeekRequestSerializer(), DeepSeekModelProfile(context_window_tokens=16_384),
    )
    engine = RunEngine(
        store, build_default_registry(include_mutating=False), model,
        memory_store=memory, context_planner=planner,
    )
    goal = "Will the formatter package signed artifacts correctly?"
    lexical = memory.search_with_quality(goal, project_id=workspace)
    assert [item["id"] for item in lexical.matches] == [stale_hit]
    assert not lexical.strong_matches
    run_id = engine.create_run(tmp_path, goal)
    assert engine.run(run_id)["final_answer"] == "done"
    selector, foreground = model.requests
    assert selector.purpose == "memory_recall"
    assert not selector.tools
    assert [row["id"] for row in json.loads(selector.messages[1]["content"])["candidates"]] == [stale_hit, relevant]
    assert relevant in foreground.messages[-1]["content"]
    assert stale_hit not in foreground.messages[-1]["content"]
    assert foreign not in foreground.messages[-1]["content"]

    run_id = engine.create_run(tmp_path, goal, RunBudget(max_llm_calls=1))
    assert engine.run(run_id)["final_answer"] == "one-call answer"
    assert [request.purpose for request in model.requests] == ["memory_recall", "agent", "agent"]
    assert stale_hit in model.requests[-1].messages[-1]["content"]

    run_id = engine.create_run(tmp_path, "Release prerequisite")
    assert engine.run(run_id)["final_answer"] == "direct answer"
    assert [request.purpose for request in model.requests] == ["memory_recall", "agent", "agent", "agent"]
    assert relevant in model.requests[-1].messages[-1]["content"]
