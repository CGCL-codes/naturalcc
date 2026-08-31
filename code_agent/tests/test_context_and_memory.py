from pathlib import Path

import pytest

from code_agent.agent_core.context_builder import ContextBuilder, ContextPlanner
from code_agent.agent_core.memory_store import PINNED_MEMORY_KINDS, MemoryStore
from code_agent.agent_core.model_gateway import DeepSeekRequestSerializer
from code_agent.agent_core.token_budget import DeepSeekModelProfile, DeepSeekTokenCounter


@pytest.fixture(scope="module")
def real_counter():
    root = Path(__file__).resolve().parents[1] / "resources" / "deepseek_v3_tokenizer"
    return DeepSeekTokenCounter.from_directory(root)


@pytest.fixture
def serializer():
    return DeepSeekRequestSerializer()


def make_planner(real_counter, serializer):
    profile = DeepSeekModelProfile(
        context_window_tokens=4096,
        default_output_reserve_tokens=512,
        safety_margin_tokens=128,
        provider_framing_tokens=64,
        compaction_trigger_ratio=0.60,
        compaction_target_ratio=0.40,
    )
    return ContextPlanner(real_counter, serializer, profile)


def test_planner_never_backfills_older_small_messages_after_skipping_recent_history(
    real_counter, serializer
):
    messages = [
        {"role": "user", "content": "A" * 5000},
        {"role": "assistant", "content": "small-old"},
        {"role": "user", "content": "B" * 5000},
        {"role": "assistant", "content": "latest"},
    ]

    plan = make_planner(real_counter, serializer).plan(
        system_rules="rules",
        goal="goal",
        messages=messages,
        active_state={},
        memories=[],
        tools=[],
    )

    assert plan.retained_message_indexes
    assert plan.retained_message_indexes == list(
        range(plan.retained_message_indexes[0], len(messages))
    )


def test_planner_keeps_complete_tool_exchange(real_counter, serializer):
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "workspace.read",
                    "args": {"path": "README.md"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "workspace.read",
            "content": "ok",
        },
    ]

    plan = make_planner(real_counter, serializer).plan(
        system_rules="rules",
        goal="goal",
        messages=messages,
        active_state={},
        memories=[],
        tools=[],
    )

    assert [message["role"] for message in plan.messages[-2:]] == [
        "assistant",
        "tool",
    ]


def test_planner_marks_trigger_without_exceeding_hard_limit(
    real_counter, serializer
):
    plan = make_planner(real_counter, serializer).plan(
        system_rules="rules",
        goal="goal",
        messages=[{"role": "user", "content": "context " * 1800}],
        active_state={"verification_pending": True},
        memories=[],
        tools=[],
    )

    assert plan.requires_compaction is True
    assert plan.breakdown.total_with_reserve <= plan.breakdown.context_window_tokens


def test_planner_places_stable_memory_before_dynamic_run_context(
    real_counter, serializer
):
    plan = make_planner(real_counter, serializer).plan(
        system_rules="stable-policy",
        goal="current-goal",
        messages=[{"role": "user", "content": "current-goal"}],
        active_state={"step": 2},
        pinned_memories=[
            {
                "id": "pinned-1",
                "scope": "project",
                "kind": "project_constraint",
                "subject": "API",
                "content": "Keep the API stable",
                "verification": "user_asserted",
            }
        ],
        memories=[
            {
                "id": "retrieved-1",
                "scope": "thread",
                "kind": "failure_pattern",
                "subject": "Tests",
                "content": "Run the focused test first",
                "verification": "test_verified",
            }
        ],
        tools=[],
        checkpoint={"version": 1, "open_work": ["finish tests"]},
        runtime_authorization={
            "workspace": "project-a",
            "target_files": ["src/a.py"],
            "authorized_paths": [],
        },
    )

    system = plan.messages[0]["content"]
    markers = [
        "stable-policy",
        "[PINNED_MEMORY_REFERENCES]",
        "[CONTEXT_CHECKPOINT]",
        "[RUNTIME_AUTHORIZATION]",
        "[USER_GOAL]",
        "[RETRIEVED_MEMORY_REFERENCES]",
        "[ACTIVE_STATE]",
    ]
    positions = [system.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "<subject>API</subject>" in system
    assert "<subject>Tests</subject>" in system
    assert plan.pinned_memory_tokens > 0
    assert plan.retrieved_memory_tokens > 0


def test_planner_keeps_exact_prefix_across_different_run_inputs(
    real_counter, serializer
):
    planner = make_planner(real_counter, serializer)
    pinned = [
        {
            "id": "stable-1",
            "scope": "project",
            "kind": "project_constraint",
            "subject": "API",
            "content": "Keep the API stable",
            "verification": "user_asserted",
        }
    ]

    first = planner.plan(
        system_rules="stable-policy",
        goal="first goal",
        messages=[],
        active_state={"step": 1},
        pinned_memories=pinned,
        memories=[{"id": "dynamic-1", "content": "first retrieval"}],
        tools=[],
        runtime_authorization={"workspace": "project-a"},
    )
    second = planner.plan(
        system_rules="stable-policy",
        goal="second goal",
        messages=[],
        active_state={"step": 9},
        pinned_memories=pinned,
        memories=[{"id": "dynamic-2", "content": "second retrieval"}],
        tools=[],
        runtime_authorization={"workspace": "project-b"},
    )

    end_marker = "[/PINNED_MEMORY_REFERENCES]"
    first_system = first.messages[0]["content"]
    second_system = second.messages[0]["content"]
    first_prefix = first_system[: first_system.index(end_marker) + len(end_marker)]
    second_prefix = second_system[: second_system.index(end_marker) + len(end_marker)]
    assert first_prefix == second_prefix


def test_context_compaction_keeps_tool_exchange_and_active_state():
    events = [
        {"role": "user", "content": "important constraint: keep API stable"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "workspace.read", "args": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "workspace.read", "content": "x" * 200},
        {"role": "assistant", "content": "working"},
    ]
    built = ContextBuilder(max_chars=220).build(
        system_rules="never escape workspace",
        goal="fix bug",
        messages=events,
        active_state={"plan": ["read", "edit"], "failed_test": "assert 1 == 2"},
        memories=[],
    )
    text = str(built)
    assert "never escape workspace" in text
    assert "assert 1 == 2" in text
    assert "keep API stable" in text
    roles = [item["role"] for item in built if "role" in item]
    assert roles.count("tool") <= roles.count("assistant")


def test_memory_lifecycle_scope_search_stale_and_delete(tmp_path: Path):
    store = MemoryStore(tmp_path / "agent.db")
    memory_id = store.create_candidate(
        scope="project",
        kind="constraint",
        project_id="p1",
        subject="python",
        content="Project must keep Python 3.12",
        source_run_id="r1",
        source_revision="rev1",
        confidence=0.9,
    )
    assert store.search("Python version", project_id="p1") == []
    store.activate(memory_id)
    assert store.search("Python 3.12", project_id="p1")[0]["id"] == memory_id
    assert store.search(
        "please fix runtime while preserving Python 3.12",
        project_id="p1",
    )[0]["id"] == memory_id
    store.update(memory_id, subject="python runtime", content="Project must keep Python 3.12 exactly", confidence=0.95)
    assert store.search("runtime", project_id="p1")[0]["content"].endswith("exactly")
    store.mark_stale_for_revision("p1", "rev2")
    assert store.search("Python 3.12", project_id="p1") == []
    store.delete(memory_id)
    assert store.get(memory_id) is None


def test_thread_memory_isolated_from_other_threads(tmp_path: Path):
    store = MemoryStore(tmp_path / "agent.db")
    memory_id = store.create_candidate(
        scope="thread",
        kind="decision",
        project_id="p1",
        subject="formatter",
        content="Use the compact formatter",
        source_run_id="r1",
        source_revision=None,
        confidence=0.8,
        thread_id="thread-a",
    )
    store.activate(memory_id)

    assert store.search("compact formatter", project_id="p1", thread_id="thread-a")
    assert store.search("compact formatter", project_id="p1", thread_id="thread-b") == []


def test_pinned_memories_are_goal_independent_and_not_duplicated_in_search(
    tmp_path: Path,
):
    store = MemoryStore(tmp_path / "agent.db")
    pinned_id = store.create_candidate(
        scope="project",
        kind="project_constraint",
        project_id="p1",
        subject="formatter constraint",
        content="Always use the compact formatter",
        source_run_id=None,
        source_revision=None,
        confidence=0.9,
    )
    retrieved_id = store.create_candidate(
        scope="project",
        kind="workflow",
        project_id="p1",
        subject="formatter workflow",
        content="Run formatter workflow after edits",
        source_run_id=None,
        source_revision=None,
        confidence=0.8,
    )
    store.activate(pinned_id)
    store.activate(retrieved_id)

    pinned = store.list_pinned_for_context(project_id="p1")
    retrieved = store.search(
        "formatter workflow",
        project_id="p1",
        exclude_kinds=PINNED_MEMORY_KINDS,
    )

    assert [item["id"] for item in pinned] == [pinned_id]
    assert [item["id"] for item in retrieved] == [retrieved_id]


def test_run_scoped_memory_is_visible_only_to_its_source_run(tmp_path: Path):
    store = MemoryStore(tmp_path / "agent.db")
    memory_id = store.create_candidate(
        scope="run",
        kind="workflow",
        project_id="p1",
        subject="focused verification",
        content="Use focused verification for this run",
        source_run_id="run-a",
        source_revision=None,
        confidence=0.8,
    )
    store.activate(memory_id)

    assert store.search("focused verification", project_id="p1", run_id="run-a")
    assert store.search("focused verification", project_id="p1", run_id="run-b") == []
