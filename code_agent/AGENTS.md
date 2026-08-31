# AGENTS.md

Working map for coding agents. Keep this file short; update it with `README.md` and `README.zh.md` when architecture, commands, dependencies, or behavior meaningfully change.

## Purpose

`code_agent` is a NaturalCC-enhanced code editing agent. It parses a target project, builds a semantic prompt, then delegates edits or dry-run reports to Aider.

Common tasks: code completion, small project-aware edits/refactors, code summarization, code repair, vulnerability detection, and optional vulnerability remediation.

## Active Entry Points

- `agent_web_api.py`: FastAPI backend, bundled UI server, workspace scan/browse, prompt preview, NDJSON Aider streaming.
- `api/agent_routes.py`: durable `/api/agent/*` conversation, run, context, budget, approval, event, cancellation, and memory routes.
- `agent_core/run_engine.py`: persisted autonomous run state machine.
- `agent_core/contracts.py`, `tool_registry.py`, `policy.py`: trusted tool contracts and permission boundary.
- `agent_core/event_store.py`, `memory_store.py`: SQLite conversation/message/event/snapshot/lease storage, governed memory proposals, review audit, goal-independent pinned memory, and active-only FTS5 retrieval memory.
- `agent_core/memory_proposals.py`, `memory_prompts.py`, `memory_projection.py`: frozen evidence, tool-free two-pass proposal generation, validation, and deterministic user-facing review DTOs.
- `agent_core/token_budget.py`, `context_builder.py`: pinned DeepSeek offline counting, full-request hard budgets, cache-friendly stable/dynamic prompt tiers, and continuous-tail ContextPlans.
- `agent_core/compaction.py`, `compaction_prompts.py`: versioned Analyzer/Summarizer checkpoints, validation, recovery, and deterministic fallback.
- `webui/src/App.jsx`, `MemoryReviewPanel.jsx`, `MemoryProposalCard.jsx`, `webui/src/styles.css`: Codex-style React workbench with conversation history, evidence selection, governed memory review, budget meters, chat output, and Run details.
- `webui/src/conversationState.js`, `budgetUi.js`: deterministic thread hydration, context payload, grouping, call/input-token budget validation, and progress helpers.
- `vscode-extension.js`, `vscode_server.py`: VS Code webview host and local-service bootstrap. Package with `npm run package` after building `webui/dist`.
- `aider_runner.py`: CLI, NaturalCC prompt generation, Aider command construction/streaming. Keep CLI behavior stable unless requested.
- `completion_prompt_agent.py`: prompt construction, symbol and completion-type inference.
- `plugins/`: Feature Plugin System.
- `rag/c/`: C/C++ parser, project graph, context retrieval. Avoid offline eval scripts unless the bug is there.
- `rag/java/`: Java parser/prompt path.
- `test_api.py`: API connectivity check, not a unit suite.

There is no legacy UI path; keep graphical work centered on `agent_web_api.py` and `webui/`.

## Runtime Notes

- Request fields: `project_dir`, ordered `target_files`, `instruction`, `model`, optional `api_key`, plus feature config.
- First target file is the NaturalCC primary parse file, even when Aider receives multiple files.
- NaturalCC parsing must work; do not silently bypass it for completion/repair flows.
- Python `clang` bindings must match system `libclang`; this repo pins `clang==18.1.8` for LLVM 18 / `libclang1-18`.

## Plugins

Plugins auto-register from `plugins/` via `@register_plugin`; the frontend renders each plugin's `config_schema`.

- `plugins/base.py`: `FeaturePlugin`, `ExecutionMode` (`aider`/`direct`/`hybrid`), `ConfigField`, `ExecutionContext`.
- `plugins/registry.py`: registration and schema listing.
- `plugins/dispatcher.py`: routes AIDER, DIRECT, and HYBRID execution.
- `plugins/code_completion.py`: original NaturalCC completion flow.
- `plugins/code_summary.py`: NaturalCC semantic summaries via Aider `--dry-run`.
- `plugins/code_repair.py`: focused repair prompt via Aider.
- `plugins/vulnerability_detection.py`: static scan with optional Aider remediation.

Add a plugin by creating `plugins/my_feature.py`, inheriting `FeaturePlugin`, implementing `metadata`, `config_schema`, `execute`, and decorating with `@register_plugin`.

## Commands

```bash
uv sync
uv run python scripts/install_deepseek_tokenizer.py
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
cd webui && npm install && npm run dev
cd webui && npm run build
npm run package
uv run python aider_runner.py -dir /path/to/project -f src/foo.c -i "补全 foo 函数实现" --preview
```

## Constraints

- `aider` must be on `PATH`.
- `libclang` must be installed for C/C++ parsing.
- C++ coverage is incomplete because parts of the parser still use C-oriented libclang arguments.
- Run the deterministic suite with `uv run --project . pytest tests test_vulnerability_detection.py -q`; also run the Web UI reducer tests and production build.
- Keep changes surgical; prefer existing patterns; do not refactor unrelated code.
- Agent mode reads automatically; write and execute tools require a run-scoped approval.
- `workspace.create_file` creates only a new UTF-8 text file: its parent must already exist, it never overwrites, and its absent-state snapshot lets `workspace.restore_snapshot` remove the run-created file.
- `workspace.create_directory` creates a directory or an explicit `parents=true` chain, treats existing directories as idempotent success, and snapshots every newly created directory; restore removes only empty directories.
- Each user turn creates a Run under a durable thread. Raw chat/events remain authoritative; only committed Run/Thread checkpoints are injected as derived context, and project memories remain governed candidates.
- Long-term memory follows `frozen evidence -> structured analyzer -> proposal composer -> validation -> deterministic review view -> user approval -> active memory`; each model stage may use at most one tool-free schema-repair call. Never render raw Proposal JSON to normal users or index an unreviewed proposal in FTS5.
- Active memory is injected in two tiers: stable kinds (`user_preference`, `project_constraint`, `architecture_decision`, `repository_convention`, plus legacy `constraint`/`decision`) are deterministically pinned before dynamic Run content; other kinds are goal-retrieved and placed after the current goal. Never duplicate pinned entries in the retrieved tier.
- Keep the model prompt ordered as stable system rules -> pinned memory -> committed checkpoint -> runtime authorization -> current goal -> retrieved memory -> WorkingState -> recent message tail. Do not put workspace paths, counters, or other per-Run values in stable system rules.
- Persist provider-reported `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` in Run state and `model.usage_recorded`; absence means “not reported”, not a zero-percent cache rate.
- SQLite schema v6 creates a non-overwriting `.before-v6-memory-proposals.bak` snapshot before migrating an existing v1-v5 database.
- Count the serialized messages, tool schemas, provider framing, output reserve, and safety margin with the pinned local DeepSeek tokenizer before every Agent-model request. Never silently fall back to character budgets or delete protected rules/WorkingState to make a request fit.
- Compaction runs only at safe points over continuous completed prefixes. Analyzer/Summarizer/repair calls use no tools, have separate `max_compaction_calls`, and must not consume `max_llm_calls`; their token/cost/time usage still contributes to total budgets.
- Thread deletion is permanent and transactional. Reject deletion when any Run is queued, running, waiting for approval, or paused; the UI must require an explicit cancellation action before deletion. Terminal Runs cannot transition back to active states.
- Only user-selected absolute paths may widen filesystem access beyond the primary workspace; model-generated paths cannot widen authorization.
- The argv command policy allows `gcc`, `g++`, `c++`, and `clang++`, but is not an OS sandbox; use a container or restricted account for untrusted repositories.
- Never pass workspace roots, API keys, artifact roots, or permissions through model-generated tool arguments.
- The legacy `/api/run` Pipeline and the durable `/api/agent/*` Runtime must remain independently usable.
