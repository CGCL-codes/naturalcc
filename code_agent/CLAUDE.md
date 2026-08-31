# CLAUDE.md

Working map for coding agents. Keep this file short; update it with `README.md` and `README.zh.md` when architecture, commands, dependencies, or behavior meaningfully change.

## Purpose

`code_agent` is a NaturalCC-enhanced code editing agent. It parses a target project, builds a semantic prompt, then delegates edits or dry-run reports to Aider.

Common tasks: code completion, small project-aware edits/refactors, code summarization, code repair, vulnerability detection, and optional vulnerability remediation.

## Active Entry Points

- `agent_web_api.py`: FastAPI backend, bundled UI server, workspace scan/browse, prompt preview, NDJSON Aider streaming.
- `api/agent_routes.py`: durable `/api/agent/*` run, approval, event, cancellation, and memory routes.
- `agent_core/run_engine.py`: persisted autonomous run state machine.
- `agent_core/contracts.py`, `tool_registry.py`, `policy.py`: trusted tool contracts and permission boundary.
- `agent_core/event_store.py`, `memory_store.py`: SQLite event/snapshot/lease and governed FTS5 memory storage.
- `agent_core/token_budget.py`, `context_builder.py`: pinned DeepSeek offline counting and hard-bounded continuous context planning.
- `agent_core/compaction.py`, `compaction_prompts.py`: two-stage Run/Thread checkpoints, recovery, and deterministic fallback.
- `webui/src/App.jsx`, `webui/src/styles.css`: React UI, feature forms, target file order, terminal/chat output, layout.
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
uv run python aider_runner.py -dir /path/to/project -f src/foo.c -i "补全 foo 函数实现" --preview
```

## Constraints

- `aider` must be on `PATH`.
- `libclang` must be installed for C/C++ parsing.
- C++ coverage is incomplete because parts of the parser still use C-oriented libclang arguments.
- Run the deterministic suite with `uv run --project . pytest tests test_vulnerability_detection.py -q`; also run the Web UI reducer tests and production build.
- Keep changes surgical; prefer existing patterns; do not refactor unrelated code.
- Agent mode reads automatically; write and execute tools require a run-scoped approval.
- Raw chat/events are authoritative. Inject only committed Run/Thread checkpoints plus the continuous tail after their watermark; never rewrite raw history during compaction.
- Agent requests must pass the pinned local DeepSeek tokenizer hard budget, including tools, framing, output reserve, and safety margin. Do not add character-count fallbacks.
- Run Analyzer/Summarizer/repair only at safe points with tools disabled. Maintenance calls use a separate compaction-call budget but still count toward total token/cost/time limits.
- The argv command policy is not an OS sandbox; use a container or restricted account for untrusted repositories.
- Never pass workspace roots, API keys, artifact roots, or permissions through model-generated tool arguments.
- The legacy `/api/run` Pipeline and the durable `/api/agent/*` Runtime must remain independently usable.
