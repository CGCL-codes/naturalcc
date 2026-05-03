# CLAUDE.md

Working map for coding agents. Keep this file short; update it with `README.md` and `README.zh.md` when architecture, commands, dependencies, or behavior meaningfully change.

## Purpose

`code_agent` is a NaturalCC-enhanced code editing agent. It parses a target project, builds a semantic prompt, then delegates edits or dry-run reports to Aider.

Common tasks: code completion, small project-aware edits/refactors, code summary, code repair, vulnerability detection, and optional vulnerability remediation.

## Active Entry Points

- `agent_web_api.py`: FastAPI backend, bundled UI server, workspace scan/browse, prompt preview, NDJSON Aider streaming.
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
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
cd webui && npm install && npm run dev
cd webui && npm run build
uv run python aider_runner.py -dir /path/to/project -f src/foo.c -i "补全 foo 函数实现" --preview
```

## Constraints

- `aider` must be on `PATH`.
- `libclang` must be installed for C/C++ parsing.
- C++ coverage is incomplete because parts of the parser still use C-oriented libclang arguments.
- No formal unit suite yet; use focused compile/build/API/CLI smoke checks.
- Keep changes surgical; prefer existing patterns; do not refactor unrelated code.
