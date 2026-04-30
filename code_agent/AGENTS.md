# AGENTS.md

This file is the working map for future coding agents. Keep it short, accurate, and useful. When code behavior, run commands, dependencies, or architecture changes in a meaningful way, update this file, `README.md`, and `README.zh.md` in the same change.

## Project Purpose

`code_agent` is a context-enhanced code editing agent. It parses a target project with NaturalCC-style static analysis, builds a semantic prompt, then delegates the actual file edits to Aider.

Primary use cases:
- complete function bodies
- complete function signatures
- complete variables, members, and types
- make small project-style-aware code edits or refactors
- detect potential vulnerabilities and optionally auto-remediate with Aider

## Main Entry Points

- `agent_web_api.py`: FastAPI backend and bundled UI server. Provides workspace scan, directory browse, command preview, prompt preview, and NDJSON Aider log streaming.
- `webui/`: React + Vite frontend. Use this for UI layout, interactions, target-file ordering, model/API-key controls, and terminal output.
- `aider_runner.py`: CLI and Aider orchestration. Keep CLI behavior stable unless the user asks to change it.
- `completion_prompt_agent.py`: Prompt construction and completion-type/symbol inference.
- `rag/c/`: C/C++ parser, project graph, context search, and offline evaluation scripts.
- `rag/java/`: Java parser/prompt path.
- `test_api.py`: OpenRouter API connectivity check, not a unit test suite.

There is no legacy UI path. Keep the graphical interface centered on `agent_web_api.py` and `webui/`.

## Runtime Flow

1. User provides `project_dir`, ordered `target_files`, `instruction`, `model`, optional `api_key`, `symbol`, `completion_type`, and `prefix`.
2. `aider_runner.py` normalizes paths.
3. The first target file is the NaturalCC primary parse file.
4. `CompletionPromptAgent` loads/parses the project if needed.
5. Parser/searcher context is assembled into a prompt.
6. `aider_runner.py` writes the prompt to a temporary message file.
7. Aider runs against all target files and edits them.

## Where To Change Things

- UI API, file scan, browsing, streaming protocol: `agent_web_api.py`
- UI screens, controls, layout, styling: `webui/src/App.jsx`, `webui/src/styles.css`
- CLI flags, provider detection, Aider command construction: `aider_runner.py`
- Prompt templates, symbol extraction, completion type inference: `completion_prompt_agent.py`
- Feature plugins (advanced capabilities): `plugins/`
- C/C++ parsing: `rag/c/cfile_parse.py`
- Project traversal/relation cleanup: `rag/c/preprocess.py`
- Cross-file context retrieval: `rag/c/node_prompt.py`
- Java support: `rag/java/`

## Feature Plugin System

- `plugins/base.py`: plugin contract and execution modes.
- `plugins/registry.py`: plugin registration/discovery.
- `plugins/dispatcher.py`: mode-based execution router.
- `plugins/code_completion.py`: code completion plugin.
- `plugins/vulnerability_detection.py`: hybrid vulnerability scan + optional Aider remediation.

`vulnerability_detection` key config:
- `scan_scope`: `targets` or `project`
- `severity_threshold`: `low` / `medium` / `high` / `critical`
- `rule_profile`: `default` / `c_cpp` / `web`
- `auto_fix`: run Aider remediation after analysis
- `max_findings`, `extra_instruction`

## Run Commands

From `code_agent/`:

Install dependencies and create the virtual environment:

```bash
uv sync
```

> `libclang` is a system library required for C/C++ parsing. Install it via your OS package manager (e.g. `sudo apt install libclang1` or `brew install libclang`) before running `uv sync` if C/C++ features are needed.

Development UI:

```bash
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
cd webui && npm install && npm run dev
```

Bundled UI:

```bash
cd webui && npm install && npm run build
cd ..
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

CLI preview:

```bash
uv run python aider_runner.py -dir /path/to/project -f src/foo.c -i "补全 foo 函数实现" --preview
```

CLI execute:

```bash
uv run python aider_runner.py -dir /path/to/project -f src/foo.c -i "补全 foo 函数实现" -m openrouter/deepseek/deepseek-chat
```

## Known Constraints

- `aider` must be installed and available on `PATH`.
- `libclang` must be available for the C/C++ path.
- C/C++ parsing still uses C-oriented libclang arguments in places; C++ coverage may be incomplete.
- The first target file controls semantic prompt focus even when Aider edits multiple files.
- Some offline `rag/` scripts contain local paths and research-only assumptions.
- There is no formal test suite yet; use focused compile/build/API/CLI smoke checks.

## Change Discipline

- Keep changes surgical and aligned with the active path (`agent_web_api.py`, `webui/`, `aider_runner.py`, `completion_prompt_agent.py`, `rag/`).
- Prefer existing patterns over new abstractions.
- Do not modify offline evaluation scripts for online UI/CLI bugs unless the bug is actually there.
- After meaningful changes, update `AGENTS.md`, `README.md`, and `README.zh.md`; remove stale instructions instead of accumulating history.
