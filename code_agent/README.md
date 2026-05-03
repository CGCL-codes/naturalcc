# NaturalCC Code Agent

[中文文档](README.zh.md)

`code_agent` is a local code-editing agent that combines static project understanding with Aider-based edits.

It first parses a project to collect functions, variables, types, members, includes, and symbol relations. It then builds a semantic prompt for the requested task and hands that prompt to Aider, which edits the selected target files.

The project supports two user-facing paths:

- Graphical UI: FastAPI backend + React/Vite frontend.
- CLI: `aider_runner.py`.

The first selected target file is always the primary file used for NaturalCC prompt construction. Aider can still receive and edit multiple target files.

## What It Is

This is not a general chat bot. It is a context-enhanced code agent for project-aware code completion and editing.

Typical tasks:

- complete a function body
- complete a function signature
- complete a variable, member, or type
- make small code changes that should follow existing project style
- detect potential vulnerabilities and optionally auto-fix them
- preview the exact semantic prompt before running Aider

Main files:

- `agent_web_api.py`: FastAPI backend and bundled frontend server.
- `webui/`: React + Vite graphical interface.
- `aider_runner.py`: CLI entry and Aider command orchestration.
- `completion_prompt_agent.py`: semantic prompt construction.
- `rag/c/`: C/C++ parsing and context retrieval.
- `rag/java/`: Java parsing and prompt path.
- `test_api.py`: OpenRouter key/connectivity check.

## Environment Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Node.js & npm (for the frontend)

### 1. Install System Dependencies

C/C++ parsing requires the system `libclang` library. Install it via your OS package manager before running `uv sync`:

- **Ubuntu/Debian**: `sudo apt install libclang1`
- **macOS**: `brew install libclang`
- **Other**: consult your distribution's package repository for `libclang`

### 2. Create the Python Environment

From `code_agent/`:

```bash
uv sync
```

This creates a `.venv` virtual environment and installs all locked Python dependencies. No manual conda or pip steps are needed.

The project expects these runtime capabilities (all handled by `uv sync`):

- `fastapi`
- `uvicorn`
- `clang` Python bindings
- `aider` on `PATH`

If you need GPU support (e.g. for vLLM-based offline evaluation), install it manually:

```bash
uv pip install vllm
```

For OpenRouter/OpenAI calls, either pass an API key in the UI/CLI or set environment variables:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENAI_API_KEY=sk-...
```

### 3. Install Frontend Dependencies

From `code_agent/`:

```bash
cd webui
npm install
```

## Using The Graphical Interface

The UI has a FastAPI backend and a React frontend.

### Quick Start (One-Click)

If you have a graphical terminal emulator installed (gnome-terminal, konsole, alacritty, etc.), or `tmux` as a fallback:

```bash
./start.sh
```

This script automatically uses the `.venv` Python and opens two terminal windows (or tmux panes):
- One for the FastAPI backend
- One for the Vite frontend dev server

When using tmux, the session is named `ncc-agent`. Press `Ctrl+B` then `D` to detach; re-attach with `tmux attach -t ncc-agent`.

### Development Mode

Use this while editing frontend code. It gives Vite hot reload.

Terminal 1:

```bash
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

Terminal 2:

```bash
cd webui
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

In development mode, Vite serves the frontend on `5173` and proxies `/api/*` requests to FastAPI on `7860`.

### Bundled Local Mode

Use this when you want one server to serve both the UI and API.

```bash
cd webui
npm run build
cd ..
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860/
```

In bundled mode, FastAPI serves `webui/dist` and all backend APIs from the same port.

### UI Workflow

1. Set the project root.
2. Select one or more target files.
3. Keep the first selected file as the primary parse file.
4. Choose or type a model.
5. Optionally enter API key, symbol, completion type, or prefix.
6. Enter the development instruction.
7. Click preview to inspect the final prompt, or execute to run Aider.

## Using The CLI

Run commands from `code_agent/`.

### Preview Prompt Only

```bash
uv run python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "补全 foo 函数实现" \
  --preview
```

### Execute Aider

```bash
uv run python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "根据现有风格完善 foo 函数实现" \
  -m openrouter/deepseek/deepseek-chat
```

### Useful CLI Options

```bash
-dir /path/to/project
-f src/foo.c include/foo.h
-i "你的修改或补全需求"
-m openrouter/deepseek/deepseek-chat
-key sk-...
-s parse_flags
-t function_body
--prefix parse_
--preview
```

`-t` accepts:

```text
member
variable
function
function_body
type
```

## API Endpoints

`agent_web_api.py` exposes:

- `GET /api/health`
- `GET /api/bootstrap`
- `GET /api/models`
- `GET|POST /api/workspace/scan`
- `GET /api/browse`
- `POST /api/command-preview`
- `POST /api/prompt/preview`
- `POST /api/run`

`/api/run` streams newline-delimited JSON events so the frontend can display live Aider logs.

## Feature Plugin System

The **Advanced** panel is now powered by a plugin architecture. Each feature is a `FeaturePlugin` under `plugins/`. The frontend renders forms dynamically from each plugin's `config_schema`, so adding a new capability does **not** require any frontend code changes.

### Architecture

- `plugins/base.py` — `FeaturePlugin` abstract base class, `ExecutionMode` (`aider`/`direct`/`hybrid`), `ConfigField` schema definition, `ExecutionContext`.
- `plugins/registry.py` — `@register_plugin` class decorator; plugins auto-register on import.
- `plugins/dispatcher.py` — routes execution to AIDER, DIRECT, or HYBRID mode.
- `plugins/code_completion.py` — the existing `symbol`/`completion_type`/`prefix` logic, migrated to a plugin.
- `plugins/code_summary.py` — NaturalCC + Aider dry-run code summaries.
- `plugins/code_repair.py` — AIDER-mode repair prompts for bug, compile, and test failures.
- `plugins/vulnerability_detection.py` — vulnerability analysis with optional Aider remediation.

### Execution Modes

| Mode | Behavior | Example |
|------|----------|---------|
| `aider` | Generate prompt → call Aider → modify code files or dry-run reports | Code completion, code repair, code summary |
| `direct` | Analyze directly → return report / write files | Static reports |
| `hybrid` | Analysis via API → generate fix prompt → Aider repair | Vulnerability detection |

### Built-in Code Summary Feature

Feature name: `code_summary` (AIDER mode)

Behavior:
- Builds the normal NaturalCC semantic prompt for selected target files, or source files under the whole project.
- Runs Aider with `--dry-run`, so summary generation does not modify files.
- Uses the selected model to produce a deeper code-aware report.

Main config fields:
- `summary_scope`: `targets` or `project`
- `detail_level`: `brief` / `standard` / `detailed`
- `include_symbols`: ask Aider to include key symbols and data flow
- `max_files`: cap files sent through NaturalCC and Aider

### NaturalCC / libclang Version Alignment

NaturalCC requires the Python `clang` bindings to match the installed system `libclang`. This project pins `clang==18.1.8`, which matches the Ubuntu LLVM 18 / `libclang1-18` family. If your system uses a different major LLVM version, update the `clang` dependency and lock file to the same major version as `libclang.so`.

### Built-in Code Repair Feature

Feature name: `code_repair` (AIDER mode)

Behavior:
- Builds a focused repair prompt from the user instruction, repair type, optional failure log, and optional extra context.
- Uses the existing NaturalCC semantic prompt path, then delegates edits to Aider.
- Biases toward minimal fixes and preserving existing interfaces.

Main config fields:
- `repair_type`: `bug_fix` / `compile_error` / `test_failure` / `safe_refactor`
- `failure_log`: compiler, test, stack trace, or runtime output
- `extra_context`: constraints, expected behavior, or reproduction notes
- `allow_refactor`: allow small supporting refactors when needed

### Built-in Vulnerability Detection Feature

Feature name: `vulnerability_detection` (HYBRID mode)

Behavior:
- Phase 1: run static pattern-based vulnerability scan and generate a report.
- Phase 2 (optional): if `auto_fix=true`, generate remediation instruction and run Aider on selected target files.

Main config fields:
- `scan_scope`: `targets` or `project`
- `severity_threshold`: `low` / `medium` / `high` / `critical`
- `rule_profile`: `default` / `c_cpp` / `web`
- `auto_fix`: enable/disable repair stage
- `max_findings`: cap report size
- `extra_instruction`: extra remediation constraints

Usage tips:
- Select target files first if you plan to enable `auto_fix`.
- Start with `auto_fix=false` and review findings before enabling automatic remediation.

### How to Add a New Feature Plugin

1. Create a new file under `plugins/`, e.g. `plugins/my_feature.py`.
2. Inherit `FeaturePlugin`, implement `metadata`, `config_schema`, and `execute`.
3. Decorate the class with `@register_plugin`.
4. Restart the backend. The frontend will automatically show the new feature and render its form.

Example:

```python
# plugins/my_feature.py
from typing import Any, Dict, Generator, List, Optional
from code_agent.plugins.base import (
    FeaturePlugin, FeatureMetadata, ExecutionMode,
    ConfigField, ConfigFieldType, ExecutionContext, PluginResult,
)
from code_agent.plugins.registry import register_plugin


@register_plugin
class MyFeaturePlugin(FeaturePlugin):

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="my_feature",           # unique ID
            label="My Feature",          # display name
            description="What it does",
            execution_mode=ExecutionMode.DIRECT,  # or AIDER / HYBRID
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="my_param",
                label="My Parameter",
                type=ConfigFieldType.TEXT,   # text / textarea / select / switch / file
                required=True,
                default="",
                placeholder="Enter value",
                help_text="This is shown under the field",
            ),
        ]

    def execute(self, context: ExecutionContext) -> Generator[str, None, None]:
        # yield strings for log output
        yield "Starting...\n"
        # ... your logic ...
        # yield PluginResult when done (for DIRECT / HYBRID)
        yield PluginResult(success=True, message="Done!")
```

### Config Field Types

| Type | Renders as | Extra properties |
|------|-----------|-----------------|
| `text` | `<input type="text">` | `placeholder`, `default` |
| `textarea` | `<textarea>` | `placeholder`, `default` |
| `select` | `<select>` | `options: [{value, label}]`, `default` |
| `switch` | `<input type="checkbox">` | `default` (bool) |
| `file` | `<input type="file">` | `accept`, `multiple` |

### File Upload

If your plugin config contains a `file` type field, the frontend will automatically send the request as `multipart/form-data`. Uploaded files are available in `context.uploaded_files` as `{field_name: UploadFile}`.

### API Changes for Plugins

`/api/bootstrap` now returns:

```json
{
  "features": [{"name": "...", "label": "...", "execution_mode": "..."}],
  "schemas": {"feature_name": [{"name": "...", "type": "...", ...}]},
  "default_feature": "code_completion"
}
```

`/api/run` accepts both JSON (backward compatible) and `multipart/form-data` (for file uploads). The payload should include:

```json
{
  "feature": "my_feature",
  "feature_config": {"my_param": "value"}
}
```

## Notes And Limitations

- C/C++ parsing depends on `libclang`.
- Some C++ syntax may not parse reliably because parts of the parser still use C-oriented libclang settings.
- `rag/` includes offline research/evaluation scripts with local-path assumptions.
- The project currently has smoke checks rather than a formal automated test suite.
- `test_api.py` checks API connectivity; it is not a parser or UI test.
