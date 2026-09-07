# NaturalCC Code Agent

[中文文档](README.zh.md)

`code_agent` is a local code-editing agent that combines NaturalCC static project understanding, a durable Agent runtime, and Aider-based edits.

It first parses a project to collect functions, variables, types, members, includes, and symbol relations. It then builds a semantic prompt for the requested task and hands that prompt to Aider, which edits the selected target files.

The project supports three user-facing paths:

- Graphical UI: FastAPI backend + React/Vite frontend.
- CLI: Bun + Ink REPL plus `aider_runner.py` for the legacy pipeline.
- VS Code extension: local service + bundled web panel.

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
- [CodeGraph CLI](https://github.com/colbymchenry/codegraph) for knowledge-graph indexing, symbol exploration, call paths, and impact analysis.

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

Install the pinned official DeepSeek V3 tokenizer for Agent-mode hard token budgets:

```bash
uv run python scripts/install_deepseek_tokenizer.py
```

The installer downloads the official archive from `cdn.deepseek.com`, verifies the pinned archive and file SHA-256 hashes, and installs only `tokenizer.json` and `tokenizer_config.json` under `resources/deepseek_v3_tokenizer/`. Agent mode fails closed if these local files are absent or invalid; it does not fall back to character counting.

The project expects these runtime capabilities (all handled by `uv sync`):

- `fastapi`
- `uvicorn`
- `clang` Python bindings
- `aider` on `PATH`

If you need GPU support (e.g. for vLLM-based offline evaluation), install it manually:

```bash
uv pip install vllm
```

Do not put real API keys in source files, README examples, screenshots, or Git history. For Web UI usage, enter the key in the Settings panel; Agent and Pipeline requests send it only with the current local request. CLI, scripts, and backend fallback paths can also read temporary environment variables:

```bash
export DEEPSEEK_API_KEY="your key"
# or
export OPENAI_API_KEY="your key"
# or, for OpenRouter/Aider paths:
export OPENROUTER_API_KEY="your key"
```

### 3. Install Frontend Dependencies

From `code_agent/`:

```bash
cd webui
npm install
```

### 4. Install CodeGraph CLI

CodeGraph is not installed by `uv sync`. Install the upstream CLI first:

```bash
# macOS / Linux / WSL
curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh
```

```powershell
# Windows PowerShell
irm https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.ps1 | iex
```

If Node.js is already available, npm is also supported:

```bash
npm i -g @colbymchenry/codegraph
```

Open a new terminal and verify:

```bash
codegraph --version
```

The upstream `codegraph install` command wires CodeGraph MCP into general agents such as Claude Code, Codex CLI, and Cursor. The NaturalCC Web UI does not require that step: the FastAPI backend calls the `codegraph` CLI directly through `CODE_AGENT_CODEGRAPH_BIN`.

## Using The Graphical Interface

The UI has a FastAPI backend and a React frontend.

### Local Mode

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

FastAPI serves `webui/dist` and all backend APIs from the same port.

### CodeGraph Setup

In the UI, open **Settings -> Knowledge graph**:

1. Enable **Use knowledge graph**.
2. Click **Initialize** when the workspace is not initialized. This runs `codegraph init .` for the selected workspace and creates a local `.codegraph/` directory.
3. Click **Sync** after external code changes, or let the Agent attempt a sync before the next `codegraph.explore` call after tool-driven edits.
4. When the status is **Ready**, the model can call `codegraph.explore` for symbol source, call paths, and change-impact context. **Open graph** renders a local HTML visualization.

`.codegraph/` is a local derived index and should not be committed. By default the backend sets `CODEGRAPH_NO_DOWNLOAD=1` and `CODEGRAPH_NO_DAEMON=1` through `CODE_AGENT_CODEGRAPH_NO_DOWNLOAD=true` and `CODE_AGENT_CODEGRAPH_NO_DAEMON=true`, so installation and indexing stay explicit and visible.

### UI Workflow

1. Create or open a conversation in the left sidebar. Reopening restores chat, workspace, file context, model, budgets, and the last run.
2. Set the workspace and model in the header. API keys are entered only in Settings and are sent with the current local request; they are not persisted with the conversation or Run snapshot.
3. Type `@filename` in the composer to search workspace files, or paste an absolute path and choose **Add context**. Explicit external paths are marked as read/write context.
4. Use **Budget** to set LLM-call and tool-call limits. The header meters show current usage; an active limit cannot be lowered below usage.
5. Send instructions normally. Every user turn creates an auditable run while inheriting the committed ThreadCheckpoint and the continuous conversation tail after its watermark.
6. Open **Run details** for approvals, pause/resume/cancel, events, changed files, verification, and governed project memories.
7. Hover a history row and use its delete action to permanently remove the conversation and its run records. For queued, running, paused, or approval-waiting work, choose **Cancel task first** in the confirmation dialog, then confirm permanent deletion.
8. Select **Pipeline** in the header to use the legacy NaturalCC prompt → Aider runtime.

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
-key "<your key>"
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
| `aider` | Generate prompt → call Aider → modify code files or dry-run reports | Code completion, code repair, code summarization |
| `direct` | Analyze directly → return report / write files | Static reports |
| `hybrid` | Analysis via API → generate fix prompt → Aider repair | Vulnerability detection |

### Built-in Code Summarization Feature

Feature name: `code_summary` (AIDER mode)

Behavior:
- Builds the normal NaturalCC semantic prompt for selected target files, or source files under the whole project.
- Runs Aider with `--dry-run`, so summary generation does not modify files.
- Uses the selected model to produce a deeper code-aware summarization.

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
- The durable Agent runtime has deterministic unit, contract, safety, API, and UI reducer tests under `tests/`; live model and Aider calls remain manual checks.
- `test_api.py` checks API connectivity; it is not a parser or UI test.

## VS Code Extension

The local VS Code extension runs the same FastAPI service and opens its bundled
web interface in an editor tab. It includes application source and the built
frontend, but deliberately does not include Python packages, Aider, `libclang`,
or local models.

Build a local installable package from `code_agent/`:

```bash
cd webui && npm run build
cd ..
npm run package
```

Install the resulting `naturalcc-code-agent-0.1.2.vsix` from VS Code's
**Extensions: Install from VSIX...** command, then run **NaturalCC: Open Code
Agent**. Configure `naturalccCodeAgent.pythonPath` to the Python interpreter
created by `uv sync` (for example, `/path/to/code_agent/.venv/bin/python`) if
the extension cannot find it automatically. The service binds only to
`127.0.0.1` and stops when the extension deactivates.

### User Environment And API Key

The extension bundles the application source but not its Python dependencies.
Users should clone this repository and prepare the runtime once:

```bash
git clone https://github.com/Zidong-Zeng/Naturalcc-coding-agent.git
cd Naturalcc-coding-agent/code_agent
uv sync
```

Then set the VS Code setting `naturalccCodeAgent.pythonPath` to the absolute
path of that environment's Python executable, for example:

```json
"naturalccCodeAgent.pythonPath": "/absolute/path/Naturalcc-coding-agent/code_agent/.venv/bin/python"
```

Open **NaturalCC: Open Code Agent**, choose a model, and enter your own API key
in the UI's **API Key** field for Agent or Pipeline requests. For CLI or
environment fallback workflows, launch VS Code with `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, or `OPENROUTER_API_KEY` set in its environment, then restart
the extension host.
## Durable Agent mode

The project now has two independent runtimes:

- **Pipeline** keeps the original feature-selection → NaturalCC prompt → Aider flow at `POST /api/run`.
- **Agent** uses a persisted state machine at `/api/agent/*`. It can choose read/search/NaturalCC tools, request approval for edits or commands, resume after a process restart, enforce budgets, record an auditable event stream, and create governed long-term-memory proposals from user-selected evidence.

Conversation messages, run state, events, snapshots, approvals, and memories are stored in `outputs/agent_runtime.db` by default. Override it with `CODE_AGENT_DB`. Configure the model without putting a key in source files:

```bash
export CODE_AGENT_MODEL=deepseek-chat
export CODE_AGENT_API_BASE=https://api.deepseek.com/v1
export DEEPSEEK_API_KEY=...
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

Agent requests use the same DeepSeek serializer for offline counting and API submission. The following optional variables tune the explicit model-level hard budget:

```bash
export CODE_AGENT_TOKENIZER_DIR=/absolute/path/to/resources/deepseek_v3_tokenizer
export CODE_AGENT_CONTEXT_WINDOW_TOKENS=65536
export CODE_AGENT_OUTPUT_RESERVE_TOKENS=4096
export CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS=512
export CODE_AGENT_PROVIDER_FRAMING_TOKENS=256
export CODE_AGENT_COMPACTION_TRIGGER_RATIO=0.72
export CODE_AGENT_COMPACTION_TARGET_RATIO=0.50
export CODE_AGENT_ANALYZER_OUTPUT_TOKENS=4096
export CODE_AGENT_SUMMARIZER_OUTPUT_TOKENS=2048
```

When the soft threshold is reached at a safe point, the runtime freezes a continuous completed prefix and runs a tool-disabled, JSON-only `CompactionAnalyzer` followed by `CheckpointSummarizer`. Only the validated committed checkpoint is injected into later requests; structured analysis remains audit data, raw messages/events remain authoritative, and API/JSON failures use a bounded deterministic fallback. Maintenance calls use `max_compaction_calls` and do not consume `max_llm_calls`, while their tokens, cost, and elapsed time still count toward total Run budgets. A request that cannot fit protected rules, goal, WorkingState, checkpoint/tail, tool schemas, output reserve, and safety margin terminates with `reason=context_hard_limit` before the model API is called.

Active memory is assembled in two cache-aware tiers. User preferences, project constraints, architecture decisions, repository conventions, and compatible legacy constraint/decision records form a deterministic, goal-independent pinned block. Other active memories are retrieved with FTS5 for the current goal (including matching Run-scoped records) and form a dynamic block. The prompt order is stable system rules -> pinned memory -> committed checkpoint -> runtime authorization -> current goal -> retrieved memory -> WorkingState -> recent message tail. This keeps the reusable prefix stable while leaving paths, counters, search results, and other changing data near the suffix; pinned IDs are excluded from dynamic retrieval to prevent duplicate injection. If the provider reports DeepSeek-compatible `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`, the Run accumulates them and **Run details -> Usage** displays the measured hit ratio. **Not reported** means the upstream API omitted these fields.

Permission defaults:

- workspace list/read/search, NaturalCC parse/search, Git status/diff: automatic;
- new directories, new UTF-8 files, file edits, Aider, and project commands: run-scoped approval. `workspace.create_directory` can create an explicit parent chain, while `workspace.create_file` requires an existing parent directory and never overwrites an existing path;
- shell interpreters, destructive Git subcommands, workspace escapes, push, and commit: denied by the built-in tools.

The Web UI uses a Codex-style workbench: durable multi-turn conversations live in the left sidebar; `@` and absolute paths create composer context; compact LLM/tool/input-token budget meters stay visible; and the Budget popover can update `max_input_tokens` before or during an active Run. Saving after a terminal `budget_exhausted` Run applies the new limit to the next message. Run details holds approvals, events, changes, verification, prompt-cache usage, and memory. ThreadCheckpoint compaction is automatic. For long-term memory, select one or more persisted chat messages with **Use as memory evidence**, then choose **Create memory suggestions**. A tool-disabled evidence analyzer and proposal composer run in sequence under the same offline DeepSeek hard-limit check used by Agent requests; an invalid structured response receives at most one tool-free schema-repair call per stage. Internal JSON is validated and projected into readable review cards with scope, type, evidence, impact, warnings, and editable fields. Only **Accept and remember** creates an active FTS5 memory; rejected, deferred, failed, and unreviewed proposals are never injected into model context.

An external absolute path becomes available only after the user explicitly adds it to the conversation. Model-generated paths outside the workspace or that explicit authorization remain denied. Common credential files retain sensitive-path protection.

Conversation deletion is an irreversible transaction that removes the thread's messages, runs, events, snapshots, approvals, and workspace leases. The backend rejects deletion while any active run remains. Project-level memories are governed separately and are not removed with a conversation.

The command runner uses an argv allowlist, a cleaned environment, workspace-scoped `cwd`, output caps, timeouts, process-group cancellation, and explicit approval. The allowlist includes the C/C++ compiler entry points `gcc`, `g++`, `c++`, and `clang++`. It is not an OS sandbox: an approved compiler, Python/Node process, package script, or test can still access the host or network with the permissions of the service process. Run the service in a container or restricted OS account for untrusted repositories.

### Agent API

```text
POST /api/agent/threads
GET  /api/agent/threads
GET  /api/agent/threads/{thread_id}
PATCH /api/agent/threads/{thread_id}
GET  /api/agent/threads/{thread_id}/messages
POST /api/agent/threads/{thread_id}/messages
POST /api/agent/context/resolve
POST /api/agent/runs
GET  /api/agent/runs
GET  /api/agent/runs/{run_id}
PATCH /api/agent/runs/{run_id}/budget
POST /api/agent/runs/{run_id}/run
POST /api/agent/runs/{run_id}/step
POST /api/agent/runs/{run_id}/approve
POST /api/agent/runs/{run_id}/reject
POST /api/agent/runs/{run_id}/pause
POST /api/agent/runs/{run_id}/resume
POST /api/agent/runs/{run_id}/cancel
GET  /api/agent/runs/{run_id}/events
GET  /api/agent/runs/{run_id}/events.ndjson
POST /api/agent/memory-proposals/from-selection
GET  /api/agent/memory-proposals
GET  /api/agent/memory-proposals/{proposal_id}/review
GET  /api/agent/memory-proposals/{proposal_id}/evidence
PATCH /api/agent/memory-proposals/{proposal_id}
POST /api/agent/memory-proposals/{proposal_id}/approve
POST /api/agent/memory-proposals/{proposal_id}/reject
POST /api/agent/memory-proposals/{proposal_id}/defer
GET  /api/agent/memories
POST /api/agent/memories
PUT  /api/agent/memories/{memory_id}
POST /api/agent/memories/{memory_id}/activate
POST /api/agent/memories/{memory_id}/reject
DELETE /api/agent/memories/{memory_id}
```

### Approval Troubleshooting

If approval returns 404, the Run is not present in the connected service's database. The UI clears the stale controls; reconnect to the original service/database and reload the conversation, or submit the instruction again if the Run was deleted. Each checkout defaults to its own `code_agent/outputs/agent_runtime.db`; `CODE_AGENT_DB` selects an explicit database. A missing snapshot or changed approval returns 409, not 404. Approvals are checked against the current pending tool call before execution.

### Tests

```bash
uv run --project code_agent pytest code_agent/tests code_agent/test_vulnerability_detection.py -q
npm --prefix code_agent/webui test
npm --prefix code_agent/webui run build
```

Required tests use scripted models and fixture workspaces; they do not need an API key or live Aider call.
