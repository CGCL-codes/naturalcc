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

### 1. Activate the Python Environment

```bash
conda activate naturalcc
```

### 2. Install Python Requirements

The project expects these runtime capabilities:

- `fastapi`
- `uvicorn`
- `clang` Python bindings
- `libclang`
- `aider` on `PATH`

If the active environment is missing pieces, install them according to your local environment policy. A common setup is:

```bash
pip install fastapi uvicorn clang aider-chat
conda install -c conda-forge libclang
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

### Development Mode

Use this while editing frontend code. It gives Vite hot reload.

Terminal 1:

```bash
python agent_web_api.py --host 127.0.0.1 --port 7860
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
python agent_web_api.py --host 127.0.0.1 --port 7860
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
python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "补全 foo 函数实现" \
  --preview
```

### Execute Aider

```bash
python aider_runner.py \
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

## Notes And Limitations

- C/C++ parsing depends on `libclang`.
- Some C++ syntax may not parse reliably because parts of the parser still use C-oriented libclang settings.
- `rag/` includes offline research/evaluation scripts with local-path assumptions.
- The project currently has smoke checks rather than a formal automated test suite.
- `test_api.py` checks API connectivity; it is not a parser or UI test.
