import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from code_agent.aider_runner import (
        DEFAULT_PROJECT_DIR,
        detect_provider,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )
else:
    from .aider_runner import (
        DEFAULT_PROJECT_DIR,
        detect_provider,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )

if __package__ in (None, ""):
    import code_agent.plugins
    from code_agent.plugins.base import ExecutionContext
    from code_agent.plugins.dispatcher import dispatcher, ndjson_event
    from code_agent.plugins.registry import registry
else:
    from . import plugins
    from .plugins.base import ExecutionContext
    from .plugins.dispatcher import dispatcher, ndjson_event
    from .plugins.registry import registry


MODELS = [
    "openrouter/anthropic/claude-3-haiku",
    "openrouter/anthropic/claude-3.5-sonnet",
    "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/deepseek/deepseek-chat",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/google/gemini-2.5-pro",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/gpt-4o",
    "openrouter/openai/gpt-4o-mini",
    "openrouter/openai/gpt-5",
    "openrouter/openai/gpt-5-chat",
    "openrouter/openai/gpt-5-codex",
    "openrouter/openai/gpt-5-mini",
    "openrouter/openrouter/auto",
    "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "openrouter/qwen/qwen3-coder",
]
DEFAULT_MODEL = "openrouter/deepseek/deepseek-chat"

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    ".aider.tags.cache.v3",
    ".aider.tags.cache.v4",
    "datasets",
    "node_modules",
    ".idea",
    ".vscode",
}
IGNORE_FILE_EXTENSIONS = {".log", ".ps"}
PARSABLE_SOURCE_EXTENSIONS = {".c", ".cpp", ".h", ".hpp", ".java"}
COMPLETION_TYPES = ["", "member", "variable", "function", "function_body", "type"]


class AgentRequest(BaseModel):
    project_dir: str = Field(default=DEFAULT_PROJECT_DIR)
    target_files: List[str] = Field(default_factory=list)
    instruction: str = ""
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    symbol: Optional[str] = None
    completion_type: Optional[str] = None
    prefix: str = ""
    feature: str = Field(default="code_completion")
    feature_config: Dict[str, Any] = Field(default_factory=dict)


class ScanRequest(BaseModel):
    project_dir: str = Field(default=DEFAULT_PROJECT_DIR)
    target_files: List[str] = Field(default_factory=list)


def should_ignore_file(filename: str) -> bool:
    return filename.startswith(".") or Path(filename).suffix.lower() in IGNORE_FILE_EXTENSIONS


def sanitize_target_files(target_files: Optional[Iterable[str]]) -> List[str]:
    if target_files is None:
        return []

    result: List[str] = []
    seen = set()
    for item in target_files:
        normalized_item = str(item or "").strip().replace("\\", "/")
        if not normalized_item:
            continue
        if Path(normalized_item).suffix.lower() in IGNORE_FILE_EXTENSIONS:
            continue
        if normalized_item in seen:
            continue
        result.append(normalized_item)
        seen.add(normalized_item)
    return result


def get_local_files(root_dir: str = DEFAULT_PROJECT_DIR) -> List[str]:
    normalized_root = normalize_project_dir(root_dir)
    if not os.path.isdir(normalized_root):
        return []

    file_list: List[str] = []
    for dirpath, dirnames, filenames in os.walk(normalized_root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in IGNORE_DIRS]
        for filename in filenames:
            if should_ignore_file(filename):
                continue
            full_path = os.path.relpath(os.path.join(dirpath, filename), normalized_root)
            file_list.append(full_path.replace("\\", "/"))
    return sorted(file_list, key=str.casefold)


def merge_file_choices(file_choices: Sequence[str], target_files: Sequence[str]) -> Tuple[List[str], List[str]]:
    selected_files = sanitize_target_files(target_files)
    merged_choices = list(file_choices)
    seen = set(merged_choices)
    for item in selected_files:
        if item in seen:
            continue
        merged_choices.append(item)
        seen.add(item)
    return merged_choices, selected_files


def mask_secret(secret: Optional[str]) -> str:
    secret = (secret or "").strip()
    if not secret:
        return "not-set"
    if len(secret) <= 8:
        return f"{secret[:2]}***"
    return f"{secret[:4]}...{secret[-4:]}"


def build_cli_command_preview(request: AgentRequest, preview_only: bool) -> str:
    selected_files = sanitize_target_files(request.target_files)
    instruction = (request.instruction or "").strip()
    model = (request.model or "").strip()
    api_key = (request.api_key or "").strip()
    # Prefer feature_config for backward/forward compatibility
    feature_config = dict(request.feature_config or {})
    symbol = (request.symbol or feature_config.get("symbol") or "").strip()
    completion_type = (request.completion_type or feature_config.get("completion_type") or "").strip()
    prefix = (request.prefix or feature_config.get("prefix") or "").strip()
    project_dir = normalize_project_dir(request.project_dir or DEFAULT_PROJECT_DIR)

    command = ["python", "aider_runner.py", "-dir", project_dir]
    if selected_files:
        command.extend(["-f", *selected_files])
    command.extend(["-i", instruction or "<fill-instruction>"])
    if model:
        command.extend(["-m", model])
    if api_key:
        command.extend(["--api-key", f"<masked:{mask_secret(api_key)}>"])
    if symbol:
        command.extend(["-s", symbol])
    if completion_type:
        command.extend(["-t", completion_type])
    if prefix:
        command.extend(["--prefix", prefix])
    if preview_only:
        command.append("--preview")

    header = [
        "# Equivalent CLI",
        "# The first target file is the NaturalCC primary parse file.",
    ]
    if not api_key:
        header.append("# Empty API key falls back to OPENROUTER_API_KEY / OPENAI_API_KEY.")

    return "\n".join(header + [" \\\n  ".join(shlex.quote(part) for part in command)])


def build_workspace_snapshot(project_dir: str, target_files: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    normalized_root = normalize_project_dir(project_dir or DEFAULT_PROJECT_DIR)
    file_choices = get_local_files(normalized_root) if os.path.isdir(normalized_root) else []
    merged_choices, selected_files = merge_file_choices(file_choices, target_files or [])
    parsable_files = [
        path for path in file_choices
        if Path(path).suffix.lower() in PARSABLE_SOURCE_EXTENSIONS
    ]
    custom_targets = [path for path in selected_files if path not in set(file_choices)]

    return {
        "normalized_root": normalized_root,
        "exists": os.path.isdir(normalized_root),
        "files": merged_choices,
        "scanned_files": file_choices,
        "selected_files": selected_files,
        "counts": {
            "visible_files": len(file_choices),
            "parsable_files": len(parsable_files),
            "selected_files": len(selected_files),
            "custom_targets": len(custom_targets),
        },
        "primary_target": selected_files[0] if selected_files else "",
        "ignore_dirs": sorted(IGNORE_DIRS),
        "parsable_extensions": sorted(PARSABLE_SOURCE_EXTENSIONS),
    }


def browse_directory(path_value: Optional[str]) -> Dict[str, Any]:
    raw_path = (path_value or DEFAULT_PROJECT_DIR).strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_file():
        path = path.parent
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    directories = []
    files = []
    try:
        entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        for entry in entries:
            if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                continue
            item = {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
            }
            if entry.is_dir():
                directories.append(item)
            elif not should_ignore_file(entry.name):
                files.append(item)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}") from exc

    return {
        "path": str(path),
        "parent": str(path.parent) if path.parent != path else "",
        "directories": directories,
        "files": files,
    }


def infer_status(log_text: str) -> str:
    if "✅ [NaturalCC Agent]" in log_text or "任务圆满完成" in log_text:
        return "success"
    if "❌" in log_text or "[错误]" in log_text or "失败" in log_text:
        return "error"
    if log_text.strip():
        return "running"
    return "idle"


def ndjson_event(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


app = FastAPI(title="NaturalCC Agent Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/bootstrap")
async def bootstrap() -> Dict[str, Any]:
    return {
        "default_project_dir": normalize_project_dir(DEFAULT_PROJECT_DIR),
        "models": MODELS,
        "default_model": DEFAULT_MODEL,
        "completion_types": COMPLETION_TYPES,
        "features": [
            {
                "name": m.name,
                "label": m.label,
                "description": m.description,
                "execution_mode": m.execution_mode.value,
            }
            for m in registry.list_plugins()
        ],
        "schemas": registry.get_schemas(),
        "default_feature": "code_completion",
    }


@app.get("/api/models")
async def models() -> Dict[str, Any]:
    return {"models": MODELS, "default_model": DEFAULT_MODEL}


@app.get("/api/workspace/scan")
async def scan_workspace_get(
    project_dir: str = Query(default=DEFAULT_PROJECT_DIR),
    target_files: Optional[List[str]] = Query(default=None),
) -> Dict[str, Any]:
    return build_workspace_snapshot(project_dir, target_files)


@app.post("/api/workspace/scan")
async def scan_workspace_post(request: ScanRequest) -> Dict[str, Any]:
    return build_workspace_snapshot(request.project_dir, request.target_files)


@app.get("/api/browse")
async def browse(path: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return browse_directory(path)


@app.post("/api/command-preview")
async def command_preview(request: AgentRequest) -> Dict[str, str]:
    return {
        "preview_command": build_cli_command_preview(request, preview_only=True),
        "run_command": build_cli_command_preview(request, preview_only=False),
        "provider": detect_provider(request.model or DEFAULT_MODEL),
        "api_key_label": mask_secret(request.api_key),
    }


@app.post("/api/prompt/preview")
async def prompt_preview(request: AgentRequest) -> Dict[str, Any]:
    selected_files = sanitize_target_files(request.target_files)
    project_dir = normalize_project_dir(request.project_dir or DEFAULT_PROJECT_DIR)

    feature = request.feature or "code_completion"
    feature_config = dict(request.feature_config or {})
    if request.symbol is not None:
        feature_config.setdefault("symbol", request.symbol)
    if request.completion_type is not None:
        feature_config.setdefault("completion_type", request.completion_type)
    if request.prefix:
        feature_config.setdefault("prefix", request.prefix)

    plugin = registry.get(feature)
    if plugin is not None:
        context = ExecutionContext(
            project_dir=project_dir,
            target_files=selected_files,
            instruction=request.instruction or "",
            model=request.model or DEFAULT_MODEL,
            api_key=request.api_key,
            feature_config={"feature": feature, **feature_config},
            uploaded_files={},
            symbol=request.symbol,
            completion_type=request.completion_type,
            prefix=request.prefix,
        )
        preview_text = plugin.preview(context)
    else:
        preview_text = preview_prompt(
            target_files=selected_files,
            user_instruction=request.instruction or "",
            model=request.model or DEFAULT_MODEL,
            api_key=request.api_key or None,
            project_dir=project_dir,
            symbol=(request.symbol or None),
            completion_type=(request.completion_type or None),
            prefix=request.prefix or "",
        )
    status = "error" if preview_text.startswith("❌") else "success"
    return {
        "status": status,
        "log": preview_text,
        "command": build_cli_command_preview(request, preview_only=True),
        "primary_target": selected_files[0] if selected_files else "",
    }


async def _stream_with_context(context: ExecutionContext) -> AsyncIterable[str]:
    for event in dispatcher.dispatch(context):
        yield event


@app.post("/api/run")
async def run_agent(request: Request) -> StreamingResponse:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        project_dir = str(form.get("project_dir", DEFAULT_PROJECT_DIR))
        target_files_raw = str(form.get("target_files", "[]"))
        instruction = str(form.get("instruction", ""))
        model = str(form.get("model", DEFAULT_MODEL))
        api_key = form.get("api_key")
        feature = str(form.get("feature", "code_completion"))
        feature_config_raw = str(form.get("feature_config", "{}"))

        target_files_list = json.loads(target_files_raw) if target_files_raw else []
        feature_config_dict = json.loads(feature_config_raw) if feature_config_raw else {}

        uploaded_files = {}
        for key in form:
            value = form[key]
            if isinstance(value, UploadFile):
                uploaded_files[key] = value

        context = ExecutionContext(
            project_dir=normalize_project_dir(project_dir),
            target_files=sanitize_target_files(target_files_list),
            instruction=instruction,
            model=model,
            api_key=api_key,
            feature_config={"feature": feature, **feature_config_dict},
            uploaded_files=uploaded_files,
        )

        return StreamingResponse(_stream_with_context(context), media_type="application/x-ndjson")

    # JSON fallback (backward compatible)
    body = await request.json()
    agent_request = AgentRequest(**body)
    selected_files = sanitize_target_files(agent_request.target_files)

    feature = agent_request.feature or "code_completion"
    feature_config = dict(agent_request.feature_config or {})
    if agent_request.symbol is not None:
        feature_config.setdefault("symbol", agent_request.symbol)
    if agent_request.completion_type is not None:
        feature_config.setdefault("completion_type", agent_request.completion_type)
    if agent_request.prefix:
        feature_config.setdefault("prefix", agent_request.prefix)

    context = ExecutionContext(
        project_dir=normalize_project_dir(agent_request.project_dir or DEFAULT_PROJECT_DIR),
        target_files=selected_files,
        instruction=agent_request.instruction or "",
        model=agent_request.model or DEFAULT_MODEL,
        api_key=agent_request.api_key,
        feature_config={"feature": feature, **feature_config},
        uploaded_files={},
        symbol=agent_request.symbol,
        completion_type=agent_request.completion_type,
        prefix=agent_request.prefix,
    )

    return StreamingResponse(_stream_with_context(context), media_type="application/x-ndjson")


DIST_DIR = Path(__file__).resolve().parent / "webui" / "dist"

if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    index_file = DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="webui/dist not found. Run `cd webui && npm run build` or use the Vite dev server.",
        )
    return FileResponse(index_file)


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:
    candidate = DIST_DIR / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    return await index()


def main() -> None:
    parser = argparse.ArgumentParser(description="NaturalCC Agent web app")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", default=7860, type=int, help="Bind port")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
