import os
import shlex
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import List, Sequence, Tuple

import gradio as gr

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
    from code_agent.agent_ui_assets import APP_HEADER_HTML, CUSTOM_CSS, USAGE_GUIDE
    from code_agent.agent_ui_bindings import CallbackHandlers, UIComponents, bind_ui_events
else:
    from .aider_runner import (
        DEFAULT_PROJECT_DIR,
        detect_provider,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )
    from .agent_ui_assets import APP_HEADER_HTML, CUSTOM_CSS, USAGE_GUIDE
    from .agent_ui_bindings import CallbackHandlers, UIComponents, bind_ui_events


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
PARSABLE_SOURCE_EXTENSIONS = {".c", ".cpp", ".h", ".hpp"}


def should_ignore_file(filename: str) -> bool:
    """Return whether a file should be excluded from the workspace file list."""
    return filename.startswith(".") or Path(filename).suffix.lower() in IGNORE_FILE_EXTENSIONS


def get_local_files(root_dir: str = DEFAULT_PROJECT_DIR) -> List[str]:
    """Scan the project tree and collect visible files relative to the project root."""
    normalized_root = normalize_project_dir(root_dir)
    if not os.path.isdir(normalized_root):
        return []

    file_list = []
    for dirpath, dirnames, filenames in os.walk(normalized_root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in IGNORE_DIRS]
        for filename in filenames:
            if should_ignore_file(filename):
                continue
            full_path = os.path.relpath(os.path.join(dirpath, filename), normalized_root)
            file_list.append(full_path.replace("\\", "/"))
    return sorted(file_list, key=str.casefold)


def sanitize_target_files(target_files) -> List[str]:
    """Normalize the target file input into a stable, deduplicated relative path list."""
    if target_files is None:
        return []

    if isinstance(target_files, str):
        target_files = [target_files]

    result: List[str] = []
    seen = set()
    for item in target_files:
        normalized_item = str(item).strip().replace("\\", "/")
        if not normalized_item:
            continue
        if Path(normalized_item).suffix.lower() in IGNORE_FILE_EXTENSIONS:
            continue
        if normalized_item in seen:
            continue
        result.append(normalized_item)
        seen.add(normalized_item)
    return result


def merge_file_choices(file_choices: Sequence[str], target_files) -> Tuple[List[str], List[str]]:
    """Merge scanned file choices with manually selected files that may not be in the current scan."""
    selected_files = sanitize_target_files(target_files)
    merged_choices = list(file_choices)
    seen = set(merged_choices)
    for item in selected_files:
        if item in seen:
            continue
        merged_choices.append(item)
        seen.add(item)
    return merged_choices, selected_files


def now_label() -> str:
    """Return the current local time label used by compact runtime summaries."""
    return datetime.now().strftime("%H:%M:%S")


def mask_secret(secret: str) -> str:
    """Mask a secret value before reflecting it back to the user interface."""
    if not secret:
        return "未填写"
    if len(secret) <= 8:
        return f"{secret[:2]}***"
    return f"{secret[:4]}...{secret[-4:]}"


def truncate_text(text: str, limit: int = 120) -> str:
    """Trim long text for dense cards while keeping the leading context visible."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def sanitize_markdown_inline(text: str) -> str:
    """Escape inline markdown delimiters before embedding compact text values."""
    return (text or "").replace("`", "'")


def sanitize_markdown_block(text: str) -> str:
    """Escape fenced markdown delimiters before embedding raw logs."""
    return (text or "").replace("```", "'''")


def render_signal_card(title: str, body: str, tone: str = "info", eyebrow: str = "状态") -> str:
    """Render a compact colored summary card used by multiple panes."""
    return f"""
    <div class="signal-card signal-{tone}">
        <div class="signal-eyebrow">{escape(eyebrow)}</div>
        <div class="signal-title">{escape(title)}</div>
        <div class="signal-copy">{escape(body)}</div>
    </div>
    """


def render_list_notice(title: str, messages: Sequence[str], tone: str = "info") -> str:
    """Render a bulleted validation or configuration notice block."""
    items_html = "".join(f"<li>{escape(message)}</li>" for message in messages)
    return f"""
    <div class="signal-card signal-{tone}">
        <div class="signal-eyebrow">{escape(title)}</div>
        <ul class="signal-list">{items_html}</ul>
    </div>
    """


def render_metric_grid(items: Sequence[Tuple[str, str]]) -> str:
    """Render a dense metric grid for project and runtime overviews."""
    cards = "".join(
        f"""
        <div class="metric-card">
            <span class="metric-label">{escape(label)}</span>
            <strong class="metric-value">{escape(value)}</strong>
        </div>
        """
        for label, value in items
    )
    return f'<div class="metric-grid">{cards}</div>'


def render_key_value_rows(items: Sequence[Tuple[str, str]]) -> str:
    """Render compact key-value rows for detailed summaries."""
    rows = "".join(
        f"""
        <div class="info-row">
            <span>{escape(label)}</span>
            <strong>{escape(value)}</strong>
        </div>
        """
        for label, value in items
    )
    return f'<div class="info-rows">{rows}</div>'


def build_workspace_bundle(
    project_dir: str,
    target_files,
    model: str,
    api_key: str,
    symbol: str,
    completion_type: str,
    prefix: str,
    instruction: str = "",
):
    """Assemble the current workspace snapshot shared by the UI summary builders."""
    normalized_root = normalize_project_dir(project_dir)
    file_choices = get_local_files(normalized_root) if os.path.isdir(normalized_root) else []
    merged_choices, selected_files = merge_file_choices(file_choices, target_files)
    return {
        "normalized_root": normalized_root,
        "file_choices": file_choices,
        "merged_choices": merged_choices,
        "selected_files": selected_files,
        "model": (model or "").strip(),
        "api_key": (api_key or "").strip(),
        "symbol": (symbol or "").strip(),
        "completion_type": (completion_type or "").strip(),
        "prefix": (prefix or "").strip(),
        "instruction": (instruction or "").strip(),
    }


def build_config_banner(bundle) -> str:
    """Build the validation banner shown at the top of the left card."""
    normalized_root = bundle["normalized_root"]
    file_choices = bundle["file_choices"]
    selected_files = bundle["selected_files"]
    model = bundle["model"]
    api_key = bundle["api_key"]

    if not os.path.isdir(normalized_root):
        return render_list_notice(
            "配置反馈",
            [
                "项目根目录不可访问。",
                "请先确认路径是否存在。",
                "修正后再刷新工程索引。",
            ],
            tone="error",
        )

    provider = detect_provider(model) if model else "未推断"
    messages = [
        f"可见文件 {len(file_choices)} 个。",
        f"目标文件 {len(selected_files)} 个，主解析文件: {selected_files[0] if selected_files else '未选择'}。",
        f"模型 provider: {provider}。",
    ]

    if not selected_files:
        return render_list_notice("配置反馈", messages + ["尚未选择目标文件。"], tone="warning")

    if not model:
        return render_list_notice("配置反馈", messages + ["模型不能为空。"], tone="error")

    if not api_key:
        messages.append("API Key 留空时会回退到环境变量。")

    return render_list_notice("配置反馈", messages, tone="success")


def build_project_overview(bundle) -> str:
    """Build the advanced project overview card for the left advanced pane."""
    normalized_root = bundle["normalized_root"]
    file_choices = bundle["file_choices"]
    selected_files = bundle["selected_files"]
    model = bundle["model"]
    provider = detect_provider(model) if model else "未推断"
    code_files = [path for path in file_choices if Path(path).suffix.lower() in PARSABLE_SOURCE_EXTENSIONS]
    new_files = [path for path in selected_files if path not in set(file_choices)]

    metrics_html = render_metric_grid(
        [
            ("可见文件", str(len(file_choices))),
            ("C/C++ 候选", str(len(code_files))),
            ("目标文件", str(len(selected_files))),
            ("自定义路径", str(len(new_files))),
        ]
    )

    rows_html = render_key_value_rows(
        [
            ("项目根目录", normalized_root),
            ("首目标文件", selected_files[0] if selected_files else "未选择"),
            ("模型 provider", provider),
            ("忽略目录", ", ".join(sorted(IGNORE_DIRS))),
        ]
    )

    return f"""
    <div class="section-copy">
        <h3>工程概览</h3>
        <p>当前工程扫描结果。</p>
    </div>
    {metrics_html}
    {rows_html}
    """


def build_advanced_control_summary(bundle) -> str:
    """Build the compact summary of symbol, completion type, and prefix controls."""
    selected_files = bundle["selected_files"]
    symbol = bundle["symbol"]
    completion_type = bundle["completion_type"]
    prefix = bundle["prefix"]
    model = bundle["model"]

    rows_html = render_key_value_rows(
        [
            ("当前模型", model or "未填写"),
            ("目标 symbol", symbol or "自动从开发指令提取"),
            ("completion_type", completion_type or "自动推断"),
            ("prefix 过滤", prefix or "未启用"),
            ("主解析文件", selected_files[0] if selected_files else "未选择"),
        ]
    )

    return f"""
    <div class="section-copy">
        <h3>进阶控制</h3>
        <p>空值会交给后端自动推断。</p>
    </div>
    {rows_html}
    """


def build_instruction_feedback(instruction: str, target_files, action: str = "idle") -> str:
    """Build the task summary card shown in the right instruction pane."""
    selected_files = sanitize_target_files(target_files)
    instruction = (instruction or "").strip()

    if not instruction:
        return render_signal_card(
            "等待开发指令",
            "输入任务后可直接预览或执行。",
            tone="warning",
            eyebrow="开发指令",
        )

    action_map = {
        "idle": "可以预览，也可以直接执行。",
        "preview": "本次只生成 prompt 预览。",
        "generate": "本次会调用 Aider 执行修改。",
        "cleared": "任务区域已清空。",
    }
    parse_target = selected_files[0] if selected_files else "未选择目标文件"
    body = (
        f"指令长度 {len(instruction)} 字；主解析文件: {parse_target}。"
        f"{action_map.get(action, action_map['idle'])}"
    )
    return render_signal_card(
        "开发指令已准备",
        body,
        tone="success" if selected_files else "info",
        eyebrow="开发指令",
    )


def build_cli_command_preview(
    target_files,
    user_instruction: str,
    model: str,
    api_key: str,
    project_dir: str,
    symbol: str,
    completion_type: str,
    prefix: str,
    preview_only: bool,
) -> str:
    """Build the reusable CLI command preview that mirrors the current UI inputs."""
    selected_files = sanitize_target_files(target_files)
    user_instruction = (user_instruction or "").strip()
    model = (model or "").strip()
    api_key = (api_key or "").strip()
    symbol = (symbol or "").strip()
    completion_type = (completion_type or "").strip()
    prefix = (prefix or "").strip()
    command = ["python", "aider_runner.py", "-dir", normalize_project_dir(project_dir or DEFAULT_PROJECT_DIR)]

    if selected_files:
        command.extend(["-f", *selected_files])
    if user_instruction:
        command.extend(["-i", user_instruction])
    else:
        command.extend(["-i", "<请填写开发指令>"])

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
        "# UI 等效命令",
        "# 第一个目标文件会作为 NaturalCC 的主解析文件。",
    ]
    if not api_key:
        header.append("# 未填写 API Key 时，命令会回退到环境变量。")

    formatted_command = " \\\n  ".join(shlex.quote(part) for part in command)
    return "\n".join(header + [formatted_command])


def build_command_summary(mode: str, project_dir: str, target_files, model: str) -> str:
    """Build the summary card shown above the console outputs."""
    selected_files = sanitize_target_files(target_files)
    mode_map = {
        "idle": ("命令行输出待命", "等待触发操作。", "info"),
        "preview": ("Prompt 预览已生成", "当前内容来自 preview 流程。", "success"),
        "generate": ("Aider 执行中", "这里展示实时日志。", "warning"),
        "success": ("Aider 执行完成", "执行回显已保留。", "success"),
        "error": ("执行失败", "请检查最近输出。", "error"),
        "cleared": ("结果区域已清空", "可以重新发起任务。", "info"),
    }
    title, body, tone = mode_map.get(mode, mode_map["idle"])
    metrics = render_metric_grid(
        [
            ("更新时间", now_label()),
            ("模型", model or "未填写"),
            ("目标文件", str(len(selected_files))),
            ("项目根目录", truncate_text(normalize_project_dir(project_dir), 36) or "未设置"),
        ]
    )
    banner = render_signal_card(title, body, tone=tone, eyebrow="命令行生成内容")
    return banner + metrics


def infer_runtime_state(log_text: str) -> Tuple[str, str, str]:
    """Infer a coarse execution state from the latest preview or Aider log text."""
    if "✅ [NaturalCC Agent]" in log_text or "任务圆满完成" in log_text:
        return "success", "执行完成", "Aider 已完成本轮修改，日志已稳定。"
    if "❌" in log_text or "[错误]" in log_text or "失败" in log_text:
        return "error", "执行失败", "执行链路中出现错误，建议先查看最近输出定位问题。"
    if log_text.strip():
        return "warning", "执行中", "NaturalCC 正在准备上下文或 Aider 正在回放执行日志。"
    return "info", "等待开始", "尚未触发任何预览或生成动作。"


def build_status_detail(
    mode: str,
    project_dir: str,
    target_files,
    model: str,
    instruction: str,
    log_text: str,
) -> str:
    """Build the detailed markdown status panel with the latest output excerpt."""
    selected_files = sanitize_target_files(target_files)
    phase_map = {
        "idle": "待命",
        "preview": "Prompt 预览",
        "generate": "执行中",
        "success": "执行完成",
        "error": "执行失败",
        "cleared": "已清空",
    }
    phase = phase_map.get(mode, "待命")
    recent_lines = [line for line in log_text.strip().splitlines() if line.strip()]
    recent_excerpt = "\n".join(recent_lines[-10:]) if recent_lines else "暂无输出。"

    summary_lines = [
        "### 当前任务状态",
        f"- 阶段: `{phase}`",
        f"- 项目: `{sanitize_markdown_inline(truncate_text(normalize_project_dir(project_dir), 56))}`",
        f"- 主解析文件: `{sanitize_markdown_inline(selected_files[0] if selected_files else '未选择')}`",
        f"- 目标文件数: `{len(selected_files)}`",
        f"- 模型: `{sanitize_markdown_inline(model or '未填写')}`",
        f"- 指令摘要: `{sanitize_markdown_inline(truncate_text(instruction or '未填写开发指令', 80))}`",
        "",
        "### 最近输出",
        "```text",
        sanitize_markdown_block(recent_excerpt),
        "```",
    ]
    return "\n".join(summary_lines)


def refresh_workspace_views(
    project_dir: str,
    target_files,
    model: str,
    api_key: str,
    symbol: str,
    completion_type: str,
    prefix: str,
    instruction: str,
):
    """Refresh file choices and summary cards after a workspace-level change."""
    bundle = build_workspace_bundle(
        project_dir=project_dir,
        target_files=target_files,
        model=model,
        api_key=api_key,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
        instruction=instruction,
    )
    return (
        gr.update(value=bundle["normalized_root"]),
        gr.update(choices=bundle["merged_choices"], value=bundle["selected_files"]),
        build_config_banner(bundle),
        build_project_overview(bundle),
        build_advanced_control_summary(bundle),
        build_instruction_feedback(bundle["instruction"], bundle["selected_files"]),
    )


def sync_workspace_panels(
    project_dir: str,
    target_files,
    model: str,
    api_key: str,
    symbol: str,
    completion_type: str,
    prefix: str,
    instruction: str,
):
    """Update lightweight workspace summaries after config inputs change."""
    bundle = build_workspace_bundle(
        project_dir=project_dir,
        target_files=target_files,
        model=model,
        api_key=api_key,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
        instruction=instruction,
    )
    return (
        build_config_banner(bundle),
        build_project_overview(bundle),
        build_advanced_control_summary(bundle),
        build_instruction_feedback(bundle["instruction"], bundle["selected_files"]),
    )


def sync_advanced_controls(model: str, target_files, symbol: str, completion_type: str, prefix: str):
    """Rebuild the advanced-control summary card."""
    bundle = {
        "selected_files": sanitize_target_files(target_files),
        "model": (model or "").strip(),
        "symbol": (symbol or "").strip(),
        "completion_type": (completion_type or "").strip(),
        "prefix": (prefix or "").strip(),
    }
    return build_advanced_control_summary(bundle)


def choose_project_dir(
    current_dir: str,
    target_files,
    model: str,
    api_key: str,
    symbol: str,
    completion_type: str,
    prefix: str,
    instruction: str,
):
    """Open a native directory picker and refresh dependent UI panels."""
    fallback_dir = normalize_project_dir(current_dir)
    root = None
    selected_dir = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(initialdir=fallback_dir)
    except Exception:
        selected_dir = fallback_dir
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    return refresh_workspace_views(
        project_dir=selected_dir or fallback_dir,
        target_files=target_files,
        model=model,
        api_key=api_key,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
        instruction=instruction,
    )


def switch_left_view(view_name: str):
    """Toggle the left-side card between general and advanced views."""
    return (
        gr.update(visible=view_name == "general"),
        gr.update(visible=view_name == "advanced"),
        gr.update(variant="primary" if view_name == "general" else "secondary"),
        gr.update(variant="primary" if view_name == "advanced" else "secondary"),
    )


def switch_right_view(view_name: str):
    """Toggle the right-side card between instruction, console, and status views."""
    return (
        gr.update(visible=view_name == "instruction"),
        gr.update(visible=view_name == "console"),
        gr.update(visible=view_name == "status"),
        gr.update(variant="primary" if view_name == "instruction" else "secondary"),
        gr.update(variant="primary" if view_name == "console" else "secondary"),
        gr.update(variant="primary" if view_name == "status" else "secondary"),
    )


def preview_prompt_from_ui(
    target_files,
    user_instruction: str,
    model: str,
    api_key: str,
    project_dir: str,
    symbol: str,
    completion_type: str,
    prefix: str,
):
    """Generate a prompt preview and switch the right-side card to console output."""
    selected_files = sanitize_target_files(target_files)
    user_instruction = user_instruction or ""
    model = model or ""
    api_key = api_key or ""
    project_dir = project_dir or DEFAULT_PROJECT_DIR
    symbol = symbol or ""
    completion_type = completion_type or ""
    prefix = prefix or ""
    preview_text = preview_prompt(
        target_files=selected_files,
        user_instruction=user_instruction,
        model=model,
        api_key=api_key,
        project_dir=normalize_project_dir(project_dir),
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
    )
    tone = "error" if preview_text.startswith("❌") else "preview"
    command_mode = "error" if tone == "error" else "preview"
    return (
        build_instruction_feedback(user_instruction, selected_files, action="preview"),
        build_command_summary(command_mode, project_dir, selected_files, model),
        build_cli_command_preview(
            target_files=selected_files,
            user_instruction=user_instruction,
            model=model,
            api_key=api_key,
            project_dir=project_dir,
            symbol=symbol,
            completion_type=completion_type,
            prefix=prefix,
            preview_only=True,
        ),
        preview_text,
        build_status_detail(
            mode="error" if tone == "error" else "preview",
            project_dir=project_dir,
            target_files=selected_files,
            model=model,
            instruction=user_instruction,
            log_text=preview_text,
        ),
        *switch_right_view("console"),
    )


def run_aider_stream_from_ui(
    target_files,
    user_instruction: str,
    model: str,
    api_key: str,
    project_dir: str,
    symbol: str,
    completion_type: str,
    prefix: str,
):
    """Stream Aider execution logs while keeping the console pane active."""
    selected_files = sanitize_target_files(target_files)
    user_instruction = user_instruction or ""
    model = model or ""
    api_key = api_key or ""
    project_dir = project_dir or DEFAULT_PROJECT_DIR
    symbol = symbol or ""
    completion_type = completion_type or ""
    prefix = prefix or ""
    instruction_feedback = build_instruction_feedback(user_instruction, selected_files, action="generate")
    cli_preview = build_cli_command_preview(
        target_files=selected_files,
        user_instruction=user_instruction,
        model=model,
        api_key=api_key,
        project_dir=project_dir,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
        preview_only=False,
    )

    yield (
        instruction_feedback,
        build_command_summary("generate", project_dir, selected_files, model),
        cli_preview,
        "正在准备执行...\n",
        build_status_detail(
            mode="generate",
            project_dir=project_dir,
            target_files=selected_files,
            model=model,
            instruction=user_instruction,
            log_text="正在准备执行...",
        ),
        *switch_right_view("console"),
    )

    for output_log in run_aider_stream(
        target_files=selected_files,
        user_instruction=user_instruction,
        model=model,
        api_key=api_key,
        project_dir=normalize_project_dir(project_dir),
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
    ):
        tone, _title, _body = infer_runtime_state(output_log)
        final_mode = {"success": "success", "error": "error"}.get(tone, "generate")
        yield (
            instruction_feedback,
            build_command_summary(final_mode, project_dir, selected_files, model),
            cli_preview,
            output_log,
            build_status_detail(
                mode=final_mode,
                project_dir=project_dir,
                target_files=selected_files,
                model=model,
                instruction=user_instruction,
                log_text=output_log,
            ),
            *switch_right_view("console"),
        )


def clear_task_view(target_files, model: str, project_dir: str):
    """Reset right-side task inputs and outputs without touching the left-side config."""
    selected_files = sanitize_target_files(target_files)
    return (
        "",
        build_instruction_feedback("", selected_files, action="cleared"),
        build_command_summary("cleared", project_dir, selected_files, model),
        "",
        "",
        build_status_detail(
            mode="cleared",
            project_dir=project_dir,
            target_files=selected_files,
            model=model,
            instruction="",
            log_text="任务内容已清空。",
        ),
        *switch_right_view("console"),
    )

custom_theme = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)


initial_bundle = build_workspace_bundle(
    project_dir=DEFAULT_PROJECT_DIR,
    target_files=[],
    model=DEFAULT_MODEL,
    api_key="",
    symbol="",
    completion_type="",
    prefix="",
    instruction="",
)
INITIAL_COMMAND_SUMMARY = build_command_summary(
    mode="idle",
    project_dir=initial_bundle["normalized_root"],
    target_files=initial_bundle["selected_files"],
    model=DEFAULT_MODEL,
)
INITIAL_STATUS_DETAIL = build_status_detail(
    mode="idle",
    project_dir=initial_bundle["normalized_root"],
    target_files=initial_bundle["selected_files"],
    model=DEFAULT_MODEL,
    instruction="",
    log_text="尚未触发预览或执行操作。",
)

with gr.Blocks(title="NaturalCC Agent", fill_width=True, fill_height=True) as demo:
    gr.HTML(APP_HEADER_HTML, elem_classes=["header-shell"])

    with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
        with gr.Column(scale=5, min_width=380, elem_id="config-panel", elem_classes=["panel-shell"]):
            gr.HTML(
                """
                <div class="panel-title">
                    <h2>项目与模型配置</h2>
                    <p>先配路径、目标文件和模型。</p>
                </div>
                """
            )
            with gr.Column(elem_classes=["panel-body"]):
                config_banner = gr.HTML(build_config_banner(initial_bundle), elem_classes=["panel-banner"])

                with gr.Column(visible=True, elem_classes=["panel-view"]) as general_config_view:
                    with gr.Column():
                        project_dir_input = gr.Textbox(
                            value=initial_bundle["normalized_root"],
                            label="项目根目录",
                            placeholder="/path/to/project",
                            info="默认当前工作目录。",
                        )
                        with gr.Row(elem_classes=["action-row"]):
                            choose_dir_btn = gr.Button("选择目录", variant="secondary")
                            refresh_btn = gr.Button("刷新工程索引", variant="secondary")

                        target_files_input = gr.Dropdown(
                            choices=initial_bundle["merged_choices"],
                            value=initial_bundle["selected_files"],
                            multiselect=True,
                            allow_custom_value=True,
                            label="Aider 目标文件",
                            info="首个目标文件用于解析。",
                        )

                    with gr.Column():
                        api_key_input = gr.Textbox(
                            label="API Key",
                            placeholder="sk-or-v1-... / sk-...",
                            type="password",
                            info="留空时回退到环境变量。",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=MODELS,
                            value=DEFAULT_MODEL,
                            allow_custom_value=True,
                            label="模型选择",
                            info="可选或直接输入。",
                        )

                with gr.Column(visible=False, elem_classes=["panel-view"]) as advanced_config_view:
                    project_overview = gr.HTML(build_project_overview(initial_bundle))
                    with gr.Column():
                        symbol_input = gr.Textbox(
                            label="目标 symbol",
                            placeholder="例如: parse_flags / terminal_state / node",
                            info="留空自动抽取。",
                        )
                        completion_type_input = gr.Dropdown(
                            choices=["", "member", "variable", "function", "function_body", "type"],
                            value="",
                            label="completion_type",
                            info="留空自动推断。",
                        )
                        prefix_input = gr.Textbox(
                            label="prefix 过滤",
                            placeholder="例如: get_term / win / parse_",
                            info="按前缀筛选时填写。",
                        )
                    advanced_control_summary = gr.HTML(build_advanced_control_summary(initial_bundle))

            with gr.Row(elem_classes=["panel-footer"]):
                general_config_btn = gr.Button("常规配置", variant="primary", elem_classes=["footer-btn"])
                advanced_config_btn = gr.Button("工程概览 / 进阶", variant="secondary", elem_classes=["footer-btn"])

        with gr.Column(scale=7, min_width=480, elem_id="execution-panel", elem_classes=["panel-shell"]):
            gr.HTML(
                """
                <div class="panel-title">
                    <h2>任务与执行</h2>
                    <p>开发指令、输出和状态分开查看。</p>
                </div>
                """
            )
            with gr.Column(elem_classes=["panel-body"]):
                with gr.Column(visible=True, elem_classes=["panel-view"]) as instruction_view:
                    instruction_feedback = gr.HTML(build_instruction_feedback("", initial_bundle["selected_files"]))
                    instruction_input = gr.Textbox(
                        label="开发指令",
                        placeholder="例如：补全 parse_flags 函数实现，并遵循项目已有错误处理风格。",
                        lines=5,
                        info="尽量写清楚目标行为。",
                    )
                    with gr.Row(elem_classes=["action-row"]):
                        preview_btn = gr.Button("预览 Prompt", variant="secondary")
                        run_btn = gr.Button("生成并执行", variant="primary")
                        clear_btn = gr.Button("清空任务", variant="secondary")
                    gr.Markdown("预览只看 prompt；执行会调用 `aider`。")

                with gr.Column(visible=False, elem_classes=["panel-view"]) as console_view:
                    command_summary = gr.HTML(INITIAL_COMMAND_SUMMARY)
                    with gr.Row(elem_classes=["console-content-row"]):
                        with gr.Column(scale=7, min_width=360):
                            generated_content_output = gr.Textbox(
                                label="命令行生成内容",
                                lines=13,
                                max_lines=16,
                                interactive=False,
                                autoscroll=True,
                                buttons=["copy"],
                                placeholder="这里会显示 prompt 预览文本或 Aider 的执行日志。",
                            )
                        with gr.Column(scale=5, min_width=320):
                            command_preview_output = gr.Textbox(
                                label="等效命令行",
                                lines=13,
                                max_lines=16,
                                interactive=False,
                                autoscroll=False,
                                buttons=["copy"],
                                placeholder="触发预览或生成后，这里会展示可复用的命令行草案。",
                            )

                with gr.Column(visible=False, elem_classes=["panel-view"]) as status_view:
                    gr.Markdown(USAGE_GUIDE, elem_classes=["status-markdown"])
                    status_detail = gr.Markdown(INITIAL_STATUS_DETAIL, elem_classes=["status-markdown"])

            with gr.Row(elem_classes=["panel-footer"]):
                instruction_panel_btn = gr.Button("开发指令", variant="primary", elem_classes=["footer-btn"])
                console_panel_btn = gr.Button("命令行生成内容", variant="secondary", elem_classes=["footer-btn"])
                status_panel_btn = gr.Button("操作说明 / 状态", variant="secondary", elem_classes=["footer-btn"])

    bind_ui_events(
        ui=UIComponents(
            general_config_view=general_config_view,
            advanced_config_view=advanced_config_view,
            general_config_btn=general_config_btn,
            advanced_config_btn=advanced_config_btn,
            instruction_view=instruction_view,
            console_view=console_view,
            status_view=status_view,
            instruction_panel_btn=instruction_panel_btn,
            console_panel_btn=console_panel_btn,
            status_panel_btn=status_panel_btn,
            choose_dir_btn=choose_dir_btn,
            refresh_btn=refresh_btn,
            project_dir_input=project_dir_input,
            target_files_input=target_files_input,
            model_dropdown=model_dropdown,
            api_key_input=api_key_input,
            symbol_input=symbol_input,
            completion_type_input=completion_type_input,
            prefix_input=prefix_input,
            instruction_input=instruction_input,
            config_banner=config_banner,
            project_overview=project_overview,
            advanced_control_summary=advanced_control_summary,
            instruction_feedback=instruction_feedback,
            command_summary=command_summary,
            command_preview_output=command_preview_output,
            generated_content_output=generated_content_output,
            status_detail=status_detail,
            preview_btn=preview_btn,
            run_btn=run_btn,
            clear_btn=clear_btn,
        ),
        handlers=CallbackHandlers(
            switch_left_view=switch_left_view,
            switch_right_view=switch_right_view,
            choose_project_dir=choose_project_dir,
            refresh_workspace_views=refresh_workspace_views,
            sync_workspace_panels=sync_workspace_panels,
            sync_advanced_controls=sync_advanced_controls,
            build_instruction_feedback=build_instruction_feedback,
            preview_prompt_from_ui=preview_prompt_from_ui,
            run_aider_stream_from_ui=run_aider_stream_from_ui,
            clear_task_view=clear_task_view,
        ),
    )


if __name__ == "__main__":
    print("正在启动 NaturalCC UI，请在浏览器中打开提供的本地地址...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=custom_theme,
        css=CUSTOM_CSS,
    )
