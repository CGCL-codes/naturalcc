import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import gradio as gr

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from code_agent.aider_runner import (
        DEFAULT_PROJECT_DIR,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )
else:
    from .aider_runner import (
        DEFAULT_PROJECT_DIR,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )


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


def should_ignore_file(filename: str) -> bool:
    return filename.startswith(".") or Path(filename).suffix.lower() in IGNORE_FILE_EXTENSIONS


def get_local_files(root_dir: str = DEFAULT_PROJECT_DIR) -> List[str]:
    """遍历项目目录，提供给用户选择已有文件的列表。"""
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
    selected_files = sanitize_target_files(target_files)
    merged_choices = list(file_choices)
    seen = set(merged_choices)
    for item in selected_files:
        if item in seen:
            continue
        merged_choices.append(item)
        seen.add(item)
    return merged_choices, selected_files


def refresh_file_choices(project_dir: str, target_files=None):
    normalized_root = normalize_project_dir(project_dir)
    if not os.path.isdir(normalized_root):
        return gr.update(value=normalized_root), gr.update(choices=[], value=[])

    file_choices = get_local_files(normalized_root)
    merged_choices, selected_files = merge_file_choices(file_choices, target_files)
    return (
        gr.update(value=normalized_root),
        gr.update(choices=merged_choices, value=selected_files),
    )


def choose_project_dir(current_dir: str):
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

    return refresh_file_choices(selected_dir or fallback_dir)


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
    return preview_prompt(
        target_files=sanitize_target_files(target_files),
        user_instruction=user_instruction,
        model=model,
        api_key=api_key,
        project_dir=normalize_project_dir(project_dir),
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
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
    yield from run_aider_stream(
        target_files=sanitize_target_files(target_files),
        user_instruction=user_instruction,
        model=model,
        api_key=api_key,
        project_dir=normalize_project_dir(project_dir),
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
    )


custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(theme=custom_theme, title="NaturalCC Agent") as demo:
    with gr.Row():
        gr.Markdown(
            """
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1>🤖 NaturalCC - Context-Aware Code Agent</h1>
                <p style='color: gray;'>结合 NaturalCC 的项目语义解析与 Aider 的自动化代码修改能力</p>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### ⚙️ Agent 配置")

            with gr.Group():
                project_dir_input = gr.Textbox(
                    value=DEFAULT_PROJECT_DIR,
                    label="📁 项目根目录",
                    placeholder="例如: D:/workspace/your_project",
                    info="默认使用当前运行 agent_ui.py 时的目录。也可以点击右侧文件夹按钮重新选择。"
                )

                with gr.Row():
                    choose_dir_btn = gr.Button("📂 选择目录", variant="secondary")
                    refresh_btn = gr.Button("🔄 刷新项目文件列表", variant="secondary")

                target_files_input = gr.Dropdown(
                    choices=get_local_files(DEFAULT_PROJECT_DIR),
                    multiselect=True,
                    allow_custom_value=True,
                    label="📄 Aider 目标文件 (支持多选及新建)",
                    info="会自动忽略 .log、.ps 等垃圾文件。这些文件会交给 Aider 实际修改，同时第一个目标文件会作为解析目标文件。"
                )

            with gr.Group():
                api_key_input = gr.Textbox(
                    label="🔑 API Key",
                    placeholder="例如: sk-or-v1-xxxxxx...",
                    type="password",
                    info="可填写 OpenRouter / OpenAI 密钥，也可留空使用环境变量"
                )

                model_dropdown = gr.Dropdown(
                    choices=MODELS,
                    value="openrouter/deepseek/deepseek-chat",
                    label="🧠 选用模型",
                    allow_custom_value=True,
                    info="可直接从下拉框中选择模型，也可以手动输入"
                )

            with gr.Accordion("🧩 进阶功能", open=False):
                with gr.Group():
                    symbol_input = gr.Textbox(
                        label="🔤 目标符号 symbol",
                        placeholder="例如: factory / winsize / node",
                        info="可留空。程序会尝试从开发指令中自动识别，例如“补全 parse_flags 函数”会自动识别 parse_flags。"
                    )

                    completion_type_input = gr.Dropdown(
                        choices=["", "member", "variable", "function", "function_body", "type"],
                        value="",
                        label="🧩 补全类型",
                        info="可留空。程序会自动推断，补全函数默认按函数体实现处理。"
                    )

                    prefix_input = gr.Textbox(
                        label="🔎 前缀过滤 prefix",
                        placeholder="例如: get_terminal_ / win / co",
                        info="一般可留空。只有在你想按前缀匹配多个候选时才需要填写。"
                    )

        with gr.Column(scale=7):
            gr.Markdown("### 📝 开发指令")

            instruction_input = gr.Textbox(
                label="💡 你想让 Agent 帮你做什么？",
                placeholder="例如：补全 get_terminal_ 开头的函数；或重构当前文件中的终端初始化逻辑，参考项目现有实现风格。",
                lines=6
            )

            with gr.Row():
                preview_btn = gr.Button("👀 仅生成 Prompt 预览", variant="secondary")
                run_btn = gr.Button("🚀 立即运行 Agent 并修改代码", variant="primary")

            gr.Markdown("### 💻 输出")
            output_console = gr.Code(
                label="Execution Console / Prompt Preview",
                language="shell",
                lines=24,
                interactive=False
            )

    choose_dir_btn.click(
        fn=choose_project_dir,
        inputs=[project_dir_input],
        outputs=[project_dir_input, target_files_input]
    )

    refresh_btn.click(
        fn=refresh_file_choices,
        inputs=[project_dir_input, target_files_input],
        outputs=[project_dir_input, target_files_input]
    )

    project_dir_input.change(
        fn=refresh_file_choices,
        inputs=[project_dir_input, target_files_input],
        outputs=[project_dir_input, target_files_input]
    )

    preview_btn.click(
        fn=preview_prompt_from_ui,
        inputs=[
            target_files_input,
            instruction_input,
            model_dropdown,
            api_key_input,
            project_dir_input,
            symbol_input,
            completion_type_input,
            prefix_input,
        ],
        outputs=[output_console]
    )

    run_btn.click(
        fn=run_aider_stream_from_ui,
        inputs=[
            target_files_input,
            instruction_input,
            model_dropdown,
            api_key_input,
            project_dir_input,
            symbol_input,
            completion_type_input,
            prefix_input,
        ],
        outputs=[output_console]
    )

if __name__ == "__main__":
    print("🌐 正在启动 NaturalCC UI，请在浏览器中打开提供的本地地址...")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
