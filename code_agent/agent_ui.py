import os
import sys
from pathlib import Path

import gradio as gr

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from code_agent.aider_runner import run_aider_stream, preview_prompt
else:
    from .aider_runner import run_aider_stream, preview_prompt


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


def get_local_files(root_dir="."):
    """遍历当前目录，提供给用户选择已有文件的列表"""
    file_list = []
    ignore_dirs = {
        ".git",
        "__pycache__",
        "venv",
        ".venv",
        ".aider.tags.cache.v3",
        "datasets",
        "node_modules",
        ".idea",
        ".vscode",
    }

    if not os.path.isdir(root_dir):
        return []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for f in filenames:
            if not f.startswith("."):
                full_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                file_list.append(full_path.replace("\\", "/"))
    return sorted(file_list)


def refresh_file_choices(project_dir: str):
    if not project_dir or not os.path.isdir(project_dir):
        return gr.update(choices=[], value=[])
    return gr.update(choices=get_local_files(project_dir), value=[])


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
                    label="📁 项目根目录",
                    placeholder="例如: /path/to/your/c_project",
                    info="必须填写，用于加载 CProjectParser 解析结果"
                )

                refresh_btn = gr.Button("🔄 刷新项目文件列表")

                target_files_input = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    allow_custom_value=True,
                    label="📄 Aider 目标文件 (支持多选及新建)",
                    info="这些文件会交给 Aider 实际修改，同时第一个目标文件会作为解析目标文件。点击刷新后可从项目目录中选择。"
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

            with gr.Group():
                symbol_input = gr.Textbox(
                    label="🔤 目标符号 symbol（可选）",
                    placeholder="例如: factory / winsize / node",
                    info="可留空。程序会尝试从开发指令中自动识别，例如“补全 parse_flags 函数”会自动识别 parse_flags。"
                )

                completion_type_input = gr.Dropdown(
                    choices=["", "member", "variable", "function", "function_body", "type"],
                    value="",
                    label="🧩 补全类型（可选）",
                    info="可留空。程序会自动推断，补全函数默认按函数体实现处理。"
                )

                prefix_input = gr.Textbox(
                    label="🔎 前缀过滤 prefix（可选）",
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

    refresh_btn.click(
        fn=refresh_file_choices,
        inputs=[project_dir_input],
        outputs=[target_files_input]
    )

    preview_btn.click(
        fn=preview_prompt,
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
        fn=run_aider_stream,
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
