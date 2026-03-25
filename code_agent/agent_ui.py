import os
import gradio as gr
# 从你刚创建的 backend 文件中引入流式运行函数
from aider_runner import run_aider_stream

# ==========================================
# 1. UI 配置数据
# ==========================================

MODELS =[
    "openrouter/anthropic/claude-3-haiku", "openrouter/anthropic/claude-3.5-sonnet", "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/anthropic/claude-haiku-4.5", "openrouter/anthropic/claude-opus-4", "openrouter/anthropic/claude-opus-4.1",
    "openrouter/anthropic/claude-opus-4.5", "openrouter/anthropic/claude-opus-4.6", "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-sonnet-4.5", "openrouter/anthropic/claude-sonnet-4.6", "openrouter/bytedance/ui-tars-1.5-7b",
    "openrouter/deepseek/deepseek-chat", "openrouter/deepseek/deepseek-chat-v3-0324", "openrouter/deepseek/deepseek-chat-v3-0324:free",
    "openrouter/deepseek/deepseek-chat-v3.1", "openrouter/deepseek/deepseek-chat:free", "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-r1-0528", "openrouter/deepseek/deepseek-r1:free", "openrouter/deepseek/deepseek-v3.2",
    "openrouter/deepseek/deepseek-v3.2-exp", "openrouter/google/gemini-2.0-flash-001", "openrouter/google/gemini-2.0-flash-exp:free",
    "openrouter/google/gemini-2.5", "openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-pro",
    "openrouter/google/gemini-2.5-pro-exp-03-25", "openrouter/google/gemini-2.5-pro-preview-03-25", "openrouter/google/gemini-3-flash-preview",
    "openrouter/google/gemini-3-pro-preview", "openrouter/google/gemini-3.1-pro-preview", "openrouter/gryphe/mythomax-l2-13b",
    "openrouter/mancer/weaver", "openrouter/meta-llama/llama-3-70b-instruct", "openrouter/minimax/minimax-m2",
    "openrouter/minimax/minimax-m2.1", "openrouter/minimax/minimax-m2.5", "openrouter/mistralai/devstral-2512",
    "openrouter/mistralai/ministral-14b-2512", "openrouter/mistralai/ministral-3b-2512", "openrouter/mistralai/ministral-8b-2512",
    "openrouter/mistralai/mistral-7b-instruct", "openrouter/mistralai/mistral-large", "openrouter/mistralai/mistral-large-2512",
    "openrouter/mistralai/mistral-small-3.1-24b-instruct", "openrouter/mistralai/mistral-small-3.2-24b-instruct",
    "openrouter/mistralai/mixtral-8x22b-instruct", "openrouter/moonshotai/kimi-k2.5", "openrouter/openai/gpt-3.5-turbo",
    "openrouter/openai/gpt-3.5-turbo-16k", "openrouter/openai/gpt-4", "openrouter/openai/gpt-4.1", "openrouter/openai/gpt-4.1-mini",
    "openrouter/openai/gpt-4.1-nano", "openrouter/openai/gpt-4o", "openrouter/openai/gpt-4o-2024-05-13", "openrouter/openai/gpt-4o-mini",
    "openrouter/openai/gpt-5", "openrouter/openai/gpt-5-chat", "openrouter/openai/gpt-5-codex", "openrouter/openai/gpt-5-mini",
    "openrouter/openai/gpt-5-nano", "openrouter/openai/gpt-5.1-codex-max", "openrouter/openai/gpt-5.2", "openrouter/openai/gpt-5.2-chat",
    "openrouter/openai/gpt-5.2-codex", "openrouter/openai/gpt-5.2-pro", "openrouter/openai/gpt-oss-120b", "openrouter/openai/gpt-oss-20b",
    "openrouter/openai/o1", "openrouter/openai/o3-mini", "openrouter/openai/o3-mini-high", "openrouter/openrouter/auto",
    "openrouter/openrouter/bodybuilder", "openrouter/openrouter/free", "openrouter/openrouter/optimus-alpha",
    "openrouter/openrouter/quasar-alpha", "openrouter/qwen/qwen-2.5-coder-32b-instruct", "openrouter/qwen/qwen-vl-plus",
    "openrouter/qwen/qwen3-235b-a22b-2507", "openrouter/qwen/qwen3-235b-a22b-thinking-2507", "openrouter/qwen/qwen3-coder",
    "openrouter/qwen/qwen3-coder-plus", "openrouter/qwen/qwen3.5-122b-a10b", "openrouter/qwen/qwen3.5-27b",
    "openrouter/qwen/qwen3.5-35b-a3b", "openrouter/qwen/qwen3.5-397b-a17b", "openrouter/qwen/qwen3.5-flash-02-23",
    "openrouter/qwen/qwen3.5-plus-02-15", "openrouter/switchpoint/router", "openrouter/undi95/remm-slerp-l2-13b",
    "openrouter/x-ai/grok-3-beta", "openrouter/x-ai/grok-3-fast-beta", "openrouter/x-ai/grok-3-mini-beta",
    "openrouter/x-ai/grok-3-mini-fast-beta", "openrouter/x-ai/grok-4", "openrouter/xiaomi/mimo-v2-flash",
    "openrouter/z-ai/glm-4.6", "openrouter/z-ai/glm-4.6:exacto", "openrouter/z-ai/glm-4.7", "openrouter/z-ai/glm-4.7-flash",
    "openrouter/z-ai/glm-5", "vertex_ai-language-models/openrouter/google/gemini-2.5-pro-preview-03-25"
]

def get_local_files(root_dir="."):
    """遍历当前目录，提供给用户选择已有文件的列表"""
    file_list =[]
    ignore_dirs = {".git", "__pycache__", "venv", ".venv", ".aider.tags.cache.v3", "datasets"}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] =[d for d in dirnames if d not in ignore_dirs]
        for f in filenames:
            if not f.startswith("."): # 忽略隐藏文件
                full_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                file_list.append(full_path.replace("\\", "/"))
    return sorted(file_list)


# ==========================================
# 2. 构建 Gradio 前端 UI
# ==========================================

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="NaturalCC Agent") as demo:
    with gr.Row():
         gr.Markdown(
             """
             <div style='text-align: center; margin-bottom: 20px;'>
                 <h1>🤖 NaturalCC - Context-Aware Code Agent</h1>
                 <p style='color: gray;'>结合 NaturalCC 的深度语义图谱与 Aider 智能体的自动化编程助手</p>
             </div>
             """
         )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### ⚙️ Agent 配置")
            
            with gr.Group():
                api_key_input = gr.Textbox(
                    label="🔑 API Key", 
                    placeholder="例如: sk-or-v1-xxxxxx...", 
                    type="password",
                    info="请填写您的 OpenRouter API 密钥，或者留空以使用环境变量 OPENROUTER_API_KEY 中的值"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=MODELS, 
                    value="openrouter/deepseek/deepseek-chat", 
                    label="🧠 选用模型",
                    allow_custom_value=True,
                    info="可直接从下拉框中选择您偏好的代码大模型"
                )

            with gr.Group():
                target_files_input = gr.Dropdown(
                    choices=get_local_files(),
                    multiselect=True,
                    allow_custom_value=True,
                    label="📄 目标文件 (支持多选及新建)",
                    info="点击选择当前目录下已有文件，或直接打字输入新文件路径并按【回车】"
                )
                
        with gr.Column(scale=7):
            gr.Markdown("### 📝 开发指令")
            
            instruction_input = gr.Textbox(
                label="💡 你想让 Agent 帮你在选中文件里做点什么？", 
                placeholder="例如：请帮我将网络请求逻辑重构成异步调用，注意参考 NaturalCC 提供的跨文件图谱结构...", 
                lines=5
            )
            
            run_btn = gr.Button("🚀 立即运行 Agent 并修改代码", variant="primary", size="lg")
            
            gr.Markdown("### 💻 终端实时输出日志")
            
            output_console = gr.Code(
                label="Aider Execution Console", 
                language="shell", 
                lines=15,
                interactive=False
            )

    # 绑定事件，直接调用 aider_runner 里的流式函数
    run_btn.click(
        fn=run_aider_stream,
        inputs=[target_files_input, instruction_input, model_dropdown, api_key_input],
        outputs=[output_console]
    )

if __name__ == "__main__":
    print("🌐 正在启动 NaturalCC UI，请在浏览器中打开提供的本地地址...")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)