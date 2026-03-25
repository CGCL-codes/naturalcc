import os
import subprocess
import tempfile
import re
import argparse

# 用于去除 Aider 输出中带有的 ANSI 颜色终端控制字符 (给 UI 使用)
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def dummy_get_naturalcc_prompt(filepath: str) -> str:
    """ 模拟你的 NaturalCC API 上下文提取 """
    return f"""
    /* NaturalCC Context Map for {filepath} */
    /* Include Dependencies Found: */
    // db_connect.h: void connect_to_db(const char* host);
    // utils.c: int retry_count = 3;
    /* End Context */
    """

def build_aider_context_and_command(target_files, user_instruction, model, api_key):
    """
    通用函数：准备 NaturalCC 上下文并构建 Aider 启动命令
    返回: (aider_command, prompt_file_path, init_log)
    """
    init_log = ""
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            init_log += "⚠️ [警告]: 未提供 API Key 且环境变量中未找到，调用可能会失败。\n"

    init_log += "🚀 [NaturalCC] 正在扫描并分析项目图谱...\n"
    
    # 获取 NaturalCC 上下文
    context_prompt = "\n".join([dummy_get_naturalcc_prompt(f) for f in target_files])
    
    final_instruction = f"""你是一个由 NaturalCC 驱动的高级代码智能体。
    请按照用户的要求修改代码。

    [用户需求]
    {user_instruction}

    [NaturalCC 提供的跨文件依赖上下文]
    这是提取的关联结构，请严格参考这些信息进行代码补全/修改：
    ```c
    {context_prompt}
    """
    # 写入 Prompt 临时文件
    tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8")
    tmp_file.write(final_instruction)
    tmp_file.close()
    prompt_file_path = tmp_file.name
    init_log += f"🧠 [NaturalCC] 上下文已生成，移交控制权给 Aider (模型: {model})...\n"

    # 拼装 Aider 命令
    aider_command =[
        "aider",
        *target_files,                      
        "--model", model,                   
        "--message-file", prompt_file_path, 
        "--no-gitignore",                   
        "--map-tokens", "0",                
        "--yes-always",
        "--no-auto-commits"
    ]

    # 添加 API Key
    if api_key:
        provider = "openrouter" if "openrouter" in model else "openai"
        key_arg = f"{provider}={api_key}"
        aider_command.extend(["--api-key", key_arg])

    return aider_command, prompt_file_path, init_log

def run_aider_cli(target_files, user_instruction, model, api_key):
    """
    专供命令行直接调用的运行模式 (保留 Aider 原生交互和颜色)
    """
    aider_command, prompt_file_path, init_log = build_aider_context_and_command(
    target_files, user_instruction, model, api_key
    )
    print(init_log)
    provider = "openrouter" if "openrouter" in model else "openai"
    safe_cmd =[cmd if not cmd.startswith(f"{provider}=") else f"{provider}=sk-***" for cmd in aider_command]
    print(f"🔧 [执行命令]: {' '.join(safe_cmd)}\n" + "-"*50)

    try:
        # 直接使用 run，让 stdout 直接输出到终端，保留丰富的控制台色彩
        subprocess.run(aider_command, check=True)
        print("\n✅ [NaturalCC Agent] 任务圆满完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [Aider] 执行异常退出，退出码：{e.returncode}")
    finally:
        if os.path.exists(prompt_file_path):
            os.remove(prompt_file_path)
    
def run_aider_stream(target_files, user_instruction, model, api_key):
    """
    专供 UI 调用的流式输出模式 (Generator yield)
    """
    output_log = ""
    if not target_files:
        yield "⚠️ [错误]: 请至少选择或输入一个目标文件！\n"
        return

    aider_command, prompt_file_path, init_log = build_aider_context_and_command(
        target_files, user_instruction, model, api_key
    )

    output_log += init_log

    provider = "openrouter" if "openrouter" in model else "openai"
    safe_cmd =[cmd if not cmd.startswith(f"{provider}=") else f"{provider}=sk-***" for cmd in aider_command]
    output_log += f"🔧 [执行命令]: {' '.join(safe_cmd)}\n"
    output_log += "-" * 60 + "\n"

    # 给 UI 输出时加上 --no-pretty 关闭高级格式化，防止控制字符导致显示错乱
    aider_command.append("--no-pretty")

    yield output_log

    try:
        process = subprocess.Popen(
            aider_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )

        for line in process.stdout:
            clean_line = ANSI_ESCAPE_PATTERN.sub('', line)
            output_log += clean_line
            yield output_log 

        process.wait()

        if process.returncode == 0:
            output_log += "\n✅ [NaturalCC Agent] 任务圆满完成！\n"
        else:
            output_log += f"\n❌[Aider] 执行异常退出，退出码：{process.returncode}\n"

    except Exception as e:
        output_log += f"\n❌ [系统错误] 发生异常: {str(e)}\n"
    finally:
        if os.path.exists(prompt_file_path):
            os.remove(prompt_file_path)
        
    yield output_log

if __name__ == "__main__":
    # 配置纯命令行的入口
    parser = argparse.ArgumentParser(description="NaturalCC Agent 命令行工具")
    parser.add_argument("--files", nargs='+', required=True, help="目标文件列表，如 src/main.c src/utils.c")
    parser.add_argument("--instruction", type=str, required=True, help="你的修改需求")
    parser.add_argument("--model", type=str, default="openrouter/deepseek/deepseek-chat", help="使用的模型")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (默认读环境变量 OPENROUTER_API_KEY)")

    args = parser.parse_args()

    run_aider_cli(
        target_files=args.files,
        user_instruction=args.instruction,
        model=args.model,
        api_key=args.api_key
    )

# 如何使用 CLI 版本
# python code_agent/aider_runner.py --files src/main.c examples\instruction-finetune\guide.md --instruction "重构这里的内存分配逻辑" --api-key "sk-or-xxxx"