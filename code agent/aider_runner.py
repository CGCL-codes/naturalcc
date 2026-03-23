import os
import subprocess
import tempfile

# 假设这是你现有的 NaturalCC API
# from naturalcc.c_graph.generator import get_naturalcc_prompt 

def dummy_get_naturalcc_prompt(filepath: str) -> str:
    """
    Naturalcc API
    """
    return """
    /* NaturalCC Context Map */
    /* Include Dependencies Found: */
    // db_connect.h: void connect_to_db(const char* host);
    // utils.c: int retry_count = 3;
    /* End Context */
    """

def run_naturalcc_with_aider(
    target_file: str, 
    user_instruction: str, 
    model: str = "openrouter/deepseek/deepseek-chat", # 默认使用你指定的 OpenRouter 模型
    api_provider: str = "openrouter",                 # 对应的 Provider
    api_key: str = None                               # 接收传入的 API Key
):
    """
    结合 NaturalCC 与 Aider 的核心主控函数
    """
    
    # 环境变量兜底逻辑
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print(f"⚠️ 警告: 未提供 api_key 且环境变量中未找到，调用 {model} 可能会失败。")
    print("haha" * 3, api_key)

    print(f"🚀 [NaturalCC] 正在分析 {target_file} 的项目图谱...")
    
    # 1. 调用你们的底层功能，获取图谱和上下文
    context_prompt = dummy_get_naturalcc_prompt(target_file)
    
    # 2. 组装终极 Prompt
    final_instruction = f"""你是一个由 NaturalCC 驱动的高级代码智能体。
    请按照用户的要求修改代码。

    【用户需求】
    {user_instruction}

    【NaturalCC 提供的跨文件依赖上下文】
    这是通过底层 Clang AST 提取的关联结构，请严格参考这些信息进行代码补全/修改：
    ```c
    {context_prompt}
    """
    # 3. 将组装好的 Prompt 写入临时文件
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tmp_file:
        tmp_file.write(final_instruction)
        prompt_file_path = tmp_file.name

    print(f"🧠 [NaturalCC] 上下文已生成，移交控制权给 Aider (模型: {model})...")

    # 4. 构建 Aider 启动命令
    # 完全仿照你的格式：aider <file> --model <model> --message-file <msg> --no-gitignore ...
    # 4. 构建 Aider 启动命令
    aider_command =[
        "aider",
        target_file,                        # 目标文件
        "--model", model,                   # 模型名称
        "--message-file", prompt_file_path, # NaturalCC 生成的指令文件
        "--no-gitignore",                   # 忽略 gitignore
        "--map-tokens", "0",                # 【修改这里】：将 Token 分配给 Repo Map 的数量设为 0，即完全关闭自带图谱
        "--yes-always",                     # 【修改这里】：自动全盘同意修改（不同版本可能叫 --yes 或 --yes-always）
    ]

    # 动态将 `--api-key openrouter=skXXX` 加入启动命令
    if api_key:
        key_arg = f"{api_provider}={api_key}"
        aider_command.extend(["--api-key", key_arg])

    # 5. 执行 Aider
    try:
        # 打印一下实际执行的命令，方便你 debug 确认格式是否正确（过滤掉真实的 key）
        debug_cmd =[cmd if not cmd.startswith(f"{api_provider}=") else f"{api_provider}=sk-***" for cmd in aider_command]
        print(f"🔧 [执行命令]: {' '.join(debug_cmd)}")
        
        # subprocess.run 会直接把 aider 的输出流重定向到当前终端
        subprocess.run(aider_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ [Aider] 执行失败，退出码：{e.returncode}")
    finally:
        # 6. 清理临时生成的指令文件
        if os.path.exists(prompt_file_path):
            os.remove(prompt_file_path)
        
    print("✅[NaturalCC Agent] 任务完成！")


# 测试
if __name__ == "__main__":
    run_naturalcc_with_aider(
        target_file="test.c", 
        user_instruction="优化这里的内存分配逻辑，注意参考上文的依赖结构。",
        model="openrouter/deepseek/deepseek-chat",
        api_provider="openrouter",
        api_key=None
    )