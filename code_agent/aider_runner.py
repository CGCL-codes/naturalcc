import os
import re
import shutil
import subprocess
import tempfile
import argparse
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from code_agent.completion_prompt_agent import CompletionPromptAgent
else:
    from .completion_prompt_agent import CompletionPromptAgent


ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# 全局缓存一个 agent，避免同一进程中重复解析同一个项目
PROMPT_AGENT = CompletionPromptAgent()


def normalize_file_path_for_parser(file_path: Optional[str], project_dir: Optional[str]) -> Optional[str]:
    if not file_path:
        return file_path
    if not project_dir:
        return file_path

    file_path = os.path.abspath(file_path) if os.path.isabs(file_path) else file_path
    project_dir_abs = os.path.abspath(project_dir)

    if os.path.isabs(file_path):
        try:
            return os.path.relpath(file_path, project_dir_abs).replace("\\", "/")
        except ValueError:
            return file_path
    return file_path.replace("\\", "/")


def detect_provider(model: str) -> str:
    """
    根据模型名前缀粗略判断 provider。
    Aider 的 --api-key 参数通常形如:
      --api-key openrouter=sk-xxx
      --api-key openai=sk-xxx
    """
    model = model or ""
    if model.startswith("openrouter/") or "openrouter" in model:
        return "openrouter"
    if model.startswith("openai/") or "/openai/" in model:
        return "openai"
    return "openai"


def ensure_aider_installed() -> None:
    if shutil.which("aider") is None:
        raise FileNotFoundError(
            "未检测到 aider 命令。请先安装 aider，并确保其已加入 PATH。"
        )


def normalize_target_files(target_files, project_dir=None):
    if target_files is None:
        return []

    if isinstance(target_files, str):
        target_files = [target_files]

    result = []
    for item in target_files:
        if not item:
            continue
        item = str(item).strip()
        if not item:
            continue

        if os.path.isabs(item):
            result.append(item)
        else:
            base_dir = project_dir or os.getcwd()
            result.append(os.path.abspath(os.path.join(base_dir, item)))

    return result


def generate_completion_prompt(
    project_dir: str,
    user_instruction: str,
    target_files,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
) -> str:
    normalized_targets = normalize_target_files(target_files, project_dir=project_dir)
    effective_file_path = normalized_targets[0] if normalized_targets else None
    normalized_file_path = normalize_file_path_for_parser(effective_file_path, project_dir)

    request = {
        "project_dir": project_dir,
        "user_instruction": user_instruction,
        "file_path": normalized_file_path,
        "symbol": symbol,
        "completion_type": completion_type or None,
        "prefix": prefix or "",
    }
    return PROMPT_AGENT.build_prompt(request)


def build_aider_context_and_command(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
    no_pretty: bool = False,
) -> Tuple[List[str], str, str, str]:
    """
    返回:
      aider_command, prompt_file_path, init_log, final_instruction
    """
    ensure_aider_installed()

    target_files = normalize_target_files(target_files, project_dir=project_dir)

    init_log = ""

    if not user_instruction or not user_instruction.strip():
        raise ValueError("user_instruction 不能为空。")

    DEFAULT_API_KEY = "Add by yourself"

    if not api_key:
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or DEFAULT_API_KEY
        )
        if not api_key:
            init_log += "⚠️ [警告]: 未提供 API Key，且环境变量中未找到 OPENROUTER_API_KEY / OPENAI_API_KEY，调用可能会失败。\n"

    init_log += "🚀 [NaturalCC] 正在扫描并分析项目图谱...\n"

    if not project_dir:
        raise ValueError("必须提供 project_dir，用于加载项目解析信息。")

    effective_file_path = target_files[0] if target_files else None
    normalized_effective_file_path = normalize_file_path_for_parser(effective_file_path, project_dir)

    prompt_from_agent = generate_completion_prompt(
        project_dir=project_dir,
        user_instruction=user_instruction,
        target_files=target_files,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
    )

    final_instruction = f"""你是一个由 NaturalCC 驱动的高级代码智能体。
请严格根据下面的项目语义上下文执行代码补全、修改或重构任务。

【说明】
1. 优先参考项目内真实存在的符号、类型、函数和成员。
2. 若需要修改代码，请直接在目标文件中完成修改。
3. 若上下文中存在候选项，优先使用项目内已有实现风格。
4. 除非必须，不要凭空捏造项目中不存在的 API。
5. 本次解析目标文件为：{normalized_effective_file_path}

{prompt_from_agent}
"""

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".txt",
        encoding="utf-8",
    )
    tmp_file.write(final_instruction)
    tmp_file.close()
    prompt_file_path = tmp_file.name

    init_log += f"🧠 [NaturalCC] Prompt 已生成，移交控制权给 Aider (模型: {model})...\n"

    aider_command = [
        "aider",
        *target_files,
        "--model",
        model,
        "--message-file",
        prompt_file_path,
        "--no-gitignore",
        "--map-tokens",
        "0",
        "--yes-always",
        "--no-auto-commits",
    ]

    if no_pretty:
        aider_command.append("--no-pretty")

    if api_key:
        provider = detect_provider(model)
        key_arg = f"{provider}={api_key}"
        aider_command.extend(["--api-key", key_arg])

    return aider_command, prompt_file_path, init_log, final_instruction


def mask_command_for_log(aider_command: List[str], model: str) -> str:
    provider = detect_provider(model)
    safe_cmd = []
    for cmd in aider_command:
        if isinstance(cmd, str) and cmd.startswith(f"{provider}="):
            safe_cmd.append(f"{provider}=sk-***")
        else:
            safe_cmd.append(cmd)
    return " ".join(safe_cmd)


def run_aider_cli(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
):
    aider_command, prompt_file_path, init_log, _final_instruction = build_aider_context_and_command(
        target_files=target_files,
        user_instruction=user_instruction,
        model=model,
        api_key=api_key,
        project_dir=project_dir,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
        no_pretty=False,
    )

    print(init_log)
    print(f"🔧 [执行命令]: {mask_command_for_log(aider_command, model)}\n" + "-" * 60)

    try:
        subprocess.run(aider_command, check=True)
        print("\n✅ [NaturalCC Agent] 任务圆满完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [Aider] 执行异常退出，退出码：{e.returncode}")
    finally:
        if os.path.exists(prompt_file_path):
            os.remove(prompt_file_path)


def run_aider_stream(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
) -> Generator[str, None, None]:
    output_log = ""

    try:
        target_files = normalize_target_files(target_files, project_dir=project_dir)

        if not project_dir:
            yield "⚠️ [错误]: 请输入项目根目录 project_dir。\n"
            return

        if not user_instruction or not user_instruction.strip():
            yield "⚠️ [错误]: 请输入修改/补全需求。\n"
            return

        aider_command, prompt_file_path, init_log, _final_instruction = build_aider_context_and_command(
            target_files=target_files,
            user_instruction=user_instruction,
            model=model,
            api_key=api_key,
            project_dir=project_dir,
            symbol=symbol,
            completion_type=completion_type,
            prefix=prefix,
            no_pretty=True,
        )

        output_log += init_log
        output_log += f"🔧 [执行命令]: {mask_command_for_log(aider_command, model)}\n"
        output_log += "-" * 60 + "\n"

        yield output_log

        process = subprocess.Popen(
            aider_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            clean_line = ANSI_ESCAPE_PATTERN.sub("", line)
            output_log += clean_line
            yield output_log

        process.wait()

        if process.returncode == 0:
            output_log += "\n✅ [NaturalCC Agent] 任务圆满完成！\n"
        else:
            output_log += f"\n❌ [Aider] 执行异常退出，退出码：{process.returncode}\n"

    except Exception as e:
        output_log += f"\n❌ [系统错误] 发生异常: {str(e)}\n"
    finally:
        try:
            if "prompt_file_path" in locals() and os.path.exists(prompt_file_path):
                os.remove(prompt_file_path)
        except Exception:
            pass

    yield output_log


def preview_prompt(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
) -> str:
    """
    仅生成并预览最终给 Aider 的 prompt，不执行 Aider。
    """
    try:
        target_files = normalize_target_files(target_files, project_dir=project_dir)
        _aider_command, prompt_file_path, init_log, final_instruction = build_aider_context_and_command(
            target_files=target_files,
            user_instruction=user_instruction,
            model=model,
            api_key=api_key,
            project_dir=project_dir,
            symbol=symbol,
            completion_type=completion_type,
            prefix=prefix,
            no_pretty=True,
        )

        preview_text = init_log + "\n" + "=" * 80 + "\n"
        preview_text += "【最终 Prompt 预览】\n"
        preview_text += "=" * 80 + "\n"
        preview_text += final_instruction

        if os.path.exists(prompt_file_path):
            os.remove(prompt_file_path)

        return preview_text
    except Exception as e:
        return f"❌ Prompt 生成失败: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NaturalCC Agent 命令行工具")
    parser.add_argument("--files", nargs="*", default=[], help="目标文件列表，如 src/main.c src/utils.c")
    parser.add_argument("--instruction", type=str, required=True, help="你的修改需求")
    parser.add_argument("--model", type=str, default="openrouter/deepseek/deepseek-chat", help="使用的模型")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (默认读环境变量)")
    parser.add_argument("--project-dir", type=str, required=True, help="项目根目录")
    parser.add_argument("--symbol", type=str, default=None, help="目标符号（可选）")
    parser.add_argument(
        "--completion-type",
        type=str,
        default=None,
        choices=[None, "member", "variable", "function", "function_body", "type"],
        help="补全类型（可选）",
    )
    parser.add_argument("--prefix", type=str, default="", help="补全前缀（可选）")
    parser.add_argument("--preview", action="store_true", help="仅预览最终 Prompt，不执行 Aider")

    args = parser.parse_args()

    if args.preview:
        print(
            preview_prompt(
                target_files=args.files,
                user_instruction=args.instruction,
                model=args.model,
                api_key=args.api_key,
                project_dir=args.project_dir,
                symbol=args.symbol,
                completion_type=args.completion_type,
                prefix=args.prefix,
            )
        )
    else:
        run_aider_cli(
            target_files=args.files,
            user_instruction=args.instruction,
            model=args.model,
            api_key=args.api_key,
            project_dir=args.project_dir,
            symbol=args.symbol,
            completion_type=args.completion_type,
            prefix=args.prefix,
        )
