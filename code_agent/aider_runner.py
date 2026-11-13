import os
import re
import shutil
import subprocess
import tempfile
import argparse
import sys
import locale
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlsplit


if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from code_agent.completion_prompt_agent import CompletionPromptAgent
    from code_agent.agent_core.model_gateway import RuntimeModelConfig
else:
    from .completion_prompt_agent import CompletionPromptAgent
    from .agent_core.model_gateway import RuntimeModelConfig


ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
OUTPUT_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "gb18030",
    "cp936",
    locale.getpreferredencoding(False) or "",
]
DEFAULT_PROJECT_DIR = str(Path.cwd().resolve())

# 全局缓存一个 agent，避免同一进程中重复解析同一个项目
PROMPT_AGENT = CompletionPromptAgent()


# [maint-note:02707a81925f]
# 维护导航：normalize_project_dir；下列为本函数源码中的接口与路径索引。
# 输入 project_dir：Optional[str]。
# 返回契约：str。
# 分支条件：not raw_value。
# 返回路径：DEFAULT_PROJECT_DIR。
# 返回路径：str(Path(raw_value).expanduser().resolve())。
# 直接调用：(project_dir or '').strip。
# 直接调用：str。
# 直接调用：Path(raw_value).expanduser().resolve。
# 直接调用：Path(raw_value).expanduser。
# 直接调用：Path。
# [/maint-note:02707a81925f]
def normalize_project_dir(project_dir: Optional[str]) -> str:
    raw_value = (project_dir or "").strip()
    if not raw_value:
        return DEFAULT_PROJECT_DIR
    return str(Path(raw_value).expanduser().resolve())


# [maint-note:7107a995ac34]
# 维护导航：normalize_file_path_for_parser；下列为本函数源码中的接口与路径索引。
# 输入 file_path：Optional[str]。
# 输入 project_dir：Optional[str]。
# 返回契约：Optional[str]。
# 分支条件：not file_path。
# 分支条件：not project_dir。
# 分支条件：os.path.isabs(file_path)。
# 返回路径：file_path。
# 返回路径：os.path.relpath(file_path, project_dir_abs).replace('\\', '/')。
# 返回路径：file_path.replace('\\', '/')。
# 捕获异常：ValueError。
# 直接调用：os.path.isabs。
# 直接调用：os.path.abspath。
# 直接调用：normalize_project_dir。
# 直接调用：os.path.relpath(file_path, project_dir_abs).replace。
# 直接调用：os.path.relpath。
# 直接调用：file_path.replace。
# 表达式分流：os.path.isabs(file_path)；真值与假值分支分别计算，不同时执行。
# [/maint-note:7107a995ac34]
def normalize_file_path_for_parser(file_path: Optional[str], project_dir: Optional[str]) -> Optional[str]:
    if not file_path:
        return file_path
    if not project_dir:
        return file_path

    file_path = os.path.abspath(file_path) if os.path.isabs(file_path) else file_path
    project_dir_abs = normalize_project_dir(project_dir)

    if os.path.isabs(file_path):
        try:
            return os.path.relpath(file_path, project_dir_abs).replace("\\", "/")
        except ValueError:
            return file_path
    return file_path.replace("\\", "/")


# [maint-note:9c2d8d45ef24]
# 维护导航：detect_provider；下列为本函数源码中的接口与路径索引。
# 输入 model：str。
# 返回契约：str。
# 分支条件：model.startswith('ollama_chat/')。
# 分支条件：model.startswith('deepseek/') or 'deepseek' in model。
# 分支条件：model.startswith('openrouter/') or 'openrouter' in model。
# 分支条件：model.startswith('openai/') or '/openai/' in model。
# 分支条件：model.startswith('deepseek')。
# 返回路径：'ollama'。
# 返回路径：'deepseek'。
# 返回路径：'openrouter'。
# 返回路径：'openai'。
# 直接调用：model.startswith。
# [/maint-note:9c2d8d45ef24]
def detect_provider(model: str) -> str:
    """
    根据模型名前缀判断 LiteLLM provider。
    --api-key 参数形如:
      --api-key deepseek=sk-xxx
      --api-key openai=sk-xxx
      --api-key openrouter=sk-xxx
    """
    model = model or ""
    if model.startswith("ollama_chat/"):
        return "ollama"
    if model.startswith("deepseek/") or "deepseek" in model:
        return "deepseek"
    if model.startswith("openrouter/") or "openrouter" in model:
        return "openrouter"
    if model.startswith("openai/") or "/openai/" in model:
        return "openai"
    # 默认按 deepseek 处理（本项目默认使用 deepseek）
    if model.startswith("deepseek"):
        return "deepseek"
    return "deepseek"


# [maint-note:a0809d657c9d]
# 维护导航：ensure_aider_installed；下列为本函数源码中的接口与路径索引。
# 返回契约：None。
# 分支条件：shutil.which('aider') is None。
# 显式抛出：FileNotFoundError('未检测到 aider 命令。请先安装 aider，并确保其已加入 PATH。')。
# 直接调用：shutil.which。
# 直接调用：FileNotFoundError。
# [/maint-note:a0809d657c9d]
def ensure_aider_installed() -> None:
    if shutil.which("aider") is None:
        raise FileNotFoundError(
            "未检测到 aider 命令。请先安装 aider，并确保其已加入 PATH。"
        )


# [maint-note:418f710e233d]
# 维护导航：normalize_target_files；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 project_dir：类型由调用方与函数体约束；缺省表达式 None。
# 分支条件：target_files is None。
# 分支条件：isinstance(target_files, str)。
# 分支条件：not item。
# 分支条件：os.path.isabs(item)。
# 遍历边界：item ← target_files。
# 返回路径：[]。
# 返回路径：result。
# 直接调用：isinstance。
# 直接调用：str(item).strip。
# 直接调用：str。
# 直接调用：os.path.isabs。
# 直接调用：result.append。
# 直接调用：normalize_project_dir。
# 直接调用：os.path.abspath。
# 直接调用：os.path.join。
# [/maint-note:418f710e233d]
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
            base_dir = normalize_project_dir(project_dir)
            result.append(os.path.abspath(os.path.join(base_dir, item)))

    return result


# [maint-note:c075018b3ce9]
# 维护导航：resolve_api_key；下列为本函数源码中的接口与路径索引。
# 输入 api_key：Optional[str]。
# 返回契约：Tuple[Optional[str], str]。
# 分支条件：api_key and api_key != 'Add by yourself'。
# 返回路径：(api_key, '')。
# 返回路径：('', '⚠️ [警告]: 未提供 API Key，且环境变量中未找到 DEEPSEEK_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY，调用可能会失败。\n')。
# 直接调用：os.environ.get。
# [/maint-note:c075018b3ce9]
def resolve_api_key(api_key: Optional[str]) -> Tuple[Optional[str], str]:
    """解析 API Key：优先参数传入 → 环境变量 DEEPSEEK_API_KEY → OPENROUTER_API_KEY → OPENAI_API_KEY"""
    api_key = api_key or (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if api_key and api_key != "Add by yourself":
        return api_key, ""
    return (
        "",
        "⚠️ [警告]: 未提供 API Key，且环境变量中未找到 DEEPSEEK_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY，调用可能会失败。\n",
    )


# [maint-note:136c080421e8]
# 维护导航：write_prompt_file；下列为本函数源码中的接口与路径索引。
# 输入 final_instruction：str。
# 返回契约：str。
# 返回路径：tmp_file.name。
# 直接调用：tempfile.NamedTemporaryFile。
# 直接调用：tmp_file.write。
# 直接调用：tmp_file.close。
# [/maint-note:136c080421e8]
def write_prompt_file(final_instruction: str) -> str:
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".txt",
        encoding="utf-8",
    )
    tmp_file.write(final_instruction)
    tmp_file.close()
    return tmp_file.name


# [maint-note:5542271f8acd]
# 维护导航：append_api_key_arg；下列为本函数源码中的接口与路径索引。
# 输入 aider_command：List[str]。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 返回契约：None。
# 分支条件：api_key and (not model.startswith('ollama_chat/'))。
# 直接调用：model.startswith。
# 直接调用：detect_provider。
# 直接调用：aider_command.extend。
# [/maint-note:5542271f8acd]
def append_api_key_arg(aider_command: List[str], model: str, api_key: Optional[str]) -> None:
    if api_key and not model.startswith("ollama_chat/"):
        provider = detect_provider(model)
        aider_command.extend(["--api-key", f"{provider}={api_key}"])


# [maint-note:bf0e45cc7b2f]
# 维护导航：build_aider_command；下列为本函数源码中的接口与路径索引。
# 输入 edit_files：List[str]。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 输入 prompt_file_path：str。
# 输入 no_pretty：bool；缺省表达式 False。
# 输入 dry_run：bool；缺省表达式 False。
# 返回契约：List[str]。
# 分支条件：dry_run。
# 分支条件：no_pretty。
# 分支条件：model.startswith('ollama_chat/')。
# 返回路径：aider_command。
# 直接调用：aider_command.append。
# 直接调用：model.startswith。
# 直接调用：aider_command.extend。
# 直接调用：append_api_key_arg。
# [/maint-note:bf0e45cc7b2f]
def build_aider_command(
    edit_files: List[str],
    model: str,
    api_key: Optional[str],
    prompt_file_path: str,
    no_pretty: bool = False,
    dry_run: bool = False,
) -> List[str]:
    aider_command = [
        "aider",
        *edit_files,
        "--model",
        model,
        "--message-file",
        prompt_file_path,
        "--no-gitignore",
        "--map-tokens",
        "0",
        "--yes-always",
        "--no-fancy-input",
        "--no-auto-commits",
    ]
    if dry_run:
        aider_command.append("--dry-run")
    if no_pretty:
        aider_command.append("--no-pretty")
    if model.startswith("ollama_chat/"):
        aider_command.extend(["--no-check-update", "--no-analytics", "--no-show-release-notes"])
    append_api_key_arg(aider_command, model, api_key)
    return aider_command


# [maint-note:1be1e11df4eb]
# 维护导航：generate_completion_prompt；下列为本函数源码中的接口与路径索引。
# 输入 project_dir：str。
# 输入 user_instruction：str。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 返回契约：str。
# 返回路径：PROMPT_AGENT.build_prompt(request)。
# 直接调用：normalize_project_dir。
# 直接调用：normalize_target_files。
# 直接调用：normalize_file_path_for_parser。
# 直接调用：PROMPT_AGENT.build_prompt。
# 表达式分流：normalized_targets；真值与假值分支分别计算，不同时执行。
# [/maint-note:1be1e11df4eb]
def generate_completion_prompt(
    project_dir: str,
    user_instruction: str,
    target_files,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
) -> str:
    project_dir = normalize_project_dir(project_dir)
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


# [maint-note:b38d3f3a547b]
# 维护导航：build_aider_context_and_command；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 输入 project_dir：Optional[str]；缺省表达式 None。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 输入 no_pretty：bool；缺省表达式 False。
# 输入 dry_run：bool；缺省表达式 False。
# 返回契约：Tuple[List[str], str, str, str]。
# 分支条件：not user_instruction or not user_instruction.strip()。
# 分支条件：not os.path.isdir(project_dir)。
# 分支条件：dry_run。
# 返回路径：(aider_command, prompt_file_path, init_log, final_instruction)。
# 显式抛出：ValueError('user_instruction 不能为空。')。
# 显式抛出：ValueError(f'项目根目录不存在或不可访问: {project_dir}')。
# 直接调用：ensure_aider_installed。
# 直接调用：normalize_project_dir。
# 直接调用：normalize_target_files。
# 直接调用：user_instruction.strip。
# 直接调用：ValueError。
# 直接调用：model.startswith。
# 直接调用：resolve_api_key。
# 直接调用：os.path.isdir。
# 直接调用：normalize_file_path_for_parser。
# 直接调用：generate_completion_prompt。
# 直接调用：write_prompt_file。
# 直接调用：build_aider_command。
# 表达式分流：model.startswith('ollama_chat/')；真值与假值分支分别计算，不同时执行。
# 表达式分流：target_files；真值与假值分支分别计算，不同时执行。
# [/maint-note:b38d3f3a547b]
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
    dry_run: bool = False,
) -> Tuple[List[str], str, str, str]:
    """
    返回:
      aider_command, prompt_file_path, init_log, final_instruction
    """
    ensure_aider_installed()

    project_dir = normalize_project_dir(project_dir)

    target_files = normalize_target_files(target_files, project_dir=project_dir)

    init_log = ""

    if not user_instruction or not user_instruction.strip():
        raise ValueError("user_instruction 不能为空。")

    api_key, api_key_log = (None, "") if model.startswith("ollama_chat/") else resolve_api_key(api_key)
    init_log += api_key_log

    init_log += "🚀 [NaturalCC] 正在扫描并分析项目图谱...\n"

    if not os.path.isdir(project_dir):
        raise ValueError(f"项目根目录不存在或不可访问: {project_dir}")

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
    init_log += f"🧠 [NaturalCC] Prompt 已生成，移交控制权给 Aider (模型: {model})...\n"
    if dry_run:
        init_log += "ℹ️ [Aider] dry-run enabled; files will not be modified.\n"

    prompt_file_path = write_prompt_file(final_instruction)
    aider_command = build_aider_command(
        edit_files=target_files,
        model=model,
        api_key=api_key,
        prompt_file_path=prompt_file_path,
        no_pretty=no_pretty,
        dry_run=dry_run,
    )

    return aider_command, prompt_file_path, init_log, final_instruction


# [maint-note:e17e759b9f89]
# 维护导航：mask_command_for_log；下列为本函数源码中的接口与路径索引。
# 输入 aider_command：List[str]。
# 输入 model：str。
# 返回契约：str。
# 分支条件：isinstance(cmd, str) and cmd.startswith(f'{provider}=')。
# 遍历边界：cmd ← aider_command。
# 返回路径：' '.join(safe_cmd)。
# 直接调用：detect_provider。
# 直接调用：isinstance。
# 直接调用：cmd.startswith。
# 直接调用：safe_cmd.append。
# 直接调用：' '.join。
# [/maint-note:e17e759b9f89]
def mask_command_for_log(aider_command: List[str], model: str) -> str:
    provider = detect_provider(model)
    safe_cmd = []
    for cmd in aider_command:
        if isinstance(cmd, str) and cmd.startswith(f"{provider}="):
            safe_cmd.append(f"{provider}=sk-***")
        else:
            safe_cmd.append(cmd)
    return " ".join(safe_cmd)


# [maint-note:5bb968edfbf0]
# 维护导航：build_subprocess_env；下列为本函数源码中的接口与路径索引。
# 输入 ollama_base_url：Optional[str]；缺省表达式 None。
# 返回契约：dict。
# 分支条件：ollama_base_url is not None。
# 分支条件：name.upper().endswith('_API_KEY') or name.upper() in {'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'}。
# 遍历边界：name ← list(env)。
# 返回路径：env。
# 写入目标：env['OLLAMA_API_BASE']。
# 写入目标：env['NO_PROXY']。
# 写入目标：env['PYTHONIOENCODING']。
# 写入目标：env['PYTHONUTF8']。
# 直接调用：os.environ.copy。
# 直接调用：RuntimeModelConfig.from_dict。
# 直接调用：list。
# 直接调用：name.upper().endswith。
# 直接调用：name.upper。
# 直接调用：urlsplit。
# 直接调用：env.setdefault。
# [/maint-note:5bb968edfbf0]
def build_subprocess_env(ollama_base_url: Optional[str] = None) -> dict:
    env = os.environ.copy()
    if ollama_base_url is not None:
        RuntimeModelConfig.from_dict({"provider": "ollama", "base_url": ollama_base_url})
        for name in list(env):
            if name.upper().endswith("_API_KEY") or name.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
                del env[name]
        parsed = urlsplit(ollama_base_url)
        env["OLLAMA_API_BASE"] = f"{parsed.scheme}://{parsed.netloc}"
        env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    # 在 Windows pipe 场景下强制 Python 子进程优先走 UTF-8，降低中文输出乱码概率。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("NO_COLOR", "1")
    return env


# [maint-note:f95fd2cfa2b8]
# 维护导航：local_ollama_base_url；下列为本函数源码中的接口与路径索引。
# 输入 base_url：Optional[str]；缺省表达式 None。
# 返回契约：str。
# 返回路径：versioned。
# 直接调用：(base_url or os.environ.get('OLLAMA_API_BASE') or 'http://127.0.0.1:11434').rstrip。
# 直接调用：os.environ.get。
# 直接调用：configured.endswith。
# 直接调用：RuntimeModelConfig.from_dict。
# 表达式分流：configured.endswith('/v1')；真值与假值分支分别计算，不同时执行。
# [/maint-note:f95fd2cfa2b8]
def local_ollama_base_url(base_url: Optional[str] = None) -> str:
    configured = (base_url or os.environ.get("OLLAMA_API_BASE") or "http://127.0.0.1:11434").rstrip("/")
    versioned = configured if configured.endswith("/v1") else configured + "/v1"
    RuntimeModelConfig.from_dict({"provider": "ollama", "base_url": versioned})
    return versioned


# [maint-note:5ca100f042b6]
# 维护导航：decode_process_line；下列为本函数源码中的接口与路径索引。
# 输入 raw_line：bytes。
# 返回契约：str。
# 分支条件：not encoding。
# 遍历边界：encoding ← OUTPUT_ENCODING_CANDIDATES。
# 返回路径：raw_line.decode(encoding).replace('\r\n', '\n').replace('\r', '\n')。
# 返回路径：raw_line.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')。
# 捕获异常：UnicodeDecodeError。
# 直接调用：raw_line.decode(encoding).replace('\r\n', '\n').replace。
# 直接调用：raw_line.decode(encoding).replace。
# 直接调用：raw_line.decode。
# 直接调用：raw_line.decode('utf-8', errors='replace').replace('\r\n', '\n').replace。
# 直接调用：raw_line.decode('utf-8', errors='replace').replace。
# [/maint-note:5ca100f042b6]
def decode_process_line(raw_line: bytes) -> str:
    for encoding in OUTPUT_ENCODING_CANDIDATES:
        if not encoding:
            continue
        try:
            return raw_line.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    return raw_line.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


# [maint-note:3f36a7af53d4]
# 维护导航：run_aider_cli；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 输入 project_dir：Optional[str]；缺省表达式 None。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 分支条件：os.path.exists(prompt_file_path)。
# 捕获异常：subprocess.CalledProcessError。
# 直接调用：model.startswith。
# 直接调用：local_ollama_base_url。
# 直接调用：build_aider_context_and_command。
# 直接调用：print。
# 直接调用：mask_command_for_log。
# 直接调用：subprocess.run。
# 直接调用：build_subprocess_env。
# 直接调用：os.path.exists。
# 直接调用：os.remove。
# 表达式分流：model.startswith('ollama_chat/')；真值与假值分支分别计算，不同时执行。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# [/maint-note:3f36a7af53d4]
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
    ollama_base_url = local_ollama_base_url() if model.startswith("ollama_chat/") else None
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
        subprocess.run(aider_command, check=True, env=build_subprocess_env(ollama_base_url))
        print("\n✅ [NaturalCC Agent] 任务圆满完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [Aider] 执行异常退出，退出码：{e.returncode}")
    finally:
        if os.path.exists(prompt_file_path):
            os.remove(prompt_file_path)


# [maint-note:3841b3b21dfc]
# 维护导航：run_aider_stream；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 输入 project_dir：Optional[str]；缺省表达式 None。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 输入 dry_run：bool；缺省表达式 False。
# 输入 base_url：Optional[str]；缺省表达式 None。
# 返回契约：Generator[str, None, None]。
# 分支条件：not os.path.isdir(project_dir)。
# 分支条件：not user_instruction or not user_instruction.strip()。
# 返回路径：无返回值。
# 捕获异常：Exception。
# 直接调用：model.startswith。
# 直接调用：local_ollama_base_url。
# 直接调用：normalize_project_dir。
# 直接调用：normalize_target_files。
# 直接调用：os.path.isdir。
# 直接调用：user_instruction.strip。
# 直接调用：build_aider_context_and_command。
# 直接调用：str。
# 直接调用：stream_aider_command。
# 表达式分流：model.startswith('ollama_chat/')；真值与假值分支分别计算，不同时执行。
# 生成器交付：f'⚠️ [错误]: 项目根目录不存在或不可访问: {project_dir}\n'。
# 生成器交付：'⚠️ [错误]: 请输入修改/补全需求。\n'。
# 生成器交付：f'\n❌ [系统错误] 发生异常: {str(e)}\n'。
# 生成器交付：stream_aider_command(aider_command, prompt_file_path, init_log, model, ollama_base_url)。
# [/maint-note:3841b3b21dfc]
def run_aider_stream(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
    dry_run: bool = False,
    base_url: Optional[str] = None,
) -> Generator[str, None, None]:
    try:
        ollama_base_url = local_ollama_base_url(base_url) if model.startswith("ollama_chat/") else None
        project_dir = normalize_project_dir(project_dir)
        target_files = normalize_target_files(target_files, project_dir=project_dir)

        if not os.path.isdir(project_dir):
            yield f"⚠️ [错误]: 项目根目录不存在或不可访问: {project_dir}\n"
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
            dry_run=dry_run,
        )

    except Exception as e:
        yield f"\n❌ [系统错误] 发生异常: {str(e)}\n"
        return

    yield from stream_aider_command(aider_command, prompt_file_path, init_log, model, ollama_base_url)


# [maint-note:ee3462cc4f2f]
# 维护导航：stream_aider_command；下列为本函数源码中的接口与路径索引。
# 输入 aider_command：List[str]。
# 输入 prompt_file_path：str。
# 输入 init_log：str。
# 输入 model：str。
# 输入 ollama_base_url：Optional[str]；缺省表达式 None。
# 返回契约：Generator[str, None, None]。
# 分支条件：process.returncode == 0。
# 分支条件：os.path.exists(prompt_file_path)。
# 遍历边界：line ← process.stdout。
# 捕获异常：Exception。
# 直接调用：mask_command_for_log。
# 直接调用：subprocess.Popen。
# 直接调用：build_subprocess_env。
# 直接调用：decode_process_line。
# 直接调用：ANSI_ESCAPE_PATTERN.sub。
# 直接调用：process.wait。
# 直接调用：str。
# 直接调用：os.path.exists。
# 直接调用：os.remove。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# 生成器交付：output_log。
# [/maint-note:ee3462cc4f2f]
def stream_aider_command(
    aider_command: List[str],
    prompt_file_path: str,
    init_log: str,
    model: str,
    ollama_base_url: Optional[str] = None,
) -> Generator[str, None, None]:
    output_log = ""
    try:
        output_log += init_log
        output_log += f"🔧 [执行命令]: {mask_command_for_log(aider_command, model)}\n"
        output_log += "-" * 60 + "\n"

        yield output_log

        process = subprocess.Popen(
            aider_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=build_subprocess_env(ollama_base_url),
        )

        assert process.stdout is not None
        for line in process.stdout:
            decoded_line = decode_process_line(line)
            clean_line = ANSI_ESCAPE_PATTERN.sub("", decoded_line)
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
            if os.path.exists(prompt_file_path):
                os.remove(prompt_file_path)
        except Exception:
            pass

    yield output_log


# [maint-note:bc2f9731e31a]
# 维护导航：preview_prompt；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 输入 project_dir：Optional[str]；缺省表达式 None。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 输入 dry_run：bool；缺省表达式 False。
# 返回契约：str。
# 分支条件：os.path.exists(prompt_file_path)。
# 返回路径：preview_text。
# 返回路径：f'❌ Prompt 生成失败: {e}'。
# 捕获异常：Exception。
# 直接调用：normalize_project_dir。
# 直接调用：normalize_target_files。
# 直接调用：build_aider_context_and_command。
# 直接调用：os.path.exists。
# 直接调用：os.remove。
# [/maint-note:bc2f9731e31a]
def preview_prompt(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
    dry_run: bool = False,
) -> str:
    """
    仅生成并预览最终给 Aider 的 prompt，不执行 Aider。
    """
    try:
        project_dir = normalize_project_dir(project_dir)
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
            dry_run=dry_run,
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
    parser.add_argument("-f", "--files", nargs="*", default=[], help="目标文件列表，如 src/main.c src/utils.c")
    parser.add_argument("-i", "--instruction", type=str, required=True, help="你的修改需求")
    parser.add_argument("-m", "--model", type=str, default="deepseek/deepseek-chat", help="使用的模型")
    parser.add_argument("-key", "--api-key", type=str, default=None, help="API Key (默认读环境变量)")
    parser.add_argument(
        "-dir", "--project-dir",
        type=str,
        default=DEFAULT_PROJECT_DIR,
        help="项目根目录，默认使用当前运行程序的目录",
    )
    parser.add_argument("-s", "--symbol", type=str, default=None, help="目标符号（可选）")
    parser.add_argument(
        "-t", "--completion-type",
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
