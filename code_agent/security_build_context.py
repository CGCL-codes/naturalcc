"""Project compile context (include paths and macros) for Clang and Cppcheck.

Without it the analysers parse each file with only the project root on the
include path, so a real project fails before analysis starts: RT-Thread's dfs.c
cannot find ``dfs.h`` (components/dfs/dfs_v2/include) or ``rtthread.h``
(include/), Clang reports every file as failed and Cppcheck reads unexpanded
macros such as ``MSH_CMD_EXPORT(cmd, free text)`` as syntax errors.

Sources, in order of preference:

1. ``compile_commands.json`` -- the path given as ``compile_commands`` in the
   feature config, or the file at the project root. Per-file flags, so
   directory-local defines (RT-Thread's ``__RT_KERNEL_SOURCE__``) are honoured.
2. ``compile_flags.txt`` at the project root -- one flag per line, for every file.
3. Inferred include directories: for each ``#include`` the project can satisfy,
   the directory that resolves it. Headers the host C library already provides
   are added with ``-idirafter`` so the system copy stays authoritative.

``include_dirs`` and ``defines`` from the feature config are appended to
whichever source applies.

Untrusted input: a compile database is user supplied, and flags such as
``-Xclang -load`` or ``-fplugin=`` would make the analyser load arbitrary code.
Only include, macro and language-standard flags are passed on; every path must
resolve inside the project directory. Flags from a database that break those
rules are dropped and reported; the same mistakes in the feature config are
rejected outright.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

PATH_FLAGS = ("-I", "-iquote", "-isystem", "-idirafter")
FILE_FLAGS = ("-include", "-imacros")
MACRO_FLAGS = ("-D", "-U")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
_MACRO_NAME = re.compile(r"^[A-Za-z_]\w*(?:\([A-Za-z_\w\s,.]*\))?$")
_STANDARD = re.compile(r"^-std=(?:c|gnu|iso9899:)[\w+]+$|^-std=(?:c|gnu)\+\+\w+$")
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"]', re.M)
MAX_DATABASE_BYTES = 50_000_000
MAX_INFERRED_HEADERS = 3000


# [maint-note:b504d02f6d37]
# 数据与行为导航：BuildContextError。
# 继承接口：ValueError。
# [/maint-note:b504d02f6d37]
class BuildContextError(ValueError):
    """An explicit feature-config value is unsafe or invalid."""


# [maint-note:6c9d5a00e459]
# 数据与行为导航：BuildContext。
# 字段 source：str；初始表达式 'default'。
# 字段 database：str | None；初始表达式 None。
# 字段 default_flags：list[str]；初始表达式 field(default_factory=list)。
# 字段 per_file：dict[str, list[str]]；初始表达式 field(default_factory=dict)。
# 字段 extra_flags：list[str]；初始表达式 field(default_factory=list)。
# 字段 dropped：list[dict[str, str]]；初始表达式 field(default_factory=list)。
# [/maint-note:6c9d5a00e459]
@dataclass
class BuildContext:
    source: str = "default"                      # compile_commands | compile_flags | inferred | default
    database: str | None = None                  # project-relative path of the file used
    default_flags: list[str] = field(default_factory=list)
    per_file: dict[str, list[str]] = field(default_factory=dict)
    extra_flags: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)

    # [maint-note:d465d2b2331c]
    # 取值顺序：优先使用当前源文件的编译参数，否则使用默认参数。
    # 用户配置的额外参数追加在末尾；返回新列表以免调用方修改共享状态。
    # 维护导航：flags_for；下列为本函数源码中的接口与路径索引。
    # 输入 relative：str。
    # 返回契约：list[str]。
    # 返回路径：[*self.per_file.get(relative, self.default_flags), *self.extra_flags]。
    # 直接调用：self.per_file.get。
    # [/maint-note:d465d2b2331c]
    def flags_for(self, relative: str) -> list[str]:
        """Analyser flags for a project-relative source, absolute paths throughout."""
        return [*self.per_file.get(relative, self.default_flags), *self.extra_flags]

    # [maint-note:d2616f51a4d4]
    # 维护导航：describe；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 分支条件：flags[index] in PATH_FLAGS + FILE_FLAGS and index + 1 < len(flags)。
    # 遍历边界：flags ← [self.default_flags, *self.per_file.values(), self.extra_flags]。
    # 遍历边界：index < len(flags)。
    # 返回路径：{'source': self.source, 'database': self.database, 'files_with_entries': len(self.per_file), 'include_dirs': len({value for (flag, value) in pairs if flag in PATH_FLAGS}), 'macros': sum((1 for (flag, _) in pairs if flag.startswith(MACRO_FLA。
    # 直接调用：set。
    # 直接调用：self.per_file.values。
    # 直接调用：len。
    # 直接调用：pairs.add。
    # 直接调用：sum。
    # 直接调用：flag.startswith。
    # 集合派生：(flag, value) 来自 pairs；过滤 flag in PATH_FLAGS。
    # 集合派生：(flag, _) 来自 pairs；过滤 flag.startswith(MACRO_FLAGS)。
    # [/maint-note:d2616f51a4d4]
    def describe(self) -> dict[str, Any]:
        """Distinct include directories and macros across every file's flags."""
        pairs: set[tuple[str, str | None]] = set()
        for flags in [self.default_flags, *self.per_file.values(), self.extra_flags]:
            index = 0
            while index < len(flags):
                if flags[index] in PATH_FLAGS + FILE_FLAGS and index + 1 < len(flags):
                    pairs.add((flags[index], flags[index + 1]))
                    index += 2
                else:
                    pairs.add((flags[index], None))
                    index += 1
        return {
            "source": self.source, "database": self.database,
            "files_with_entries": len(self.per_file),
            "include_dirs": len({value for flag, value in pairs if flag in PATH_FLAGS}),
            "macros": sum(1 for flag, _ in pairs if flag.startswith(MACRO_FLAGS)),
            "dropped": self.dropped[:20], "dropped_count": len(self.dropped),
        }


# [maint-note:f235f753fbe7]
# 维护导航：_inside；下列为本函数源码中的接口与路径索引。
# 输入 root：Path。
# 输入 candidate：Path。
# 返回契约：Path | None。
# 返回路径：None。
# 返回路径：resolved。
# 捕获异常：ValueError。
# 直接调用：candidate.resolve。
# 直接调用：resolved.relative_to。
# [/maint-note:f235f753fbe7]
def _inside(root: Path, candidate: Path) -> Path | None:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


# [maint-note:01c8bdda4690]
# 维护导航：_split_flag；下列为本函数源码中的接口与路径索引。
# 输入 arguments：list[str]。
# 输入 index：int。
# 输入 names：tuple[str, ...]。
# 返回契约：tuple[str, str, int] | None。
# 分支条件：argument == name。
# 分支条件：index + 1 < len(arguments)。
# 分支条件：argument.startswith(name)。
# 遍历边界：name ← sorted(names, key=len, reverse=True)。
# 返回路径：(name, arguments[index + 1], index + 2)。
# 返回路径：None。
# 返回路径：(name, value[1:] if value.startswith('=') and name not in MACRO_FLAGS else value, index + 1)。
# 直接调用：sorted。
# 直接调用：len。
# 直接调用：argument.startswith。
# 直接调用：value.startswith。
# 表达式分流：value.startswith('=') and name not in MACRO_FLAGS；真值与假值分支分别计算，不同时执行。
# [/maint-note:01c8bdda4690]
def _split_flag(arguments: list[str], index: int, names: tuple[str, ...]) -> tuple[str, str, int] | None:
    """Recognise ``-Ifoo``, ``-I foo`` and ``-I=foo``; return (name, value, next_index)."""
    argument = arguments[index]
    for name in sorted(names, key=len, reverse=True):
        if argument == name:
            if index + 1 < len(arguments):
                return name, arguments[index + 1], index + 2
            return None
        if argument.startswith(name):
            value = argument[len(name):]
            return name, value[1:] if value.startswith("=") and name not in MACRO_FLAGS else value, index + 1
    return None


# [maint-note:ee05bf04ec35]
# 信任边界：编译数据库由外部项目提供，不允许直接交给分析器执行。
# 只保留 include、宏定义与语言标准参数，插件加载等参数会被拒绝。
# 路径参数解析后必须仍在项目内，避免读取外部头文件或强制包含文件。
# 数据库违规项记录后丢弃；显式配置违规项以严格模式抛错。
# 响应文件不会被展开，避免参数白名单被间接绕过。
# 维护导航：sanitize_flags；下列为本函数源码中的接口与路径索引。
# 输入 arguments：Iterable[str]。
# 输入 directory：Path。
# 输入 root：Path。
# 输入 origin：str。
# 输入 dropped：list[dict[str, str]] | None。
# 输入 strict：bool；缺省表达式 False。
# 返回契约：list[str]。
# 分支条件：strict。
# 分支条件：dropped is not None。
# 分支条件：'\x00' in argument or '\n' in argument。
# 分支条件：parsed is None。
# 分支条件：_STANDARD.match(argument)。
# 分支条件：argument.startswith('@')。
# 分支条件：argument.startswith('-')。
# 分支条件：name in MACRO_FLAGS。
# 分支条件：not _MACRO_NAME.match(macro)。
# 分支条件：target is None。
# 分支条件：name in FILE_FLAGS and (not target.is_file()) or (name in PATH_FLAGS and (not target.is_dir()))。
# 遍历边界：index < len(arguments)。
# 返回路径：kept。
# 显式抛出：BuildContextError(f'{origin}: {flag!r} {reason}')。
# 直接调用：BuildContextError。
# 直接调用：dropped.append。
# 直接调用：list。
# 直接调用：len。
# 直接调用：refuse。
# 直接调用：_split_flag。
# 直接调用：_STANDARD.match。
# 直接调用：kept.append。
# 直接调用：argument.startswith。
# 直接调用：value.split。
# 直接调用：_MACRO_NAME.match。
# 直接调用：_inside。
# 直接调用：target.is_file。
# 直接调用：target.is_dir。
# 直接调用：kept.extend。
# 直接调用：str。
# [/maint-note:ee05bf04ec35]
def sanitize_flags(arguments: Iterable[str], directory: Path, root: Path, origin: str,
                   dropped: list[dict[str, str]] | None, strict: bool = False) -> list[str]:
    """Keep include, macro and -std flags, with paths resolved inside ``root``.

    ``strict`` raises BuildContextError instead of dropping (feature-config values).
    """
    # [maint-note:c2e1470a8dc6]
    # 维护导航：refuse；下列为本函数源码中的接口与路径索引。
    # 输入 flag：str。
    # 输入 reason：str。
    # 返回契约：None。
    # 分支条件：strict。
    # 分支条件：dropped is not None。
    # 显式抛出：BuildContextError(f'{origin}: {flag!r} {reason}')。
    # 直接调用：BuildContextError。
    # 直接调用：dropped.append。
    # [/maint-note:c2e1470a8dc6]
    def refuse(flag: str, reason: str) -> None:
        if strict:
            raise BuildContextError(f"{origin}: {flag!r} {reason}")
        if dropped is not None:
            dropped.append({"origin": origin, "flag": flag, "reason": reason})

    arguments, kept, index = list(arguments), [], 0
    while index < len(arguments):
        argument = arguments[index]
        if "\x00" in argument or "\n" in argument:
            refuse(argument, "contains a control character")
            index += 1
            continue
        parsed = _split_flag(arguments, index, PATH_FLAGS + FILE_FLAGS + MACRO_FLAGS)
        if parsed is None:
            if _STANDARD.match(argument):
                kept.append(argument)
            elif argument.startswith("@"):
                refuse(argument, "is a response file; its arguments are not read")
            elif argument.startswith("-"):
                refuse(argument, "is not an include, macro or -std flag")
            index += 1
            continue
        name, value, index = parsed
        if name in MACRO_FLAGS:
            macro = value.split("=", 1)[0]
            if not _MACRO_NAME.match(macro):
                refuse(f"{name}{value}", "is not a valid macro definition")
                continue
            kept.append(f"{name}{value}")
            continue
        target = _inside(root, directory / value)
        if target is None:
            refuse(f"{name} {value}", "points outside the project directory")
            continue
        if name in FILE_FLAGS and not target.is_file() or name in PATH_FLAGS and not target.is_dir():
            refuse(f"{name} {value}", "does not exist")
            continue
        kept.extend([name, str(target)])
    return kept


# [maint-note:76bbb81d270d]
# 维护导航：_split_list；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 name：str。
# 返回契约：list[str]。
# 分支条件：value in (None, '')。
# 分支条件：isinstance(value, str)。
# 分支条件：isinstance(value, list) and all((isinstance(item, str) for item in value))。
# 返回路径：[]。
# 返回路径：[item.strip() for item in re.split('[\\n,;]', value) if item.strip()]。
# 返回路径：[item.strip() for item in value if item.strip()]。
# 显式抛出：BuildContextError(f'{name} must be a list of strings or a comma/newline separated string')。
# 直接调用：isinstance。
# 直接调用：item.strip。
# 直接调用：re.split。
# 直接调用：all。
# 直接调用：BuildContextError。
# 集合派生：item 来自 re.split('[\\n,;]', value)；过滤 item.strip()。
# 集合派生：item 来自 value。
# 集合派生：item 来自 value；过滤 item.strip()。
# [/maint-note:76bbb81d270d]
def _split_list(value: Any, name: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\n,;]", value) if item.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise BuildContextError(f"{name} must be a list of strings or a comma/newline separated string")


# [maint-note:9136f845271a]
# 维护导航：validate_config；下列为本函数源码中的接口与路径索引。
# 输入 config：dict[str, Any]。
# 返回契约：str | None。
# 分支条件：len(values) > 200。
# 分支条件：Path(value).is_absolute() or value.startswith('~')。
# 分支条件：name == 'include_dirs'。
# 分支条件：name == 'defines' and (not _MACRO_NAME.match(value.split('=', 1)[0]))。
# 分支条件：not isinstance(path, str)。
# 分支条件：path and (Path(path).is_absolute() or path.startswith('~'))。
# 遍历边界：name ← ('include_dirs', 'defines')。
# 遍历边界：value ← values。
# 返回路径：f'{name} accepts at most 200 entries'。
# 返回路径：'include_dirs must be project-relative paths'。
# 返回路径：f'defines entry {value!r} is not a valid macro definition'。
# 返回路径：'compile_commands must be a project-relative path'。
# 返回路径：str(error)。
# 返回路径：None。
# 捕获异常：BuildContextError。
# 直接调用：_split_list。
# 直接调用：config.get。
# 直接调用：len。
# 直接调用：Path(value).is_absolute。
# 直接调用：Path。
# 直接调用：value.startswith。
# 直接调用：_MACRO_NAME.match。
# 直接调用：value.split。
# 直接调用：isinstance。
# 直接调用：Path(path).is_absolute。
# 直接调用：path.startswith。
# 直接调用：str。
# [/maint-note:9136f845271a]
def validate_config(config: dict[str, Any]) -> str | None:
    """Shape checks that need no project directory (paths are checked on load)."""
    try:
        for name in ("include_dirs", "defines"):
            values = _split_list(config.get(name), name)
            if len(values) > 200:
                return f"{name} accepts at most 200 entries"
            for value in values:
                if Path(value).is_absolute() or value.startswith("~"):
                    if name == "include_dirs":
                        return "include_dirs must be project-relative paths"
                if name == "defines" and not _MACRO_NAME.match(value.split("=", 1)[0]):
                    return f"defines entry {value!r} is not a valid macro definition"
        path = config.get("compile_commands", "")
        if not isinstance(path, str):
            return "compile_commands must be a project-relative path"
        if path and (Path(path).is_absolute() or path.startswith("~")):
            return "compile_commands must be a project-relative path"
    except BuildContextError as error:
        return str(error)
    return None


# [maint-note:114736e8690c]
# 维护导航：_load_database；下列为本函数源码中的接口与路径索引。
# 输入 root：Path。
# 输入 path：Path。
# 输入 context：BuildContext。
# 返回契约：None。
# 分支条件：path.stat().st_size > MAX_DATABASE_BYTES。
# 分支条件：not isinstance(entries, list)。
# 分支条件：not isinstance(entry, dict) or not isinstance(entry.get('file'), str)。
# 分支条件：source is None。
# 分支条件：not isinstance(arguments, list)。
# 分支条件：skip。
# 分支条件：argument in {'-o', '-MF', '-MT', '-MQ'}。
# 分支条件：argument in {'-c', '-S', '-E'} or Path(argument).name == source.name。
# 遍历边界：entry ← entries。
# 遍历边界：argument ← arguments[1:]。
# 显式抛出：BuildContextError(f'{path.name} exceeds {MAX_DATABASE_BYTES // 1000000} MB')。
# 显式抛出：BuildContextError(f'{path.relative_to(root).as_posix()} is not valid JSON: {error}')。
# 显式抛出：BuildContextError(f'{path.relative_to(root).as_posix()} must be a JSON array')。
# 捕获异常：(OSError, UnicodeDecodeError, json.JSONDecodeError)。
# 捕获异常：ValueError。
# 写入目标：context.per_file[relative]。
# 直接调用：path.stat。
# 直接调用：BuildContextError。
# 直接调用：json.loads。
# 直接调用：path.read_text。
# 直接调用：path.relative_to(root).as_posix。
# 直接调用：path.relative_to。
# 直接调用：isinstance。
# 直接调用：entry.get。
# 直接调用：Path。
# 直接调用：directory.is_absolute。
# 直接调用：_inside。
# 直接调用：context.dropped.append。
# 直接调用：shlex.split。
# 直接调用：source.relative_to(root).as_posix。
# 直接调用：source.relative_to。
# 直接调用：body.append。
# 直接调用：sanitize_flags。
# 表达式分流：directory.is_absolute()；真值与假值分支分别计算，不同时执行。
# [/maint-note:114736e8690c]
def _load_database(root: Path, path: Path, context: BuildContext) -> None:
    if path.stat().st_size > MAX_DATABASE_BYTES:
        raise BuildContextError(f"{path.name} exceeds {MAX_DATABASE_BYTES // 1_000_000} MB")
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildContextError(f"{path.relative_to(root).as_posix()} is not valid JSON: {error}") from None
    if not isinstance(entries, list):
        raise BuildContextError(f"{path.relative_to(root).as_posix()} must be a JSON array")
    base = path.parent
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            continue
        # A relative "directory" is resolved against the database's own folder, so a
        # database shipped inside the project works wherever the project is unpacked.
        directory = Path(entry.get("directory") or ".")
        directory = directory if directory.is_absolute() else base / directory
        source = _inside(root, directory / entry["file"])
        if source is None:
            context.dropped.append({"origin": "compile_commands", "flag": entry["file"],
                                    "reason": "source file outside the project directory"})
            continue
        arguments = entry.get("arguments")
        if not isinstance(arguments, list):
            try:
                arguments = shlex.split(entry.get("command") or "")
            except ValueError:
                continue
        relative = source.relative_to(root).as_posix()
        # Drop the compiler, the output and the source itself; keep the rest for sanitising.
        body, skip = [], False
        for argument in arguments[1:]:
            if skip:
                skip = False
                continue
            if argument in {"-o", "-MF", "-MT", "-MQ"}:
                skip = True
                continue
            if argument in {"-c", "-S", "-E"} or Path(argument).name == source.name:
                continue
            body.append(argument)
        effective = _inside(root, directory) or root
        context.per_file[relative] = sanitize_flags(body, effective, root, "compile_commands", context.dropped)


# [maint-note:84b776a0edb2]
# 维护导航：system_include_dirs；下列为本函数源码中的接口与路径索引。
# 返回契约：tuple[str, ...]。
# 分支条件：executable is None。
# 分支条件：line.startswith('#include <...> search starts here')。
# 分支条件：line.startswith('End of search list')。
# 分支条件：collecting and line.strip()。
# 遍历边界：line ← lines。
# 返回路径：tuple((path for path in ('/usr/local/include', '/usr/include') if Path(path).is_dir()))。
# 返回路径：()。
# 返回路径：tuple(found)。
# 捕获异常：(OSError, subprocess.TimeoutExpired)。
# 直接调用：shutil.which。
# 直接调用：tuple。
# 直接调用：Path(path).is_dir。
# 直接调用：Path。
# 直接调用：subprocess.run。
# 直接调用：result.stderr.splitlines。
# 直接调用：line.startswith。
# 直接调用：line.strip。
# 直接调用：found.append。
# 直接调用：line.strip().split。
# 集合派生：path 来自 ('/usr/local/include', '/usr/include')；过滤 Path(path).is_dir()。
# [/maint-note:84b776a0edb2]
@lru_cache(maxsize=1)
def system_include_dirs() -> tuple[str, ...]:
    """The host compiler's system include search path (empty when no clang)."""
    executable = shutil.which("clang")
    if executable is None:
        return tuple(path for path in ("/usr/local/include", "/usr/include") if Path(path).is_dir())
    try:
        result = subprocess.run([executable, "-xc", "-E", "-v", "-"], input="", capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ()
    lines, collecting, found = result.stderr.splitlines(), False, []
    for line in lines:
        if line.startswith("#include <...> search starts here"):
            collecting = True
        elif line.startswith("End of search list"):
            break
        elif collecting and line.strip():
            found.append(line.strip().split(" (")[0])
    return tuple(found)


# [maint-note:6cb648a54b80]
# 维护导航：_infer；下列为本函数源码中的接口与路径索引。
# 输入 root：Path。
# 输入 targets：list[Path]。
# 输入 context：BuildContext。
# 返回契约：None。
# 分支条件：path.suffix.lower() in HEADER_SUFFIXES and path.is_file() and (not any((part.startswith('.') for part in path.relative_to(root).parts)))。
# 分支条件：current in seen。
# 分支条件：kind == '"' and local.is_file()。
# 分支条件：resolved is not None。
# 分支条件：any(((directory / name).is_file() for directory in system))。
# 分支条件：not candidates。
# 遍历边界：path ← root.rglob('*')。
# 遍历边界：queue and len(seen) < MAX_INFERRED_HEADERS。
# 遍历边界：(kind, name) ← _INCLUDE.findall(text)。
# 遍历边界：directory ← regular[1:]。
# 遍历边界：directory ← after。
# 捕获异常：OSError。
# 写入目标：context.default_flags。
# 直接调用：root.rglob。
# 直接调用：path.suffix.lower。
# 直接调用：path.is_file。
# 直接调用：any。
# 直接调用：part.startswith。
# 直接调用：path.relative_to。
# 直接调用：headers.setdefault(path.name, []).append。
# 直接调用：headers.setdefault。
# 直接调用：Path。
# 直接调用：system_include_dirs。
# 直接调用：list。
# 直接调用：set。
# 直接调用：len。
# 直接调用：queue.pop。
# 直接调用：seen.add。
# 直接调用：current.read_text。
# 直接调用：_INCLUDE.findall。
# 直接调用：local.is_file。
# 直接调用：queue.append。
# 直接调用：local.resolve。
# 直接调用：next。
# 直接调用：(directory / name).is_file。
# 直接调用：resolved.resolve。
# 直接调用：headers.get。
# 直接调用：path.as_posix().endswith。
# 直接调用：path.as_posix。
# 直接调用：max。
# 直接调用：best.as_posix。
# 直接调用：(after if _shadows_system(directory, system) else regular).append。
# 直接调用：_shadows_system。
# 直接调用：best.resolve。
# 直接调用：flags.extend。
# 直接调用：str。
# 集合派生：part 来自 path.relative_to(root).parts。
# 集合派生：path 来自 system_include_dirs()。
# 集合派生：directory 来自 [*regular, *after]；过滤 (directory / name).is_file()。
# 集合派生：directory 来自 system。
# 集合派生：path 来自 headers.get(Path(name).name, [])；过滤 path.as_posix().endswith('/' + name)。
# 表达式分流：_shadows_system(directory, system)；真值与假值分支分别计算，不同时执行。
# [/maint-note:6cb648a54b80]
def _infer(root: Path, targets: list[Path], context: BuildContext) -> None:
    """Add the project directories that resolve each target's #include directives."""
    headers: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in HEADER_SUFFIXES and path.is_file() and \
                not any(part.startswith(".") for part in path.relative_to(root).parts):
            headers.setdefault(path.name, []).append(path)
    system = [Path(path) for path in system_include_dirs()]
    regular: list[Path] = [root]
    after: list[Path] = []
    queue, seen = list(targets), set()
    while queue and len(seen) < MAX_INFERRED_HEADERS:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            text = current.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for kind, name in _INCLUDE.findall(text):
            local = current.parent / name
            if kind == '"' and local.is_file():
                queue.append(local.resolve())
                continue
            resolved = next((directory / name for directory in [*regular, *after] if (directory / name).is_file()), None)
            if resolved is not None:
                queue.append(resolved.resolve())
                continue
            if any((directory / name).is_file() for directory in system):
                continue  # the host provides it; a project copy must not be pulled in for it
            candidates = [path for path in headers.get(Path(name).name, [])
                          if path.as_posix().endswith("/" + name)]
            if not candidates:
                continue
            # Several project copies: take the one closest to the including file.
            best = max(candidates, key=lambda path: len(_common(path.parts, current.parts)))
            directory = Path(best.as_posix()[: -len(name) - 1])
            # A directory that also holds copies of host headers (RT-Thread's libc
            # compatibility layer) is searched after the system, never before it.
            (after if _shadows_system(directory, system) else regular).append(directory)
            queue.append(best.resolve())
    flags = []
    for directory in regular[1:]:
        flags.extend(["-I", str(directory)])
    for directory in after:
        flags.extend(["-idirafter", str(directory)])
    context.default_flags = flags


# [maint-note:4ec14fa197e0]
# 维护导航：_shadows_system_cached；下列为本函数源码中的接口与路径索引。
# 输入 directory：str。
# 输入 system：tuple[str, ...]。
# 返回契约：bool。
# 分支条件：len(relative.parts) <= 2 and any(((Path(base) / relative).is_file() for base in system))。
# 遍历边界：header ← root.rglob('*.h')。
# 返回路径：True。
# 返回路径：False。
# 直接调用：Path。
# 直接调用：root.rglob。
# 直接调用：header.relative_to。
# 直接调用：len。
# 直接调用：any。
# 直接调用：(Path(base) / relative).is_file。
# 集合派生：base 来自 system。
# [/maint-note:4ec14fa197e0]
@lru_cache(maxsize=512)
def _shadows_system_cached(directory: str, system: tuple[str, ...]) -> bool:
    root = Path(directory)
    for header in root.rglob("*.h"):
        relative = header.relative_to(root)
        if len(relative.parts) <= 2 and any((Path(base) / relative).is_file() for base in system):
            return True
    return False


# [maint-note:613e36b7c48e]
# 维护导航：_shadows_system；下列为本函数源码中的接口与路径索引。
# 输入 directory：Path。
# 输入 system：list[Path]。
# 返回契约：bool。
# 返回路径：_shadows_system_cached(str(directory), tuple((str(path) for path in system)))。
# 直接调用：_shadows_system_cached。
# 直接调用：str。
# 直接调用：tuple。
# 集合派生：path 来自 system。
# [/maint-note:613e36b7c48e]
def _shadows_system(directory: Path, system: list[Path]) -> bool:
    return _shadows_system_cached(str(directory), tuple(str(path) for path in system))


# [maint-note:49c1a3743fb4]
# 维护导航：_common；下列为本函数源码中的接口与路径索引。
# 输入 left：tuple[str, ...]。
# 输入 right：tuple[str, ...]。
# 返回契约：tuple[str, ...]。
# 分支条件：a != b。
# 遍历边界：(a, b) ← zip(left, right)。
# 返回路径：tuple(common)。
# 直接调用：zip。
# 直接调用：common.append。
# 直接调用：tuple。
# [/maint-note:49c1a3743fb4]
def _common(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    common = []
    for a, b in zip(left, right):
        if a != b:
            break
        common.append(a)
    return tuple(common)


# [maint-note:bb55330827b3]
# 维护导航：load_build_context；下列为本函数源码中的接口与路径索引。
# 输入 project_dir：str | Path。
# 输入 config：dict[str, Any] | None；缺省表达式 None。
# 输入 target_files：Iterable[str | Path]；缺省表达式 ()。
# 返回契约：BuildContext。
# 分支条件：requested。
# 分支条件：path is None。
# 分支条件：path.is_dir()。
# 分支条件：not path.is_file()。
# 分支条件：path is not None。
# 分支条件：(root / 'compile_flags.txt').is_file()。
# 分支条件：sources。
# 遍历边界：directory ← _split_list(config.get('include_dirs'), 'include_dirs')。
# 遍历边界：define ← _split_list(config.get('defines'), 'defines')。
# 返回路径：context。
# 显式抛出：BuildContextError(f'compile_commands {requested!r} points outside the project directory')。
# 显式抛出：BuildContextError(f'compile_commands {requested!r} does not exist')。
# 写入目标：context.default_flags。
# 写入目标：context.source。
# 直接调用：Path(project_dir).resolve。
# 直接调用：Path。
# 直接调用：BuildContext。
# 直接调用：str(config.get('compile_commands') or '').strip。
# 直接调用：str。
# 直接调用：config.get。
# 直接调用：_inside。
# 直接调用：BuildContextError。
# 直接调用：path.is_dir。
# 直接调用：path.is_file。
# 直接调用：path.relative_to(root).as_posix。
# 直接调用：path.relative_to。
# 直接调用：_load_database。
# 直接调用：(root / 'compile_flags.txt').is_file。
# 直接调用：(root / 'compile_flags.txt').read_text(encoding='utf-8', errors='replace').splitlines。
# 直接调用：(root / 'compile_flags.txt').read_text。
# 直接调用：sanitize_flags。
# 直接调用：line.strip。
# 直接调用：Path(p).is_absolute。
# 直接调用：p.resolve。
# 直接调用：p.suffix.lower。
# 直接调用：_infer。
# 直接调用：_split_list。
# 直接调用：context.extra_flags.extend。
# 表达式分流：path.is_file()；真值与假值分支分别计算，不同时执行。
# 集合派生：line 来自 lines；过滤 line.strip()。
# 表达式分流：Path(p).is_absolute()；真值与假值分支分别计算，不同时执行。
# 集合派生：p 来自 target_files。
# 集合派生：p 来自 sources；过滤 p.suffix.lower() in SOURCE_SUFFIXES | HEADER_SUFFIXES。
# [/maint-note:bb55330827b3]
def load_build_context(project_dir: str | Path, config: dict[str, Any] | None = None,
                       target_files: Iterable[str | Path] = ()) -> BuildContext:
    """Resolve the compile context for a scan. Raises BuildContextError for bad config."""
    root = Path(project_dir).resolve()
    config = config or {}
    context = BuildContext()
    requested = str(config.get("compile_commands") or "").strip()
    if requested:
        path = _inside(root, root / requested)
        if path is None:
            raise BuildContextError(f"compile_commands {requested!r} points outside the project directory")
        if path.is_dir():
            path = path / "compile_commands.json"
        if not path.is_file():
            raise BuildContextError(f"compile_commands {requested!r} does not exist")
    else:
        path = root / "compile_commands.json"
        path = path if path.is_file() else None
    if path is not None:
        context.source, context.database = "compile_commands", path.relative_to(root).as_posix()
        _load_database(root, path, context)
    elif (root / "compile_flags.txt").is_file():
        context.source, context.database = "compile_flags", "compile_flags.txt"
        lines = (root / "compile_flags.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        context.default_flags = sanitize_flags([line.strip() for line in lines if line.strip()],
                                               root, root, "compile_flags.txt", context.dropped)
    else:
        sources = [Path(p) if Path(p).is_absolute() else root / p for p in target_files]
        sources = [p.resolve() for p in sources if p.suffix.lower() in SOURCE_SUFFIXES | HEADER_SUFFIXES]
        if sources:
            context.source = "inferred"
            _infer(root, sources, context)
    for directory in _split_list(config.get("include_dirs"), "include_dirs"):
        context.extra_flags.extend(sanitize_flags(["-I", directory], root, root, "include_dirs", None, strict=True))
    for define in _split_list(config.get("defines"), "defines"):
        context.extra_flags.extend(sanitize_flags([f"-D{define}"], root, root, "defines", None, strict=True))
    return context
