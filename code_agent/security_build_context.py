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


class BuildContextError(ValueError):
    """An explicit feature-config value is unsafe or invalid."""


@dataclass
class BuildContext:
    source: str = "default"                      # compile_commands | compile_flags | inferred | default
    database: str | None = None                  # project-relative path of the file used
    default_flags: list[str] = field(default_factory=list)
    per_file: dict[str, list[str]] = field(default_factory=dict)
    extra_flags: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)

    def flags_for(self, relative: str) -> list[str]:
        """Analyser flags for a project-relative source, absolute paths throughout."""
        return [*self.per_file.get(relative, self.default_flags), *self.extra_flags]

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


def _inside(root: Path, candidate: Path) -> Path | None:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


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


def sanitize_flags(arguments: Iterable[str], directory: Path, root: Path, origin: str,
                   dropped: list[dict[str, str]] | None, strict: bool = False) -> list[str]:
    """Keep include, macro and -std flags, with paths resolved inside ``root``.

    ``strict`` raises BuildContextError instead of dropping (feature-config values).
    """
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


def _split_list(value: Any, name: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\n,;]", value) if item.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise BuildContextError(f"{name} must be a list of strings or a comma/newline separated string")


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


@lru_cache(maxsize=512)
def _shadows_system_cached(directory: str, system: tuple[str, ...]) -> bool:
    root = Path(directory)
    for header in root.rglob("*.h"):
        relative = header.relative_to(root)
        if len(relative.parts) <= 2 and any((Path(base) / relative).is_file() for base in system):
            return True
    return False


def _shadows_system(directory: Path, system: list[Path]) -> bool:
    return _shadows_system_cached(str(directory), tuple(str(path) for path in system))


def _common(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    common = []
    for a, b in zip(left, right):
        if a != b:
            break
        common.append(a)
    return tuple(common)


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
