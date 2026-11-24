from __future__ import annotations

from codecs import getincrementaldecoder
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path


_INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md")
_MAX_FILE_BYTES = 4_096
_MAX_TOTAL_BYTES = 8_192
_MAX_FILES = 16
_MIN_FILE_BYTES = _MAX_TOTAL_BYTES // _MAX_FILES


def _rule_paths(directories: Iterable[Path]) -> Iterator[Path]:
    for directory in sorted(directories, key=lambda path: (len(path.parts), path.as_posix())):
        for filename in _INSTRUCTION_FILES:
            path = directory / filename
            try:
                if not path.is_symlink() and path.is_file():
                    yield path
            except OSError:
                continue


def load_repository_instructions(
    workspace: str | Path, target_files: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Load root rules and ancestors of requested targets in increasing specificity."""
    root = Path(workspace).resolve()
    directories = {root}
    for value in target_files:
        try:
            requested = root / value
            if not requested.is_relative_to(root) or ".." in requested.relative_to(root).parts:
                continue
            target = requested.resolve()
            if not target.is_relative_to(root):
                continue
            directory = target if target.is_dir() else target.parent
            while directory != root:
                directories.add(directory)
                directory = directory.parent
        except (OSError, RuntimeError, ValueError):
            continue

    result: list[dict[str, str]] = []
    remaining = _MAX_TOTAL_BYTES
    paths = list(islice(_rule_paths(directories), _MAX_FILES))
    for index, path in enumerate(paths):
        reserve = _MIN_FILE_BYTES * (len(paths) - index - 1)
        limit = min(_MAX_FILE_BYTES, remaining - reserve)
        try:
            if path.resolve(strict=True) != path or not path.is_file():
                continue
            with path.open("rb") as file:
                data = file.read(limit + 1)
        except (OSError, RuntimeError, ValueError):
            continue
        remaining -= min(len(data), limit)
        truncated = len(data) > limit
        decoder = getincrementaldecoder("utf-8-sig")(errors="replace")
        content = decoder.decode(data[:limit], final=not truncated)
        if truncated:
            complete_lines, _, _ = content.rpartition("\n")
            content = complete_lines or content
        content = content.strip()
        if content:
            if truncated:
                content += "\n\n[Repository instructions truncated by the runtime.]"
            result.append({"path": path.relative_to(root).as_posix(), "content": content})
    return result
