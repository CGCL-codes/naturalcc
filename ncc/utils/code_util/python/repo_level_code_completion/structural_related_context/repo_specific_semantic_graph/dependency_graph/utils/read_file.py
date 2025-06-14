from pathlib import Path
from typing import Tuple

import chardet

from dependency_graph.models import PathLike

default_encoding = "utf-8"
common_encodings = (
    "utf-8",
    "utf-16",
    "latin-1",
    "ascii",
    "windows-1252",
    "cp1251",
    "cp1253",
    "cp1254",
    "cp1255",
    "cp1256",
    "shift_jis",
    "big5",
    "gb2312",
)


def detect_file_encoding(file_path: Path) -> str:
    """Function to detect encoding"""
    # Read the file as binary data
    raw_data = file_path.read_bytes()
    # Detect encoding
    detected = chardet.detect(raw_data)
    encoding = detected["encoding"]
    return encoding


def read_file_with_encodings(file_path: Path, encodings: Tuple[str]) -> Tuple[str, str]:
    """Attempt to read a file using various encodings, return content if successful"""
    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            return content, encoding
        except (UnicodeDecodeError, TypeError, ValueError, UnicodeError):
            continue
    raise ValueError(
        f"Could not read file with any of the provided encodings: {encodings}"
    )


def read_file_with_limit(content: str, max_lines_to_read: int = None) -> str:
    """Helper function to return a limited number of lines from the content."""
    if max_lines_to_read is None:
        return content
    else:
        return "\n".join(content.splitlines()[:max_lines_to_read])


def read_file_to_string(file_path: PathLike, *, max_lines_to_read: int = None) -> str:
    """Function to detect encoding and read file to string with an optional line limit."""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        content, _ = read_file_with_encodings(file_path, (default_encoding,))
        return read_file_with_limit(content, max_lines_to_read)
    except ValueError:
        pass

    try:
        detected_encoding = detect_file_encoding(file_path)
        # Read the file with the detected encoding
        content, _ = read_file_with_encodings(file_path, (detected_encoding,))
        return read_file_with_limit(content, max_lines_to_read)
    except ValueError:
        pass

    try:
        content, _ = read_file_with_encodings(file_path, common_encodings)
        return read_file_with_limit(content, max_lines_to_read)
    except ValueError:
        pass

    raise ValueError(f"Could not read file: {file_path}")
