from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import Mapping
from urllib.request import Request, urlopen
import zipfile


TOKENIZER_URL = "https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip"
ARCHIVE_SHA256 = "c954ca6f6e54281d72d3c27e2430cea7663f81292b39982e2f97890c66c302de"
FILES = {
    "deepseek_v3_tokenizer/tokenizer.json": (
        "ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d"
    ),
    "deepseek_v3_tokenizer/tokenizer_config.json": (
        "144a6d92b6012baeb4f2ac41d48ed3458e758f977a0fb5caf75ff07698fc844c"
    ),
}
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1] / "resources" / "deepseek_v3_tokenizer"
)


class TokenizerIntegrityError(RuntimeError):
    """Raised when downloaded tokenizer bytes do not match the pinned manifest."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def install_verified_archive(
    archive_bytes: bytes,
    destination: str | Path,
    *,
    archive_sha256: str = ARCHIVE_SHA256,
    files: Mapping[str, str] = FILES,
) -> list[Path]:
    actual_archive_hash = _sha256(archive_bytes)
    if actual_archive_hash != archive_sha256:
        raise TokenizerIntegrityError(
            "DeepSeek tokenizer archive SHA-256 mismatch: "
            f"expected {archive_sha256}, got {actual_archive_hash}"
        )

    verified: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            archive_names = set(archive.namelist())
            for member_name, expected_hash in files.items():
                if member_name not in archive_names:
                    raise TokenizerIntegrityError(
                        f"DeepSeek tokenizer archive is missing {member_name}"
                    )
                content = archive.read(member_name)
                actual_hash = _sha256(content)
                if actual_hash != expected_hash:
                    raise TokenizerIntegrityError(
                        f"DeepSeek tokenizer file SHA-256 mismatch for {member_name}: "
                        f"expected {expected_hash}, got {actual_hash}"
                    )
                verified.append((Path(member_name).name, content))
    except zipfile.BadZipFile as exc:
        raise TokenizerIntegrityError(
            "DeepSeek tokenizer archive is not a valid ZIP file"
        ) from exc

    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for filename, content in verified:
        target = root / filename
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{filename}.",
                suffix=".tmp",
                dir=root,
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
            temporary_path = None
            installed.append(target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return installed


def download_archive(url: str = TOKENIZER_URL) -> bytes:
    request = Request(url, headers={"User-Agent": "naturalcc-code-agent-tokenizer-installer/1"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS URL by default
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the pinned official DeepSeek V3 tokenizer locally."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Destination directory for the two verified tokenizer files.",
    )
    args = parser.parse_args()

    archive_bytes = download_archive()
    installed = install_verified_archive(archive_bytes, args.destination)
    for path in installed:
        print(f"verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
