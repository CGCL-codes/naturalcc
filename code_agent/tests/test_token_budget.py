import hashlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from code_agent.agent_core.token_budget import (
    ContextHardLimitExceeded,
    DeepSeekModelProfile,
    DeepSeekTokenCounter,
    TokenizerUnavailable,
)
from code_agent.agent_core.model_gateway import DeepSeekRequestSerializer
from code_agent.scripts.install_deepseek_tokenizer import (
    TokenizerIntegrityError,
    install_verified_archive,
)


def test_deepseek_profile_rejects_invalid_budget_geometry():
    with pytest.raises(ValueError, match="target_ratio"):
        DeepSeekModelProfile(
            context_window_tokens=1000,
            default_output_reserve_tokens=100,
            safety_margin_tokens=50,
            compaction_trigger_ratio=0.5,
            compaction_target_ratio=0.7,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_output_reserve_tokens", -1),
        ("safety_margin_tokens", -1),
        ("provider_framing_tokens", -1),
        ("analyzer_output_tokens", 0),
        ("summarizer_output_tokens", 0),
    ],
)
def test_deepseek_profile_rejects_negative_or_zero_token_components(
    field: str, value: int
):
    with pytest.raises(ValueError, match=field):
        DeepSeekModelProfile(**{field: value})


def test_token_counter_fails_closed_when_local_tokenizer_is_missing(tmp_path: Path):
    with pytest.raises(TokenizerUnavailable, match="DeepSeek tokenizer"):
        DeepSeekTokenCounter.from_directory(tmp_path / "missing")


def test_token_counter_uses_fast_native_tokenizers_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        '{"chat_template": "{{ messages }}", "bos_token": "<s>"}',
        encoding="utf-8",
    )
    sentinel = object()

    class FakeTokenizer:
        @staticmethod
        def from_file(path: str):
            assert path == str(root.resolve() / "tokenizer.json")
            return sentinel

    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(Tokenizer=FakeTokenizer),
    )
    monkeypatch.setitem(sys.modules, "transformers", None)

    counter = DeepSeekTokenCounter.from_directory(root)

    assert counter.tokenizer is sentinel


@pytest.fixture(scope="module")
def real_counter():
    root = Path(__file__).resolve().parents[1] / "resources" / "deepseek_v3_tokenizer"
    return DeepSeekTokenCounter.from_directory(root)


@pytest.fixture
def serializer():
    return DeepSeekRequestSerializer()


def test_chat_template_renders_added_tokens_as_token_text(real_counter):
    rendered = real_counter._render_chat(
        [{"role": "user", "content": "Hello!"}]
    )

    assert rendered == (
        "<｜begin▁of▁sentence｜><｜User｜>Hello!<｜Assistant｜>"
    )
    assert "'__type': 'AddedToken'" not in rendered
    assert real_counter.count_text(rendered) == 5


def test_request_count_includes_tools_and_reserve(real_counter, serializer):
    profile = DeepSeekModelProfile(context_window_tokens=8192)
    messages = [{"role": "user", "content": "Read README.md"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "workspace_read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]

    without_tools = real_counter.count_request(messages, [], serializer, profile)
    with_tools = real_counter.count_request(messages, tools, serializer, profile)

    assert with_tools.tool_schema_tokens > 0
    assert with_tools.total_with_reserve > without_tools.total_with_reserve


def test_hard_limit_rejects_oversized_required_context(real_counter, serializer):
    profile = DeepSeekModelProfile(
        context_window_tokens=512,
        default_output_reserve_tokens=128,
        safety_margin_tokens=64,
    )

    with pytest.raises(ContextHardLimitExceeded):
        real_counter.assert_fits(
            [{"role": "system", "content": "x" * 5000}],
            [],
            serializer,
            profile,
        )


def _tokenizer_archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_installer_rejects_archive_hash_mismatch_without_writing(tmp_path: Path):
    archive = _tokenizer_archive({"deepseek_v3_tokenizer/tokenizer.json": b"bad"})

    with pytest.raises(TokenizerIntegrityError, match="archive SHA-256"):
        install_verified_archive(
            archive,
            tmp_path / "tokenizer",
            archive_sha256="0" * 64,
            files={"deepseek_v3_tokenizer/tokenizer.json": hashlib.sha256(b"bad").hexdigest()},
        )

    assert not (tmp_path / "tokenizer").exists()


def test_installer_writes_only_verified_allowlisted_files(tmp_path: Path):
    contents = {
        "deepseek_v3_tokenizer/tokenizer.json": b'{"version": "1.0"}',
        "deepseek_v3_tokenizer/tokenizer_config.json": b'{"model_max_length": 1}',
        "deepseek_v3_tokenizer/deepseek_tokenizer.py": b"raise SystemExit",
        "__MACOSX/._tokenizer.json": b"metadata",
    }
    archive = _tokenizer_archive(contents)
    allowed = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in contents.items()
        if name.endswith(("tokenizer.json", "tokenizer_config.json"))
        and not name.startswith("__MACOSX")
    }
    destination = tmp_path / "tokenizer"

    installed = install_verified_archive(
        archive,
        destination,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        files=allowed,
    )

    assert installed == [destination / "tokenizer.json", destination / "tokenizer_config.json"]
    assert (destination / "tokenizer.json").read_bytes() == contents[
        "deepseek_v3_tokenizer/tokenizer.json"
    ]
    assert not (destination / "deepseek_tokenizer.py").exists()
    assert not (tmp_path / "__MACOSX").exists()
