import json
from pathlib import Path

import pytest

from code_agent.security_build_context import (
    BuildContextError, load_build_context, sanitize_flags, validate_config)


def make_project(root: Path) -> Path:
    (root / "include").mkdir(parents=True)
    (root / "include" / "kernel.h").write_text("#define KERNEL 1\n")
    (root / "src").mkdir()
    (root / "src" / "main.c").write_text('#include <kernel.h>\nint main(void) { return KERNEL; }\n')
    return root


def write_database(root: Path, entries: list[dict]) -> None:
    (root / "compile_commands.json").write_text(json.dumps(entries))


# --- compile_commands.json -------------------------------------------------------

def test_database_with_relative_directory_resolves_inside_the_project(tmp_path):
    """A database shipped inside a project must work wherever the project is unpacked."""
    root = make_project(tmp_path / "project")
    write_database(root, [{"directory": ".", "file": "src/main.c",
                           "arguments": ["cc", "-Iinclude", "-DFOO=1", "-idirafter", "include", "-c", "src/main.c"]}])
    context = load_build_context(root)
    assert context.source == "compile_commands" and context.database == "compile_commands.json"
    flags = context.flags_for("src/main.c")
    assert flags == ["-I", str(root / "include"), "-DFOO=1", "-idirafter", str(root / "include")]


def test_command_string_form_and_per_file_flags(tmp_path):
    root = make_project(tmp_path)
    (root / "src" / "other.c").write_text("int other;\n")
    write_database(root, [
        {"directory": str(root), "file": "src/main.c", "command": "gcc -Iinclude -D__KERNEL__ -o out.o -c src/main.c"},
        {"directory": str(root), "file": str(root / "src" / "other.c"), "command": "gcc -Iinclude -c src/other.c"},
    ])
    context = load_build_context(root)
    assert "-D__KERNEL__" in context.flags_for("src/main.c")
    assert "-D__KERNEL__" not in context.flags_for("src/other.c")
    assert "out.o" not in " ".join(context.flags_for("src/main.c"))


@pytest.mark.parametrize("flag", [
    ["-Xclang", "-load", "-Xclang", "evil.so"],
    ["-fplugin=evil.so"],
    ["@response.rsp"],
    ["-B/tmp/evil"],
    ["--sysroot=/"],
    ["-isysroot", "/"],
])
def test_flags_that_could_load_code_are_never_passed(tmp_path, flag):
    root = make_project(tmp_path)
    write_database(root, [{"directory": ".", "file": "src/main.c", "arguments": ["cc", *flag, "-Iinclude", "-c", "src/main.c"]}])
    context = load_build_context(root)
    passed = context.flags_for("src/main.c")
    assert passed == ["-I", str(root / "include")]
    assert context.dropped, "the refused flag must be reported"


@pytest.mark.parametrize("include", ["/etc", "../outside", "include/../../outside"])
def test_database_paths_outside_the_project_are_dropped(tmp_path, include):
    root = make_project(tmp_path / "project")
    (tmp_path / "outside").mkdir()
    write_database(root, [{"directory": ".", "file": "src/main.c", "arguments": ["cc", "-I", include, "-c", "src/main.c"]}])
    context = load_build_context(root)
    assert context.flags_for("src/main.c") == []
    assert any("outside the project" in item["reason"] for item in context.dropped)


def test_database_entry_for_a_file_outside_the_project_is_ignored(tmp_path):
    root = make_project(tmp_path / "project")
    write_database(root, [{"directory": str(tmp_path), "file": "elsewhere.c", "arguments": ["cc", "-c", "elsewhere.c"]}])
    context = load_build_context(root)
    assert context.per_file == {}
    assert context.dropped


def test_explicit_database_path_must_stay_in_the_project(tmp_path):
    root = make_project(tmp_path / "project")
    with pytest.raises(BuildContextError, match="outside"):
        load_build_context(root, {"compile_commands": "../db.json"})
    with pytest.raises(BuildContextError, match="does not exist"):
        load_build_context(root, {"compile_commands": "build/compile_commands.json"})
    (root / "build").mkdir()
    write_database(root / "build", [{"directory": "..", "file": "src/main.c", "arguments": ["cc", "-Iinclude", "-c", "src/main.c"]}])
    context = load_build_context(root, {"compile_commands": "build"})
    assert context.database == "build/compile_commands.json"
    assert context.flags_for("src/main.c") == ["-I", str(root / "include")]


def test_malformed_database_is_reported(tmp_path):
    root = make_project(tmp_path)
    (root / "compile_commands.json").write_text("{not json")
    with pytest.raises(BuildContextError, match="not valid JSON"):
        load_build_context(root)


# --- compile_flags.txt and feature config ------------------------------------------

def test_compile_flags_txt_applies_to_every_file(tmp_path):
    root = make_project(tmp_path)
    (root / "compile_flags.txt").write_text("-Iinclude\n-DGLOBAL\n-std=gnu99\n")
    context = load_build_context(root)
    assert context.source == "compile_flags"
    assert context.flags_for("src/anything.c") == ["-I", str(root / "include"), "-DGLOBAL", "-std=gnu99"]


def test_config_include_dirs_and_defines_are_appended(tmp_path):
    root = make_project(tmp_path)
    write_database(root, [{"directory": ".", "file": "src/main.c", "arguments": ["cc", "-c", "src/main.c"]}])
    context = load_build_context(root, {"include_dirs": "include", "defines": ["A=1", "B"]})
    assert context.flags_for("src/main.c") == ["-I", str(root / "include"), "-DA=1", "-DB"]


@pytest.mark.parametrize("config, message", [
    ({"include_dirs": ["../outside"]}, "outside the project"),
    ({"include_dirs": ["missing"]}, "does not exist"),
    ({"defines": ["1BAD"]}, "not a valid macro"),
])
def test_unsafe_config_is_rejected_not_dropped(tmp_path, config, message):
    root = make_project(tmp_path / "project")
    (tmp_path / "outside").mkdir()
    with pytest.raises(BuildContextError, match=message):
        load_build_context(root, config)


@pytest.mark.parametrize("config, ok", [
    ({}, True),
    ({"include_dirs": ["include", "components/x/include"], "defines": ["A", "B=2", "F(x)=x"]}, True),
    ({"include_dirs": "include\ncomponents/x/include", "defines": "A,B=2"}, True),
    ({"include_dirs": ["/etc"]}, False),
    ({"include_dirs": 7}, False),
    ({"defines": ["-DBAD"]}, False),
    ({"compile_commands": "/abs/compile_commands.json"}, False),
    ({"compile_commands": 3}, False),
])
def test_validate_config(config, ok):
    assert (validate_config(config) is None) is ok


def test_sanitize_keeps_macros_with_values_and_standards(tmp_path):
    kept = sanitize_flags(["-DO_EXEC=O_PATH", "-D", "X(a)=a", "-UY", "-std=c11", "-O2", "-Wall"], tmp_path, tmp_path,
                          "test", dropped := [])
    assert kept == ["-DO_EXEC=O_PATH", "-DX(a)=a", "-UY", "-std=c11"]
    assert {item["flag"] for item in dropped} == {"-O2", "-Wall"}


# --- inference ------------------------------------------------------------------------

def test_inferred_include_directories(tmp_path):
    root = make_project(tmp_path)
    (root / "components" / "fs" / "include").mkdir(parents=True)
    (root / "components" / "fs" / "include" / "fs.h").write_text('#include "kernel.h"\n')
    (root / "src" / "main.c").write_text('#include "fs.h"\n#include <kernel.h>\nint main(void) { return 0; }\n')
    context = load_build_context(root, {}, ["src/main.c"])
    assert context.source == "inferred"
    flags = context.flags_for("src/main.c")
    assert ["-I", str(root / "components" / "fs" / "include")] == flags[flags.index(str(root / "components" / "fs" / "include")) - 1:][:2]
    assert str(root / "include") in flags


def test_inference_keeps_host_headers_authoritative(tmp_path, monkeypatch):
    """A project folder that copies host headers is searched after the system ones."""
    from code_agent import security_build_context as module
    system = tmp_path / "sysroot"
    system.mkdir()
    (system / "signal.h").write_text("/* host */\n")
    monkeypatch.setattr(module, "system_include_dirs", lambda: (str(system),))
    root = tmp_path / "project"
    (root / "libc" / "include").mkdir(parents=True)
    (root / "libc" / "include" / "signal.h").write_text("/* project copy */\n")
    (root / "libc" / "include" / "compat.h").write_text("/* only here */\n")
    (root / "main.c").write_text("#include <signal.h>\n#include <compat.h>\nint x;\n")
    context = load_build_context(root, {}, ["main.c"])
    flags = context.flags_for("main.c")
    assert flags == ["-idirafter", str(root / "libc" / "include")]


def test_describe_counts_distinct_directories_and_macros(tmp_path):
    root = make_project(tmp_path)
    (root / "lib").mkdir()
    write_database(root, [
        {"directory": ".", "file": "src/main.c", "arguments": ["cc", "-Iinclude", "-Ilib", "-DA", "-c", "src/main.c"]},
        {"directory": ".", "file": "src/b.c", "arguments": ["cc", "-Iinclude", "-DA", "-DB", "-c", "src/b.c"]},
    ])
    summary = load_build_context(root).describe()
    assert summary["include_dirs"] == 2 and summary["macros"] == 2 and summary["files_with_entries"] == 2
