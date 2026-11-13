from pathlib import Path

from code_agent.scripts.measure_backend_comments import count_source, is_backend_path


def test_strings_and_docstrings_do_not_inflate_comment_count():
    source = '''# module note
"""Documentation with # text
and another # inside a string.
"""
prompt = "# system instructions"  # actual comment

value = 1
'''
    assert count_source(source) == {
        "physical_lines": 7, "nonblank_lines": 6, "comment_lines": 2,
    }


def test_scope_excludes_dependencies_datasets_tests_and_frontend():
    for name in ["agent_web_api.py", "agent_core/run_engine.py", "rag/c/cfile_parse.py"]:
        assert is_backend_path(Path(name))
    for name in ["test_api.py", "tests/test_agent_api.py", ".venv/lib/example.py",
                 "artifacts/contract5/project/example.py", "webui/src/App.jsx",
                 "scripts/build_contract5_dataset.py", "rag/vendor/example.py"]:
        assert not is_backend_path(Path(name))


def test_empty_and_inline_comments():
    assert count_source("") == {"physical_lines": 0, "nonblank_lines": 0, "comment_lines": 0}
    assert count_source("x = 1  # first # second\n") == {
        "physical_lines": 1, "nonblank_lines": 1, "comment_lines": 1,
    }
