import sys

from mypy.modulefinder import BuildSource
from mypy.stubgen import (
    StubSource,
    generate_asts_for_modules,
    mypy_options,
    Options,
)

from dependency_graph.utils.log import setup_logger

# Initialize logging
logger = setup_logger()


def generate_python_stub(code: str, include_docstrings: bool = False) -> str:
    """
    Generate a python stub from a python code string. This is a wrapper around mypy's stubgen
    If an error occurs, the original code is returned
    """
    try:
        from mypy.stubgen import ASTStubGenerator
    except ImportError:
        raise ImportError("mypy version is too old, generating stub is not supported")

    try:
        options = Options(
            pyversion=sys.version_info[:2],
            no_import=True,
            inspect=False,
            doc_dir="",
            search_path=[],
            interpreter=sys.executable,
            ignore_errors=True,
            parse_only=True,
            include_private=False,
            output_dir="/tmp/out",
            modules=[],
            packages=[],
            files=["/tmp/mock.py"],
            verbose=False,
            quiet=True,
            export_less=True,
            include_docstrings=include_docstrings,
        )
        mypy_opts = mypy_options(options)

        stub_source = StubSource("test", None)
        stub_source.source = BuildSource(None, None, text=code)
        generate_asts_for_modules([stub_source], False, mypy_opts, False)

        gen = ASTStubGenerator(
            stub_source.runtime_all,
            include_private=False,
            analyzed=True,
            export_less=False,
            include_docstrings=include_docstrings,
        )
        stub_source.ast.accept(gen)
        output = gen.output()
    except (SystemExit, Exception) as e:
        # mypy may raise SystemExit for `Critical error during semantic analysis`
        logger.error(f"Error generating stub: {e}, will return the original code")
        output = code
    return output
