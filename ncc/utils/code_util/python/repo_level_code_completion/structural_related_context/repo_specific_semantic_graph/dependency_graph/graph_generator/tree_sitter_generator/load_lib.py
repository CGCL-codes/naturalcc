import platform
import sys
from pathlib import Path

TS_LIB_PATH = Path(__file__).parent.parent.parent / "lib"


def get_builtin_lib_path(parent_dir: Path = TS_LIB_PATH) -> Path:
    if sys.platform.startswith("linux"):
        lib_path = parent_dir / "languages-linux-x86_64.so"
    elif sys.platform == "darwin":
        machine = platform.machine()
        if machine == "x86_64":
            lib_path = parent_dir / "languages-darwin-x86_64.dylib"
        elif machine == "arm64":
            lib_path = parent_dir / "languages-darwin-arm64.dylib"
        else:
            raise RuntimeError("Unsupported Darwin platform: " + machine)
    else:
        raise RuntimeError("Unsupported platform: " + sys.platform)
    return lib_path
