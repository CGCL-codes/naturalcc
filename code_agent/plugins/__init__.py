import importlib
import pkgutil
import sys
from pathlib import Path


def _import_all_plugins():
    for _, module_name, _ in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
        if module_name != __name__:
            importlib.import_module(module_name)


# When running standalone, ensure parent package is importable
if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

_import_all_plugins()
