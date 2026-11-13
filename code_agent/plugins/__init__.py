import importlib
import pkgutil
import sys
from pathlib import Path


# [maint-note:4c76ce67e629]
# 维护导航：_import_all_plugins；下列为本函数源码中的接口与路径索引。
# 分支条件：module_name != __name__。
# 遍历边界：(_, module_name, _) ← pkgutil.iter_modules(__path__, prefix=__name__ + '.')。
# 直接调用：pkgutil.iter_modules。
# 直接调用：importlib.import_module。
# [/maint-note:4c76ce67e629]
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
