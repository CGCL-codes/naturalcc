"""Bootstrap the bundled source tree as the ``code_agent`` Python package.

VS Code installs an extension under its extension identifier rather than the
repository's ``code_agent`` directory name.  Registering this directory as a
package preserves the project's existing absolute and relative imports without
copying the backend into a second directory.
"""

import sys
import types
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


# [maint-note:42d7a5e6b22d]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 返回契约：None。
# 写入目标：package.__path__。
# 写入目标：package.__file__。
# 直接调用：types.ModuleType。
# 直接调用：str。
# 直接调用：sys.modules.setdefault。
# 直接调用：run_web_app。
# [/maint-note:42d7a5e6b22d]
def main() -> None:
    package = types.ModuleType("code_agent")
    package.__path__ = [str(PACKAGE_DIR)]
    package.__file__ = str(PACKAGE_DIR / "__init__.py")
    sys.modules.setdefault("code_agent", package)

    from code_agent.agent_web_api import main as run_web_app

    run_web_app()


if __name__ == "__main__":
    main()
