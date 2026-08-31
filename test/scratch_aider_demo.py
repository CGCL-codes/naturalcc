# -*- coding: utf-8 -*-

def normalize_path_parts(path: str) -> list[str]:
    """
    将 Windows 或 Unix 风格路径统一切分成非空片段。

    同时处理 "\\" 和 "/" 分隔符，忽略空片段和 "."，
    保留普通目录名，不做真实文件系统访问。

    Args:
        path: 要处理的路径字符串

    Returns:
        非空路径片段列表
    """
    # 将反斜杠替换为正斜杠，统一分隔符
    normalized = path.replace("\\", "/")
    # 按 "/" 分割并过滤空片段和 "."
    parts = [part for part in normalized.split("/") if part and part != "."]
    return parts


if __name__ == "__main__":
    print(normalize_path_parts("a\\b//./c"))
