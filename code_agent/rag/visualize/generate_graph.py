#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 RAG 项目生成知识图谱 JSON，用于后续可视化。

支持：
  - C/C++ 项目（调用 rag.c.preprocess.CProjectParser）
  - Java 项目（调用 rag.java.java_project_parser_ts.JavaProjectParserTS）

用法示例：
    python rag/visualize/generate_graph.py -d /path/to/c_project -l c -o c_project.json
    python rag/visualize/generate_graph.py -d /path/to/java_project -l java -o java_project.json

依赖：
  - C: 需要 libclang 和 clang Python 包（clang==18.1.8）
  - Java: 需要 tree-sitter 相关包（tree-sitter-java 或 tree-sitter-language-pack）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module_from_path(module_name: str, file_path: Path, package: str | None = None):
    """通过文件路径直接加载 Python 模块（绕过包导入限制）。

    Args:
        module_name: 注册到 sys.modules 的模块名。
        file_path: 模块文件路径。
        package: 若提供，则设置为 module.__package__，用于支持相对导入。
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    if package is not None:
        module.__package__ = package
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_c_project(proj_dir: str) -> dict:
    """解析 C/C++ 项目，返回知识图谱 dict。"""
    rag_c_dir = Path(__file__).resolve().parent.parent / "c"

    # 先把依赖模块加载到 __main__ 包下，供 preprocess 的相对导入使用
    _load_module_from_path(
        "__main__.utils",
        rag_c_dir / "utils.py",
        package="__main__",
    )
    _load_module_from_path(
        "__main__.cfile_parse",
        rag_c_dir / "cfile_parse.py",
        package="__main__",
    )
    _load_module_from_path(
        "__main__.node_prompt",
        rag_c_dir / "node_prompt.py",
        package="__main__",
    )

    # 现在加载 preprocess
    preprocess = _load_module_from_path(
        "__main__.preprocess",
        rag_c_dir / "preprocess.py",
        package="__main__",
    )
    parser = preprocess.CProjectParser()
    result = parser.parse_dir(proj_dir)
    return result


def parse_java_project(proj_dir: str) -> dict:
    """解析 Java 项目，返回知识图谱 dict。"""
    rag_java_dir = Path(__file__).resolve().parent.parent / "java"

    # 先把依赖模块加载到 __main__ 包下，供主解析器的相对导入使用
    _load_module_from_path(
        "__main__.javafile_parse_ts",
        rag_java_dir / "javafile_parse_ts.py",
        package="__main__",
    )
    _load_module_from_path(
        "__main__.node_prompt_java_ts",
        rag_java_dir / "node_prompt_java_ts.py",
        package="__main__",
    )

    # 加载主解析器
    java_parser = _load_module_from_path(
        "__main__.java_project_parser_ts",
        rag_java_dir / "java_project_parser_ts.py",
        package="__main__",
    )
    parser = java_parser.JavaProjectParserTS()
    result = parser.parse_dir(proj_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 C/C++ 或 Java 项目解析为知识图谱 JSON"
    )
    parser.add_argument(
        "-d", "--dir", required=True,
        help="项目根目录路径"
    )
    parser.add_argument(
        "-l", "--lang", required=True, choices=["c", "java"],
        help="项目语言：c 或 java"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="JSON 缩进空格数（默认 2，设为 0 可最小化）"
    )
    args = parser.parse_args()

    proj_dir = Path(args.dir).expanduser().resolve()
    if not proj_dir.is_dir():
        print(f"❌ 目录不存在: {proj_dir}")
        sys.exit(1)

    print(f"🔧 语言: {args.lang}")
    print(f"📁 项目目录: {proj_dir}")
    print(f"💾 输出文件: {args.output}")
    print("=" * 50)

    try:
        if args.lang == "c":
            data = parse_c_project(str(proj_dir))
        else:
            data = parse_java_project(str(proj_dir))
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if args.indent > 0:
            json.dump(data, f, indent=args.indent, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)

    # 简单统计
    total_modules = len(data)
    total_symbols = sum(
        len([k for k in file_info if k])
        for file_info in data.values()
    )
    print("=" * 50)
    print(f"✅ 解析完成！")
    print(f"   模块（文件）数: {total_modules}")
    print(f"   符号节点数: {total_symbols}")
    print(f"   输出: {output_path}")
    print(f"\n下一步可视化：")
    print(f"   python rag/visualize/visualize.py -i {output_path} -o graph.html")


if __name__ == "__main__":
    main()
