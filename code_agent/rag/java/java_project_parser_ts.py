# -*- coding: utf-8 -*-
import os
import re
from typing import Dict, Set, List, Optional, Tuple

try:
    from .javafile_parse_ts import JavaTSParser
    from .node_prompt_java_ts import JavaProjectSearcherTS
except ImportError:
    from javafile_parse_ts import JavaTSParser
    from node_prompt_java_ts import JavaProjectSearcherTS


class JavaProjectParserTS:
    """
    Project-level Java parser (tree-sitter) with cross-file resolution.
    - Pass1: parse each file -> nodes
    - Pass2: resolve imports/includes + resolve type-like rels + rewrite Call/Access when possible
    """

    # [maint-note:fb391d814f4f]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 写入目标：self.parser。
    # 写入目标：self.searcher。
    # 写入目标：self.file_pattern。
    # 写入目标：self.java_pattern。
    # 写入目标：self.proj_dir。
    # 写入目标：self.parse_res。
    # 写入目标：self.EXPAND_WILDCARD_IMPORTS。
    # 写入目标：self.WILDCARD_IMPORT_MAX_MEMBERS。
    # 直接调用：JavaTSParser。
    # 直接调用：JavaProjectSearcherTS。
    # 直接调用：re.compile。
    # [/maint-note:fb391d814f4f]
    def __init__(self):
        self.parser = JavaTSParser()
        self.searcher = JavaProjectSearcherTS()

        self.file_pattern = re.compile(r"[^\w\-]")  # 与你原来一致：过滤目录名/文件名
        self.java_pattern = re.compile(r"\.java$")

        self.proj_dir = None
        self.parse_res: Dict[str, Dict] = {}

        # ---- knobs (可调整) ----
        # 对 import com.a.* 是否展开成多个 include 节点（可能节点爆炸）
        self.EXPAND_WILDCARD_IMPORTS = False
        # 展开时最多包含多少个文件
        self.WILDCARD_IMPORT_MAX_MEMBERS = 30

    # [maint-note:f9c9c5f17eaa]
    # 维护导航：set_proj_dir；下列为本函数源码中的接口与路径索引。
    # 输入 dir_path：str。
    # 写入目标：self.proj_dir。
    # 直接调用：dir_path.endswith。
    # 直接调用：print。
    # 表达式分流：dir_path.endswith(os.sep)；真值与假值分支分别计算，不同时执行。
    # [/maint-note:f9c9c5f17eaa]
    def set_proj_dir(self, dir_path: str):
        self.proj_dir = dir_path if dir_path.endswith(os.sep) else dir_path + os.sep
        print(f"\n📁 设置项目目录: {self.proj_dir}")

    # [maint-note:12a764f1bb5b]
    # 维护导航：_get_all_java_paths；下列为本函数源码中的接口与路径索引。
    # 输入 target_path：str。
    # 返回契约：Set[str]。
    # 分支条件：not os.path.isdir(target_path)。
    # 分支条件：item.startswith('.')。
    # 分支条件：os.path.isdir(fpath)。
    # 分支条件：re.search(self.file_pattern, item) is None。
    # 分支条件：os.path.isfile(fpath) and self.java_pattern.search(fpath)。
    # 分支条件：re.search(self.file_pattern, base) is None。
    # 遍历边界：stack。
    # 遍历边界：item ← os.listdir(d)。
    # 返回路径：set()。
    # 返回路径：java_files。
    # 直接调用：print。
    # 直接调用：os.path.isdir。
    # 直接调用：set。
    # 直接调用：stack.pop。
    # 直接调用：os.listdir。
    # 直接调用：item.startswith。
    # 直接调用：os.path.join。
    # 直接调用：re.search。
    # 直接调用：stack.append。
    # 直接调用：os.path.isfile。
    # 直接调用：self.java_pattern.search。
    # 直接调用：os.path.splitext。
    # 直接调用：java_files.add。
    # [/maint-note:12a764f1bb5b]
    def _get_all_java_paths(self, target_path: str) -> Set[str]:
        print(f"🔍 扫描 Java 文件: {target_path}")
        if not os.path.isdir(target_path):
            print("❌ 目录不存在！")
            return set()

        stack = [target_path]
        java_files: Set[str] = set()

        while stack:
            d = stack.pop()
            for item in os.listdir(d):
                if item.startswith("."):
                    continue
                fpath = os.path.join(d, item)

                if os.path.isdir(fpath):
                    if re.search(self.file_pattern, item) is None:
                        stack.append(fpath)
                else:
                    if os.path.isfile(fpath) and self.java_pattern.search(fpath):
                        base = os.path.splitext(item)[0]
                        if re.search(self.file_pattern, base) is None:
                            java_files.add(fpath)

        return java_files

    # [maint-note:a621a31bea60]
    # 维护导航：_module_name；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：str。
    # 返回契约：str。
    # 返回路径：fpath[len(self.proj_dir):]。
    # 直接调用：len。
    # [/maint-note:a621a31bea60]
    def _module_name(self, fpath: str) -> str:
        return fpath[len(self.proj_dir):]

    # [maint-note:755c426892df]
    # 维护导航：parse_dir；下列为本函数源码中的接口与路径索引。
    # 输入 proj_dir：str。
    # 返回契约：Dict[str, Dict]。
    # 分支条件：res。
    # 遍历边界：fpath ← java_files。
    # 返回路径：self.parse_res。
    # 捕获异常：Exception。
    # 写入目标：self.parse_res。
    # 写入目标：self.parse_res[module]。
    # 直接调用：self.set_proj_dir。
    # 直接调用：sorted。
    # 直接调用：self._get_all_java_paths。
    # 直接调用：print。
    # 直接调用：len。
    # 直接调用：self._module_name。
    # 直接调用：self.parser.parse_file。
    # 直接调用：self.searcher.set_proj。
    # 直接调用：self._resolve_project_types。
    # [/maint-note:755c426892df]
    def parse_dir(self, proj_dir: str) -> Dict[str, Dict]:
        self.set_proj_dir(proj_dir)
        java_files = sorted(self._get_all_java_paths(proj_dir))
        print(f"📄 共找到 {len(java_files)} 个 Java 源文件")

        self.parse_res = {}

        # Pass 1: parse all files
        for fpath in java_files:
            module = self._module_name(fpath)
            try:
                res = self.parser.parse_file(fpath).nodes
                if res:
                    self.parse_res[module] = res
            except Exception as e:
                print(f"     ❌ 解析失败: {module}: {e}")

        # Build project index
        self.searcher.set_proj(proj_dir, self.parse_res)

        # Pass 2: resolve types + imports + calls/access/new
        self._resolve_project_types()

        print(f"✅ 目录解析完成: {proj_dir}")
        print(f"📦 共解析 {len(self.parse_res)} 个文件节点\n")
        return self.parse_res

    # --------------------------
    # Pass 2: cross-file resolve
    # --------------------------

    # [maint-note:b323f5b8fa9f]
    # 维护导航：_resolve_project_types；下列为本函数源码中的接口与路径索引。
    # 分支条件：not rels。
    # 分支条件：not r or len(r) < 2。
    # 分支条件：rtype in ('Typeof', 'TypeofParam', 'TypeofLocal', 'Extend', 'Implement', 'New')。
    # 分支条件：fqcn。
    # 分支条件：rtype == 'Call'。
    # 分支条件：rewritten。
    # 分支条件：rtype == 'Access'。
    # 遍历边界：(module, file_info) ← self.parse_res.items()。
    # 遍历边界：(name, node) ← list(file_info.items())。
    # 遍历边界：r ← rels。
    # 写入目标：node['rels']。
    # 直接调用：print。
    # 直接调用：self.parse_res.items。
    # 直接调用：file_info.get。
    # 直接调用：mod.get。
    # 直接调用：self._inject_import_virtual_nodes。
    # 直接调用：self._inject_static_import_virtual_nodes。
    # 直接调用：list。
    # 直接调用：file_info.items。
    # 直接调用：node.get。
    # 直接调用：len。
    # 直接调用：self.searcher.resolve_type。
    # 直接调用：new_rels.append。
    # 直接调用：self.searcher.resolve_call。
    # 直接调用：self.searcher.resolve_access。
    # 直接调用：self._dedup_rels。
    # [/maint-note:b323f5b8fa9f]
    def _resolve_project_types(self):
        print("🔧 跨文件类型解析 & 清洗关系中 ...")

        for module, file_info in self.parse_res.items():
            mod = file_info.get("", {})
            imports = mod.get("imports", []) or []
            static_imports = mod.get("static_imports", []) or []

            # 1) imports -> include-like virtual nodes (for downstream DFS)
            self._inject_import_virtual_nodes(module, file_info, imports)
            self._inject_static_import_virtual_nodes(file_info, static_imports)

            # 2) resolve rels in every node
            for name, node in list(file_info.items()):
                rels = node.get("rels")
                if not rels:
                    continue

                new_rels: List[List[str]] = []
                for r in rels:
                    if not r or len(r) < 2:
                        continue
                    tgt, rtype = r[0], r[1]

                    # ---- type-like rels: rewrite to FQCN when possible ----
                    if rtype in ("Typeof", "TypeofParam", "TypeofLocal", "Extend", "Implement", "New"):
                        fqcn = self.searcher.resolve_type(module, tgt)
                        if fqcn:
                            new_rels.append([fqcn, rtype])
                        else:
                            new_rels.append([tgt, rtype])
                        continue

                    # ---- call rels: try rewrite receiver to FQCN#method ----
                    if rtype == "Call":
                        rewritten = self.searcher.resolve_call(module, tgt)
                        if rewritten:
                            new_rels.append([rewritten, rtype])
                        else:
                            new_rels.append([tgt, rtype])
                        continue

                    # ---- field access: try rewrite receiver to FQCN.field ----
                    if rtype == "Access":
                        rewritten = self.searcher.resolve_access(module, tgt)
                        if rewritten:
                            new_rels.append([rewritten, rtype])
                        else:
                            new_rels.append([tgt, rtype])
                        continue

                    # ---- keep others: Use / Assign / etc. ----
                    new_rels.append([tgt, rtype])

                # dedup after rewrite
                node["rels"] = self._dedup_rels(new_rels)

    # [maint-note:dcb8a94bb344]
    # 维护导航：_inject_import_virtual_nodes；下列为本函数源码中的接口与路径索引。
    # 输入 module：str。
    # 输入 file_info：Dict。
    # 输入 imports：List[str]。
    # 分支条件：target。
    # 分支条件：self.EXPAND_WILDCARD_IMPORTS and imp.endswith('.*')。
    # 分支条件：not members。
    # 遍历边界：imp ← imports。
    # 遍历边界：(fqcn, fpath) ← members。
    # 写入目标：file_info[f'import:{imp}']。
    # 写入目标：file_info[key]。
    # 直接调用：self.searcher.resolve_import_to_file。
    # 直接调用：imp.endswith。
    # 直接调用：self.searcher.package_members。
    # [/maint-note:dcb8a94bb344]
    def _inject_import_virtual_nodes(self, module: str, file_info: Dict, imports: List[str]):
        """
        - Default: each import generates one include-like node if resolvable
        - Optionally expand wildcard imports to multiple include nodes
        """
        for imp in imports:
            target = self.searcher.resolve_import_to_file(module, imp)
            if target:
                file_info[f"import:{imp}"] = {
                    "type": "Variable",
                    "def": f"import {imp};",
                    "sline": -1,
                    "include": [target, None],
                }

            # Optional expansion for wildcard imports
            if self.EXPAND_WILDCARD_IMPORTS and imp.endswith(".*"):
                pkg_prefix = imp[:-2]
                members = self.searcher.package_members(pkg_prefix)

                if not members:
                    continue

                # avoid explosion
                members = members[: self.WILDCARD_IMPORT_MAX_MEMBERS]
                for fqcn, fpath in members:
                    key = f"import:{imp}:{fqcn}"
                    file_info[key] = {
                        "type": "Variable",
                        "def": f"import {imp}; // -> {fqcn}",
                        "sline": -1,
                        "include": [fpath, None],
                    }

    # [maint-note:75e78d71df6c]
    # 维护导航：_inject_static_import_virtual_nodes；下列为本函数源码中的接口与路径索引。
    # 输入 file_info：Dict。
    # 输入 static_imports：List[str]。
    # 分支条件：not target。
    # 遍历边界：imp ← static_imports。
    # 写入目标：file_info[f'static_import:{imp}']。
    # 直接调用：self.searcher.resolve_static_import_to_file。
    # [/maint-note:75e78d71df6c]
    def _inject_static_import_virtual_nodes(self, file_info: Dict, static_imports: List[str]):
        for imp in static_imports:
            target = self.searcher.resolve_static_import_to_file(imp)
            if not target:
                continue
            file_info[f"static_import:{imp}"] = {
                "type": "Variable",
                "def": f"import static {imp};",
                "sline": -1,
                "include": [target, None],
            }

    # --------------------------
    # utils
    # --------------------------

    # [maint-note:029a11e4ca03]
    # 维护导航：_dedup_rels；下列为本函数源码中的接口与路径索引。
    # 输入 rels：List[List[str]]。
    # 返回契约：List[List[str]]。
    # 分支条件：not r or len(r) < 2。
    # 分支条件：key in seen。
    # 遍历边界：r ← rels。
    # 返回路径：out。
    # 直接调用：set。
    # 直接调用：len。
    # 直接调用：seen.add。
    # 直接调用：out.append。
    # [/maint-note:029a11e4ca03]
    def _dedup_rels(self, rels: List[List[str]]) -> List[List[str]]:
        seen = set()
        out: List[List[str]] = []
        for r in rels:
            if not r or len(r) < 2:
                continue
            tgt, rtype = r[0], r[1]
            key = (tgt, rtype)
            if key in seen:
                continue
            seen.add(key)
            out.append([tgt, rtype])
        return out
