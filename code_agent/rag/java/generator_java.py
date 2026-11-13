# -*- coding: utf-8 -*-

import os
import json
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from .tokenizer import CModelTokenizer
    from .utils import ONLY_DEF
except:
    from tokenizer import CModelTokenizer
    from utils import ONLY_DEF


STD_LIB_PREFIXES = (
    "java.", "javax.", "jdk.", "sun.", "org.w3c.", "org.xml.", "jakarta."
)

IMPORT_LINE_RE = re.compile(r"^\s*import\s+(static\s+)?([a-zA-Z0-9_.]+(\.\*)?)\s*;\s*$", re.M)


# [maint-note:5ae88aea2a92]
# 维护导航：_norm_path；下列为本函数源码中的接口与路径索引。
# 输入 p：str。
# 返回契约：str。
# 返回路径：p.replace('\\', '/')。
# 直接调用：p.replace。
# [/maint-note:5ae88aea2a92]
def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


# [maint-note:a7801098d0b4]
# 数据与行为导航：JavaGenerator。
# 继承接口：object。
# [/maint-note:a7801098d0b4]
class JavaGenerator(object):
    """
    Java prompt generator adapted to preprocess_java_ts graph.
    """
    # [maint-note:1bf34b2a0a8a]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 proj_dir：str。
    # 输入 info_dir：str。
    # 输入 model：str。
    # 写入目标：self.proj_dir。
    # 写入目标：self.info_dir。
    # 写入目标：self.tokenizer。
    # 写入目标：self.project。
    # 写入目标：self.proj_info。
    # 直接调用：os.path.abspath。
    # 直接调用：CModelTokenizer。
    # [/maint-note:1bf34b2a0a8a]
    def __init__(self, proj_dir: str, info_dir: str, model: str):
        self.proj_dir = os.path.abspath(proj_dir)   # DS_REPO_DIR
        self.info_dir = os.path.abspath(info_dir)   # DS_GRAPH_DIR
        self.tokenizer = CModelTokenizer(model)

        self.project: Optional[str] = None
        self.proj_info: Optional[Dict[str, Dict]] = None  # loaded graph json: {rel_file_path -> file_info}

    # ----------------------------
    # project loading / path utils
    # ----------------------------

    # [maint-note:9de1876c9e67]
    # 维护导航：_set_project；下列为本函数源码中的接口与路径索引。
    # 输入 project：str。
    # 分支条件：project == self.project。
    # 分支条件：not os.path.isfile(info_file)。
    # 返回路径：无返回值。
    # 写入目标：self.project。
    # 写入目标：self.proj_info。
    # 直接调用：os.path.join。
    # 直接调用：os.path.isfile。
    # 直接调用：print。
    # 直接调用：open。
    # 直接调用：json.load。
    # 资源作用域：open(info_file, 'r', encoding='utf-8')；退出该 with 时调用对应上下文清理。
    # [/maint-note:9de1876c9e67]
    def _set_project(self, project: str):
        if project == self.project:
            return

        info_file = os.path.join(self.info_dir, f"{project}.json")
        if not os.path.isfile(info_file):
            print(f"未知项目 {project} 在 {self.info_dir}")
            self.project = project
            self.proj_info = None
            return

        self.project = project
        with open(info_file, "r", encoding="utf-8") as f:
            self.proj_info = json.load(f)

    # [maint-note:0615a124f181]
    # 维护导航：_project_root；下列为本函数源码中的接口与路径索引。
    # 返回契约：str。
    # 返回路径：os.path.join(self.proj_dir, self.project) if self.project else self.proj_dir。
    # 直接调用：os.path.join。
    # 表达式分流：self.project；真值与假值分支分别计算，不同时执行。
    # [/maint-note:0615a124f181]
    def _project_root(self) -> str:
        # repo root directory for current project
        return os.path.join(self.proj_dir, self.project) if self.project else self.proj_dir

    # [maint-note:01a4f1da55bd]
    # 维护导航：_abs_to_rel_in_repo；下列为本函数源码中的接口与路径索引。
    # 输入 abs_fpath：str。
    # 返回契约：str。
    # 返回路径：_norm_path(rel)。
    # 直接调用：self._project_root。
    # 直接调用：os.path.relpath。
    # 直接调用：_norm_path。
    # [/maint-note:01a4f1da55bd]
    def _abs_to_rel_in_repo(self, abs_fpath: str) -> str:
        """
        把绝对路径转成 repo 内相对路径，去匹配 proj_info 的 key（preprocess_java_ts 输出就是 repo 内相对路径）。
        """
        root = self._project_root()
        rel = os.path.relpath(abs_fpath, root)
        return _norm_path(rel)

    # ----------------------------
    # prompt building helpers
    # ----------------------------

    # [maint-note:846084def4a8]
    # 维护导航：get_suffix；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：str。
    # 返回契约：str。
    # 分支条件：fpath.startswith(repo_root + '/')。
    # 返回路径：f'// path: {fpath}\n'。
    # 直接调用：fpath.replace。
    # 直接调用：self.proj_dir.replace('\\', '/').rstrip。
    # 直接调用：self.proj_dir.replace。
    # 直接调用：fpath.startswith。
    # 直接调用：len。
    # [/maint-note:846084def4a8]
    def get_suffix(self, fpath: str) -> str:
        """
        将绝对路径裁剪为 repo 内相对路径，避免 prompt 中出现机器相关前缀
        """
        fpath = fpath.replace("\\", "/")
        repo_root = self.proj_dir.replace("\\", "/").rstrip("/")

        if fpath.startswith(repo_root + "/"):
            fpath = fpath[len(repo_root) + 1:]

        return f"// path: {fpath}\n"


    # [maint-note:b7f38261da37]
    # 维护导航：_extract_imports_from_source；下列为本函数源码中的接口与路径索引。
    # 输入 source_code：str。
    # 返回契约：List[str]。
    # 分支条件：not fqn。
    # 分支条件：fqn.startswith(STD_LIB_PREFIXES)。
    # 遍历边界：m ← IMPORT_LINE_RE.finditer(source_code)。
    # 返回路径：imports。
    # 直接调用：IMPORT_LINE_RE.finditer。
    # 直接调用：m.group。
    # 直接调用：fqn.startswith。
    # 直接调用：imports.append。
    # [/maint-note:b7f38261da37]
    def _extract_imports_from_source(self, source_code: str) -> List[str]:
        """
        兜底：如果图里找不到 imports（或你想用样本片段里的 imports），就从 source_code 扫一遍。
        """
        imports: List[str] = []
        for m in IMPORT_LINE_RE.finditer(source_code):
            fqn = m.group(2)
            if not fqn:
                continue
            if fqn.startswith(STD_LIB_PREFIXES):
                continue
            imports.append(fqn)
        return imports

    # [maint-note:9c3d2dde28e9]
    # 维护导航：_get_file_info；下列为本函数源码中的接口与路径索引。
    # 输入 rel_path：str。
    # 返回契约：Optional[Dict]。
    # 分支条件：not self.proj_info。
    # 返回路径：None。
    # 返回路径：self.proj_info.get(rel_path)。
    # 直接调用：self.proj_info.get。
    # [/maint-note:9c3d2dde28e9]
    def _get_file_info(self, rel_path: str) -> Optional[Dict]:
        if not self.proj_info:
            return None
        return self.proj_info.get(rel_path)

    # [maint-note:555a1cf1e889]
    # 维护导航：_list_all_files_in_package；下列为本函数源码中的接口与路径索引。
    # 输入 pkg：str。
    # 返回契约：List[str]。
    # 分支条件：not self.proj_info。
    # 分支条件：mod.get('package') == pkg。
    # 遍历边界：(fpath, finfo) ← self.proj_info.items()。
    # 返回路径：[]。
    # 返回路径：sorted(set(out))。
    # 直接调用：self.proj_info.items。
    # 直接调用：finfo.get。
    # 直接调用：mod.get。
    # 直接调用：out.append。
    # 直接调用：sorted。
    # 直接调用：set。
    # [/maint-note:555a1cf1e889]
    def _list_all_files_in_package(self, pkg: str) -> List[str]:
        """
        给 wildcard import 用：找所有 module.package == pkg 的文件。
        """
        if not self.proj_info:
            return []
        out = []
        for fpath, finfo in self.proj_info.items():
            mod = finfo.get("", {})
            if mod.get("package") == pkg:
                out.append(fpath)
        return sorted(set(out))

    # [maint-note:9ada596ed25c]
    # 维护导航：_resolve_imports_via_graph；下列为本函数源码中的接口与路径索引。
    # 输入 cur_rel：str。
    # 返回契约：List[str]。
    # 分支条件：not finfo。
    # 分支条件：not isinstance(k, str) or not k.startswith('import:')。
    # 分支条件：isinstance(inc, list) and len(inc) >= 1 and inc[0]。
    # 分支条件：deps。
    # 分支条件：not isinstance(imp, str)。
    # 分支条件：imp.startswith(STD_LIB_PREFIXES)。
    # 分支条件：imp.endswith('.*')。
    # 分支条件：fp.endswith('/' + rel) or fp.endswith(rel)。
    # 遍历边界：(k, v) ← finfo.items()。
    # 遍历边界：imp ← imports。
    # 遍历边界：f ← self._list_all_files_in_package(pkg)。
    # 遍历边界：fpath ← self.proj_info.keys()。
    # 返回路径：[]。
    # 返回路径：sorted(deps)。
    # 直接调用：self._get_file_info。
    # 直接调用：set。
    # 直接调用：finfo.items。
    # 直接调用：isinstance。
    # 直接调用：k.startswith。
    # 直接调用：v.get。
    # 直接调用：len。
    # 直接调用：deps.add。
    # 直接调用：_norm_path。
    # 直接调用：sorted。
    # 直接调用：finfo.get。
    # 直接调用：mod.get。
    # 直接调用：imp.startswith。
    # 直接调用：imp.endswith。
    # 直接调用：self._list_all_files_in_package。
    # 直接调用：imp.replace。
    # 直接调用：self.proj_info.keys。
    # 直接调用：fp.endswith。
    # [/maint-note:9ada596ed25c]
    def _resolve_imports_via_graph(self, cur_rel: str) -> List[str]:
        """
        核心：适配 preprocess_java_ts 的 import:* 虚拟节点 / include 边。
        返回“依赖文件”的 repo 内相对路径列表。
        """
        finfo = self._get_file_info(cur_rel)
        if not finfo:
            return []

        deps: Set[str] = set()

        # 1) 优先：import:* 虚拟节点（preprocess_java_ts 已解析到项目内文件）
        for k, v in finfo.items():
            if not isinstance(k, str) or not k.startswith("import:"):
                continue
            inc = v.get("include")
            if isinstance(inc, list) and len(inc) >= 1 and inc[0]:
                deps.add(_norm_path(inc[0]))

        if deps:
            return sorted(deps)

        # 2) 回退：用 module.imports 自己解析（best-effort）
        mod = finfo.get("", {})
        imports = mod.get("imports") or []
        for imp in imports:
            if not isinstance(imp, str):
                continue
            if imp.startswith(STD_LIB_PREFIXES):
                continue
            if imp.endswith(".*"):
                pkg = imp[:-2]
                for f in self._list_all_files_in_package(pkg):
                    deps.add(f)
            else:
                # best-effort：找 suffix 匹配 /a/b/C.java
                rel = imp.replace(".", "/") + ".java"
                for fpath in self.proj_info.keys():
                    fp = _norm_path(fpath)
                    if fp.endswith("/" + rel) or fp.endswith(rel):
                        deps.add(fp)
                        break

        return sorted(deps)

    # [maint-note:9b15afa51891]
    # 维护导航：_collect_api_from_dep_file；下列为本函数源码中的接口与路径索引。
    # 输入 dep_rel：str。
    # 返回契约：str。
    # 分支条件：not finfo。
    # 分支条件：k == '' or not isinstance(k, str) or (not isinstance(v, dict))。
    # 分支条件：'.' in k or '#' in k。
    # 分支条件：v.get('type') in ('Class', 'Interface', 'Enum', 'Annotation', 'Record')。
    # 分支条件：type_nodes。
    # 分支条件：tdef。
    # 分支条件：not isinstance(k, str) or not isinstance(v, dict)。
    # 分支条件：k.startswith(dot_prefix) and v.get('type') in ('Field',)。
    # 分支条件：field_def。
    # 分支条件：k.startswith(hash_prefix) and v.get('type') in ('Method', 'Constructor')。
    # 分支条件：method_def。
    # 分支条件：sig。
    # 遍历边界：(k, v) ← finfo.items()。
    # 遍历边界：(_, tname, tinfo) ← type_nodes。
    # 遍历边界：(_, sig) ← mems。
    # 返回路径：''。
    # 返回路径：text + ('\n' if text else '')。
    # 直接调用：self._get_file_info。
    # 直接调用：finfo.items。
    # 直接调用：isinstance。
    # 直接调用：v.get。
    # 直接调用：type_nodes.append。
    # 直接调用：type_nodes.sort。
    # 直接调用：chunks.append。
    # 直接调用：(tinfo.get('def') or '').strip。
    # 直接调用：tinfo.get。
    # 直接调用：k.startswith。
    # 直接调用：self._format_field_declaration。
    # 直接调用：mems.append。
    # 直接调用：self._format_method_prompt。
    # 直接调用：mems.sort。
    # 直接调用：'\n'.join(chunks).strip。
    # 直接调用：'\n'.join。
    # 表达式分流：text；真值与假值分支分别计算，不同时执行。
    # [/maint-note:9b15afa51891]
    def _collect_api_from_dep_file(self, dep_rel: str) -> str:
        """
        从依赖文件抽取“API 级”信息（类型头 + 成员签名），用于 prompt。
        适配 preprocess_java_ts 默认 schema：
          - 顶层/内部类型 key: "Foo" or "Outer$Inner"
          - field key: "Owner.field"
          - method/ctor key: "Owner#method(...)" / "Owner#Owner(...)"
        """
        finfo = self._get_file_info(dep_rel)
        if not finfo:
            return ""

        # types: k no '.' and no '#' and k != ''
        type_nodes: List[Tuple[int, str, Dict]] = []
        for k, v in finfo.items():
            if k == "" or not isinstance(k, str) or not isinstance(v, dict):
                continue
            if "." in k or "#" in k:
                continue
            if v.get("type") in ("Class", "Interface", "Enum", "Annotation", "Record"):
                sline = v.get("sline", 0) or 0
                type_nodes.append((sline, k, v))

        type_nodes.sort(key=lambda x: x[0])

        chunks: List[str] = []
        if type_nodes:
            chunks.append(f"// {dep_rel}")

        for _, tname, tinfo in type_nodes:
            # type header
            tdef = (tinfo.get("def") or "").strip()
            if tdef:
                chunks.append(tdef)

            # members
            dot_prefix = tname + "."
            hash_prefix = tname + "#"

            mems: List[Tuple[int, str]] = []
            for k, v in finfo.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                if k.startswith(dot_prefix) and v.get("type") in ("Field",):
                    field_def = self._format_field_declaration(v.get("def") or "")
                    if field_def:
                        mems.append((v.get("sline", 0) or 0, field_def))
                elif k.startswith(hash_prefix) and v.get("type") in ("Method", "Constructor"):
                    method_def = self._format_method_prompt(v)
                    if method_def:
                        mems.append((v.get("sline", 0) or 0, method_def))

            mems.sort(key=lambda x: x[0])
            for _, sig in mems:
                if sig:
                    chunks.append("    " + sig)  # 添加缩进使格式更清晰

            chunks.append("")  # blank between types

        text = "\n".join(chunks).strip()
        return text + ("\n" if text else "")

    # [maint-note:73d39feec72e]
    # 维护导航：_format_field_declaration；下列为本函数源码中的接口与路径索引。
    # 输入 field_def：str。
    # 返回契约：str。
    # 分支条件：not field_def。
    # 分支条件：s.startswith('@') and '(' not in s and (' ' not in s)。
    # 分支条件：'=' in s。
    # 分支条件：not s.endswith(';')。
    # 返回路径：''。
    # 返回路径：s。
    # 直接调用：field_def.strip。
    # 直接调用：s.startswith。
    # 直接调用：s.split('=')[0].strip。
    # 直接调用：s.split。
    # 直接调用：s.endswith。
    # [/maint-note:73d39feec72e]
    def _format_field_declaration(self, field_def: str) -> str:
        """格式化字段声明，移除初始化值，只保留类型和名称"""
        if not field_def:
            return ""
        s = field_def.strip()
        # 移除注解行（如 @Nullable, @Override 等单独的注解）
        if s.startswith("@") and "(" not in s and " " not in s:
            return ""
        # 移除字段初始化值
        if "=" in s:
            s = s.split("=")[0].strip()
        if not s.endswith(";"):
            s += ";"
        return s

    # [maint-note:0485b3a417c9]
    # 维护导航：_format_method_declaration；下列为本函数源码中的接口与路径索引。
    # 输入 method_def：str。
    # 返回契约：str。
    # 分支条件：not method_def。
    # 分支条件：s.startswith('@') and s.endswith(';')。
    # 分支条件：s.startswith('@') and '(' not in s and (' ' not in s)。
    # 分支条件：s.endswith('{')。
    # 分支条件：not s.endswith(';')。
    # 返回路径：''。
    # 返回路径：s。
    # 直接调用：method_def.strip。
    # 直接调用：s.startswith。
    # 直接调用：s.endswith。
    # 直接调用：s[:-1].strip。
    # [/maint-note:0485b3a417c9]
    def _format_method_declaration(self, method_def: str) -> str:
        """格式化方法声明，转换为接口风格的签名"""
        if not method_def:
            return ""
        s = method_def.strip()
        # 移除单独的注解（如 @Override; @Nullable; 等错误格式）
        if s.startswith("@") and s.endswith(";"):
            return ""
        if s.startswith("@") and "(" not in s and " " not in s:
            return ""
        # 移除方法体开头的 { 并添加 ;
        if s.endswith("{"):
            s = s[:-1].strip() + ";"
        elif not s.endswith(";"):
            s += ";"
        return s

    # [maint-note:24d37a6fa951]
    # 维护导航：_format_method_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 node_info：Dict。
    # 返回契约：str。
    # 分支条件：not method_def。
    # 分支条件：ONLY_DEF。
    # 分支条件：body。
    # 返回路径：''。
    # 返回路径：self._format_method_declaration(method_def)。
    # 返回路径：f'{method_def} {body}' if not method_def.endswith('{') else f'{method_def}\n{body}'。
    # 直接调用：(node_info.get('def') or '').strip。
    # 直接调用：node_info.get。
    # 直接调用：self._format_method_declaration。
    # 直接调用：method_def.endswith。
    # 表达式分流：not method_def.endswith('{')；真值与假值分支分别计算，不同时执行。
    # [/maint-note:24d37a6fa951]
    def _format_method_prompt(self, node_info: Dict) -> str:
        method_def = (node_info.get("def") or "").strip()
        if not method_def:
            return ""

        if ONLY_DEF:
            return self._format_method_declaration(method_def)

        body = node_info.get("body") or ""
        if body:
            return f"{method_def} {body}" if not method_def.endswith("{") else f"{method_def}\n{body}"

        return self._format_method_declaration(method_def)

    # ----------------------------
    # public API
    # ----------------------------

    # [maint-note:805202c2855a]
    # 维护导航：retrieve_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 project：str。
    # 输入 abs_fpath：str。
    # 输入 source_code：str。
    # 分支条件：not self.proj_info。
    # 分支条件：not dep_files。
    # 分支条件：imp.endswith('.*')。
    # 分支条件：fp.endswith('/' + rel) or fp.endswith(rel)。
    # 分支条件：dep in seen。
    # 分支条件：api。
    # 分支条件：not prompt。
    # 分支条件：self.tokenizer.cal_token_nums(source_code) > half_length。
    # 分支条件：not self.tokenizer.judge_prompt(prompt, max_prompt_length)。
    # 分支条件：current_length + line_length <= max_prompt_length。
    # 遍历边界：imp ← imports。
    # 遍历边界：f ← self._list_all_files_in_package(imp[:-2])。
    # 遍历边界：fpath ← self.proj_info.keys()。
    # 遍历边界：dep ← dep_files。
    # 遍历边界：line ← lines。
    # 返回路径：(self.tokenizer.truncate_concat(source_code, '', suffix), False)。
    # 返回路径：(self.tokenizer.truncate_concat(source_code, prompt, suffix), True)。
    # 直接调用：self._set_project。
    # 直接调用：self.get_suffix。
    # 直接调用：self.tokenizer.truncate_concat。
    # 直接调用：self._abs_to_rel_in_repo。
    # 直接调用：self._resolve_imports_via_graph。
    # 直接调用：self._extract_imports_from_source。
    # 直接调用：set。
    # 直接调用：imp.endswith。
    # 直接调用：self._list_all_files_in_package。
    # 直接调用：deps.add。
    # 直接调用：imp.replace。
    # 直接调用：self.proj_info.keys。
    # 直接调用：_norm_path。
    # 直接调用：fp.endswith。
    # 直接调用：sorted。
    # 直接调用：seen.add。
    # 直接调用：self._collect_api_from_dep_file。
    # 直接调用：prompt_parts.append。
    # 直接调用：'\n'.join(prompt_parts).strip。
    # 直接调用：'\n'.join。
    # 直接调用：self.tokenizer.cal_prompt_max_length。
    # 直接调用：int。
    # 直接调用：self.tokenizer.cal_token_nums。
    # 直接调用：self.tokenizer.judge_prompt。
    # 直接调用：prompt.split。
    # 直接调用：truncated_lines.append。
    # [/maint-note:805202c2855a]
    def retrieve_prompt(self, project: str, abs_fpath: str, source_code: str):
        """
        返回: (final_prompt_text, has_ctx)
        has_ctx=True 表示检索到了依赖上下文并注入到 prompt 区域；False 表示 prompt 为空（只用 source+suffix）
        """
        self._set_project(project)
        suffix = self.get_suffix(abs_fpath)

        # 1) 没有图信息 => 一定没有上下文
        if not self.proj_info:
            return self.tokenizer.truncate_concat(source_code, "", suffix), False

        cur_rel = self._abs_to_rel_in_repo(abs_fpath)

        # 2) 依赖文件（图优先）
        dep_files = self._resolve_imports_via_graph(cur_rel)

        # 3) fallback：从 source_code 抽 import
        if not dep_files:
            imports = self._extract_imports_from_source(source_code)
            deps: Set[str] = set()
            for imp in imports:
                if imp.endswith(".*"):
                    for f in self._list_all_files_in_package(imp[:-2]):
                        deps.add(f)
                else:
                    rel = imp.replace(".", "/") + ".java"
                    for fpath in self.proj_info.keys():
                        fp = _norm_path(fpath)
                        if fp.endswith("/" + rel) or fp.endswith(rel):
                            deps.add(fp)
                            break
            dep_files = sorted(deps)

        # 4) 组 prompt：依赖 API
        prompt_parts: List[str] = []
        seen: Set[str] = set()
        for dep in dep_files:
            if dep in seen:
                continue
            seen.add(dep)
            api = self._collect_api_from_dep_file(dep)
            if api:
                prompt_parts.append(api)

        prompt = "\n".join(prompt_parts).strip()

        # 5) prompt 为空 => 没检索到上下文
        if not prompt:
            return self.tokenizer.truncate_concat(source_code, "", suffix), False

        # 6) 有上下文 => 后续做截断策略
        max_prompt_length = self.tokenizer.cal_prompt_max_length(source_code, suffix)

        half_length = int(0.5 * self.tokenizer.max_input_length)
        if self.tokenizer.cal_token_nums(source_code) > half_length:
            source_code = source_code[-half_length:]

        if not self.tokenizer.judge_prompt(prompt, max_prompt_length):
            lines = prompt.split("\n")
            truncated_lines = []
            current_length = 0
            for line in lines:
                # 模型无关的 token 计数（兼容 HF tokenizer / tiktoken）
                line_length = self.tokenizer.cal_token_nums(line)

                if current_length + line_length <= max_prompt_length:
                    truncated_lines.append(line)
                    current_length += line_length
                else:
                    break
            prompt = "\n".join(truncated_lines)

        return self.tokenizer.truncate_concat(source_code, prompt, suffix), True
