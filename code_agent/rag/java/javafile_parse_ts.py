import os
import re
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

# -----------------------
# Helpers
# -----------------------

_PRIMITIVES = {
    "byte","short","int","long","float","double","boolean","char","void"
}

_GENERIC_CLEAN = re.compile(r"<.*?>")  # 非严格，但够用：把 List<Foo> -> List
_ARR_CLEAN = re.compile(r"\[\]")

# [maint-note:8a2719a5b4eb]
# 维护导航：_text；下列为本函数源码中的接口与路径索引。
# 输入 src：bytes。
# 输入 node：类型由调用方与函数体约束。
# 返回契约：str。
# 返回路径：src[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')。
# 直接调用：src[node.start_byte:node.end_byte].decode。
# [/maint-note:8a2719a5b4eb]
def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


# [maint-note:5348126dea98]
# 维护导航：_text_span；下列为本函数源码中的接口与路径索引。
# 输入 src：bytes。
# 输入 start：int。
# 输入 end：int。
# 返回契约：str。
# 返回路径：src[start:end].decode('utf-8', errors='ignore')。
# 直接调用：src[start:end].decode。
# [/maint-note:5348126dea98]
def _text_span(src: bytes, start: int, end: int) -> str:
    return src[start:end].decode("utf-8", errors="ignore")

# [maint-note:4f57ac1ef2e3]
# 维护导航：_node_line；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 返回契约：int。
# 返回路径：int(node.start_point[0]) + 1。
# 直接调用：int。
# [/maint-note:4f57ac1ef2e3]
def _node_line(node) -> int:
    # tree-sitter point 是 0-based
    return int(node.start_point[0]) + 1

# [maint-note:9739a690f37a]
# 维护导航：_strip_type；下列为本函数源码中的接口与路径索引。
# 输入 type_str：str。
# 返回契约：str。
# 返回路径：t。
# 直接调用：type_str.strip。
# 直接调用：_GENERIC_CLEAN.sub。
# 直接调用：t.replace。
# 直接调用：' '.join。
# 直接调用：t.split。
# [/maint-note:9739a690f37a]
def _strip_type(type_str: str) -> str:
    t = type_str.strip()
    t = _GENERIC_CLEAN.sub("", t)
    t = t.replace("...", "[]")  # varargs
    t = t.replace("@", " ")     # 粗略去注解
    t = " ".join(t.split())
    return t

# [maint-note:74d8ae619e07]
# 维护导航：_simple_name；下列为本函数源码中的接口与路径索引。
# 输入 type_str：str。
# 返回契约：str。
# 返回路径：t.split('.')[-1].strip()。
# 直接调用：_strip_type。
# 直接调用：_ARR_CLEAN.sub('', t).strip。
# 直接调用：_ARR_CLEAN.sub。
# 直接调用：t.split('.')[-1].strip。
# 直接调用：t.split。
# [/maint-note:74d8ae619e07]
def _simple_name(type_str: str) -> str:
    t = _strip_type(type_str)
    # 去数组
    t = _ARR_CLEAN.sub("", t).strip()
    # 取最后一段
    return t.split(".")[-1].strip()


# [maint-note:a94914a83443]
# 维护导航：_decl_text_before_body；下列为本函数源码中的接口与路径索引。
# 输入 src：bytes。
# 输入 node：类型由调用方与函数体约束。
# 输入 body_types：类型由调用方与函数体约束。
# 返回契约：str。
# 分支条件：c.type in body_types。
# 遍历边界：c ← node.children。
# 返回路径：_text_span(src, node.start_byte, c.start_byte).rstrip()。
# 返回路径：_text(src, node).strip()。
# 直接调用：_text_span(src, node.start_byte, c.start_byte).rstrip。
# 直接调用：_text_span。
# 直接调用：_text(src, node).strip。
# 直接调用：_text。
# [/maint-note:a94914a83443]
def _decl_text_before_body(src: bytes, node, body_types) -> str:
    for c in node.children:
        if c.type in body_types:
            return _text_span(src, node.start_byte, c.start_byte).rstrip()
    return _text(src, node).strip()

# [maint-note:67e52b9a11c0]
# 数据与行为导航：JavaFileParseResult。
# 字段 module：Dict。
# 字段 nodes：Dict[str, Dict]。
# 字段 package：Optional[str]。
# 字段 imports：List[str]。
# 字段 top_types：List[str]。
# 字段 inner_types：Dict[str, Set[str]]。
# [/maint-note:67e52b9a11c0]
@dataclass
class JavaFileParseResult:
    module: Dict
    nodes: Dict[str, Dict]          # name -> node_info
    package: Optional[str]
    imports: List[str]              # e.g. com.a.Foo, com.b.*
    top_types: List[str]            # simple names declared in file (top-level)
    inner_types: Dict[str, Set[str]]# outer_simple -> {inner_simple}

class JavaTSParser:
    """
    Parse a single .java file using tree-sitter-java, producing a node dict similar to your CParser.
    This stage is "syntax parse", not cross-file resolution.
    """
    # [maint-note:336616e8e457]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 写入目标：self.parser。
    # 直接调用：self._build_parser。
    # [/maint-note:336616e8e457]
    def __init__(self):
        self.parser = self._build_parser()

    # [maint-note:7bb24bc97445]
    # 维护导航：_build_parser；下列为本函数源码中的接口与路径索引。
    # 分支条件：hasattr(parser, 'language')。
    # 返回路径：tsl.get_parser('java')。
    # 返回路径：parser。
    # 返回路径：tslp.get_parser('java')。
    # 显式抛出：RuntimeError(f'无法初始化 Java tree-sitter parser。\n可选修复方式:\n1. 安装兼容后端: pip install tree-sitter tree-sitter-java\n2. 或使用新版语言包: pip install tree-sitter-language-pack\n3. 或回退兼容版本的 tree-sitter / tree-sitter-languages 组合\n详细错误:\n{details}')。
    # 捕获异常：Exception。
    # 写入目标：parser.language。
    # 直接调用：importlib.import_module。
    # 直接调用：tsl.get_parser。
    # 直接调用：errors.append。
    # 直接调用：tree_sitter.Language。
    # 直接调用：ts_java.language。
    # 直接调用：tree_sitter.Parser。
    # 直接调用：hasattr。
    # 直接调用：parser.set_language。
    # 直接调用：tslp.get_parser。
    # 直接调用：'\n'.join。
    # 直接调用：RuntimeError。
    # [/maint-note:7bb24bc97445]
    def _build_parser(self):
        errors = []

        try:
            tsl = importlib.import_module("tree_sitter_languages")
            return tsl.get_parser("java")
        except Exception as exc:
            errors.append(f"tree_sitter_languages.get_parser('java') 失败: {exc!r}")

        try:
            tree_sitter = importlib.import_module("tree_sitter")
            ts_java = importlib.import_module("tree_sitter_java")

            language = tree_sitter.Language(ts_java.language())
            parser = tree_sitter.Parser()
            if hasattr(parser, "language"):
                parser.language = language
            else:
                parser.set_language(language)
            return parser
        except Exception as exc:
            errors.append(f"tree_sitter + tree_sitter_java 直连模式失败: {exc!r}")

        try:
            tslp = importlib.import_module("tree_sitter_language_pack")
            return tslp.get_parser("java")
        except Exception as exc:
            errors.append(f"tree_sitter_language_pack.get_parser('java') 失败: {exc!r}")

        details = "\n".join(errors)
        raise RuntimeError(
            "无法初始化 Java tree-sitter parser。\n"
            "可选修复方式:\n"
            "1. 安装兼容后端: pip install tree-sitter tree-sitter-java\n"
            "2. 或使用新版语言包: pip install tree-sitter-language-pack\n"
            "3. 或回退兼容版本的 tree-sitter / tree-sitter-languages 组合\n"
            f"详细错误:\n{details}"
        )

    # [maint-note:ed08c87afc5c]
    # 维护导航：parse_file；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：str。
    # 返回契约：JavaFileParseResult。
    # 分支条件：child.type == 'package_declaration'。
    # 分支条件：child.type == 'import_declaration'。
    # 分支条件：imp.startswith('static ')。
    # 分支条件：child.type in ('class_declaration', 'interface_declaration', 'enum_declaration', 'annotation_type_declaration', 'record_declaration')。
    # 遍历边界：child ← root.children。
    # 返回路径：JavaFileParseResult(module=module_node, nodes=nodes, package=package, imports=imports, top_types=top_types, inner_types=inner_types)。
    # 写入目标：module_node['package']。
    # 写入目标：module_node['imports']。
    # 写入目标：nodes['']。
    # 直接调用：open。
    # 直接调用：f.read。
    # 直接调用：self.parser.parse。
    # 直接调用：os.path.abspath。
    # 直接调用：_text。
    # 直接调用：pkg.replace('package', '').replace(';', '').strip。
    # 直接调用：pkg.replace('package', '').replace。
    # 直接调用：pkg.replace。
    # 直接调用：imp.strip().rstrip。
    # 直接调用：imp.strip。
    # 直接调用：imp.replace('import', '', 1).strip。
    # 直接调用：imp.replace。
    # 直接调用：imp.startswith。
    # 直接调用：imp[len('static '):].strip。
    # 直接调用：len。
    # 直接调用：module_node.setdefault('static_imports', []).append。
    # 直接调用：module_node.setdefault。
    # 直接调用：imports.append。
    # 直接调用：self._parse_type_decl。
    # 直接调用：JavaFileParseResult。
    # 资源作用域：open(fpath, 'rb')；退出该 with 时调用对应上下文清理。
    # [/maint-note:ed08c87afc5c]
    def parse_file(self, fpath: str) -> JavaFileParseResult:
        with open(fpath, "rb") as f:
            src = f.read()

        tree = self.parser.parse(src)
        root = tree.root_node

        package = None
        imports: List[str] = []
        nodes: Dict[str, Dict] = {}

        # Module node
        module_node = {
            "type": "Module",
            "file_path": os.path.abspath(fpath),
        }

        # Collect package/imports and type declarations
        top_types: List[str] = []
        inner_types: Dict[str, Set[str]] = {}

        for child in root.children:
            if child.type == "package_declaration":
                # package_declaration: 'package' scoped_identifier ';'
                pkg = _text(src, child)
                # extract identifier text
                # simplest: remove keywords/; and spaces
                pkg = pkg.replace("package", "").replace(";", "").strip()
                package = pkg
                module_node["package"] = package

            elif child.type == "import_declaration":
                imp = _text(src, child)
                # examples:
                # import com.a.Foo;
                # import com.a.*;
                # import static com.a.Foo.bar;
                imp = imp.strip().rstrip(";")
                imp = imp.replace("import", "", 1).strip()
                # ignore static for type resolution baseline (can add later)
                if imp.startswith("static "):
                    imp = imp[len("static "):].strip()
                    # static import can be a.b.C.m or a.b.C.*
                    # keep it separately if you want
                    module_node.setdefault("static_imports", []).append(imp)
                    continue
                imports.append(imp)
            elif child.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "annotation_type_declaration",
                "record_declaration",
            ):
                self._parse_type_decl(src, child, nodes, top_types, inner_types, outer=None)

        module_node["imports"] = imports

        # add module itself into nodes under "" like your existing schema
        nodes[""] = module_node

        return JavaFileParseResult(
            module=module_node,
            nodes=nodes,
            package=package,
            imports=imports,
            top_types=top_types,
            inner_types=inner_types
        )

    # [maint-note:32bc3ad250b2]
    # 维护导航：_parse_type_decl；下列为本函数源码中的接口与路径索引。
    # 输入 src：bytes。
    # 输入 node：类型由调用方与函数体约束。
    # 输入 nodes：Dict[str, Dict]。
    # 输入 top_types：List[str]。
    # 输入 inner_types：Dict[str, Set[str]]。
    # 输入 outer：Optional[str]。
    # 分支条件：c.type == 'identifier'。
    # 分支条件：not name_node。
    # 分支条件：not outer。
    # 分支条件：c.type in ('class_body', 'interface_body', 'enum_body', 'annotation_type_body')。
    # 分支条件：c.type == 'superclass'。
    # 分支条件：t。
    # 分支条件：c.type == 'super_interfaces'。
    # 分支条件：rels。
    # 分支条件：not type_body。
    # 分支条件：m.type == 'field_declaration'。
    # 分支条件：m.type == 'method_declaration'。
    # 分支条件：m.type == 'constructor_declaration'。
    # 分支条件：m.type in ('class_declaration', 'interface_declaration', 'enum_declaration', 'annotation_type_declaration', 'record_declaration')。
    # 遍历边界：c ← node.children。
    # 遍历边界：p ← parts。
    # 遍历边界：m ← type_body.children。
    # 返回路径：无返回值。
    # 写入目标：entry['rels']。
    # 写入目标：nodes[qname]。
    # 直接调用：_text。
    # 直接调用：_node_line。
    # 直接调用：kind_map.get。
    # 直接调用：top_types.append。
    # 直接调用：inner_types.setdefault(outer, set()).add。
    # 直接调用：inner_types.setdefault。
    # 直接调用：set。
    # 直接调用：_decl_text_before_body。
    # 直接调用：_strip_type。
    # 直接调用：_text(src, c).replace('extends', '').strip。
    # 直接调用：_text(src, c).replace。
    # 直接调用：rels.append。
    # 直接调用：txt.replace('implements', '').strip。
    # 直接调用：txt.replace。
    # 直接调用：p.strip。
    # 直接调用：txt.split。
    # 直接调用：self._parse_field。
    # 直接调用：self._parse_method。
    # 直接调用：self._parse_ctor。
    # 直接调用：self._parse_type_decl。
    # 表达式分流：outer；真值与假值分支分别计算，不同时执行。
    # 集合派生：p 来自 txt.split(',')；过滤 p.strip()。
    # [/maint-note:32bc3ad250b2]
    def _parse_type_decl(
        self,
        src: bytes,
        node,
        nodes: Dict[str, Dict],
        top_types: List[str],
        inner_types: Dict[str, Set[str]],
        outer: Optional[str],
    ):
        # Find name
        name_node = None
        for c in node.children:
            if c.type == "identifier":
                name_node = c
                break
        if not name_node:
            return

        simple = _text(src, name_node)
        sline = _node_line(node)

        # Determine kind
        kind_map = {
            "class_declaration": "Class",
            "interface_declaration": "Interface",
            "enum_declaration": "Enum",
            "annotation_type_declaration": "Annotation",
            "record_declaration": "Record",
        }
        ntype = kind_map.get(node.type, "Class")

        # Unique name in graph:
        # top-level: SimpleName
        # inner: Outer$Inner (like JVM)
        qname = f"{outer}${simple}" if outer else simple

        # store top-level types
        if not outer:
            top_types.append(simple)
        else:
            inner_types.setdefault(outer, set()).add(simple)

        type_body = None
        for c in node.children:
            if c.type in ("class_body", "interface_body", "enum_body", "annotation_type_body"):
                type_body = c
                break

        type_def = _decl_text_before_body(
            src,
            node,
            ("class_body", "interface_body", "enum_body", "annotation_type_body"),
        )

        entry = {
            "type": ntype,
            "def": type_def,
            "sline": sline,
        }

        # extends/implements (syntax-level)
        rels = []
        for c in node.children:
            if c.type == "superclass":
                # superclass -> type_identifier / scoped_type_identifier
                t = _strip_type(_text(src, c).replace("extends", "").strip())
                if t:
                    rels.append([t, "Extend"])
            elif c.type == "super_interfaces":
                # super_interfaces includes implements + interface list
                txt = _text(src, c)
                txt = txt.replace("implements", "").strip()
                parts = [p.strip() for p in txt.split(",") if p.strip()]
                for p in parts:
                    rels.append([_strip_type(p), "Implement"])

        if rels:
            entry["rels"] = rels

        nodes[qname] = entry

        # parse members + nested types
        if not type_body:
            return

        for m in type_body.children:
            if m.type == "field_declaration":
                self._parse_field(src, m, nodes, owner=qname)
            elif m.type == "method_declaration":
                self._parse_method(src, m, nodes, owner=qname)
            elif m.type == "constructor_declaration":
                self._parse_ctor(src, m, nodes, owner=qname)
            elif m.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "annotation_type_declaration",
                "record_declaration",
            ):
                self._parse_type_decl(src, m, nodes, top_types, inner_types, outer=qname)

    # [maint-note:e244f15bfb9f]
    # 维护导航：_parse_field；下列为本函数源码中的接口与路径索引。
    # 输入 src：bytes。
    # 输入 node：类型由调用方与函数体约束。
    # 输入 nodes：Dict[str, Dict]。
    # 输入 owner：str。
    # 分支条件：c.type in ('type', 'integral_type', 'floating_point_type', 'boolean_type', 'void_type')。
    # 分支条件：c.type == 'variable_declarator'。
    # 分支条件：not declarators。
    # 分支条件：cc.type == 'identifier'。
    # 分支条件：not name_node。
    # 分支条件：t and t not in _PRIMITIVES。
    # 遍历边界：c ← node.children。
    # 遍历边界：d ← declarators。
    # 遍历边界：cc ← d.children。
    # 返回路径：无返回值。
    # 写入目标：entry['rels']。
    # 写入目标：nodes[key]。
    # 直接调用：_text。
    # 直接调用：declarators.append。
    # 直接调用：_strip_type。
    # 直接调用：_node_line。
    # 直接调用：_text(src, node).splitlines()[0].strip。
    # 直接调用：_text(src, node).splitlines。
    # 表达式分流：type_str；真值与假值分支分别计算，不同时执行。
    # [/maint-note:e244f15bfb9f]
    def _parse_field(self, src: bytes, node, nodes: Dict[str, Dict], owner: str):
        # field_declaration: modifiers? type variable_declarator (',' variable_declarator)* ';'
        # We'll grab type as first child with type == "type" or one of type nodes.
        type_str = None
        declarators = []

        for c in node.children:
            if c.type in ("type", "integral_type", "floating_point_type", "boolean_type", "void_type"):
                type_str = _text(src, c)
            if c.type == "variable_declarator":
                declarators.append(c)

        if not declarators:
            return

        t = _strip_type(type_str) if type_str else None
        for d in declarators:
            name_node = None
            for cc in d.children:
                if cc.type == "identifier":
                    name_node = cc
                    break
            if not name_node:
                continue
            fname = _text(src, name_node)
            sline = _node_line(d)
            key = f"{owner}.{fname}"
            entry = {
                "type": "Field",
                "def": _text(src, node).splitlines()[0].strip(),
                "sline": sline,
                "in_class": owner,
            }
            if t and t not in _PRIMITIVES:
                entry["rels"] = [[t, "Typeof"]]
            nodes[key] = entry

    # [maint-note:4c4f21cdb065]
    # 维护导航：_parse_method；下列为本函数源码中的接口与路径索引。
    # 输入 src：bytes。
    # 输入 node：类型由调用方与函数体约束。
    # 输入 nodes：Dict[str, Dict]。
    # 输入 owner：str。
    # 分支条件：c.type == 'identifier' and name is None。
    # 分支条件：c.type in ('type', 'integral_type', 'floating_point_type', 'boolean_type', 'void_type') and ret_type is None。
    # 分支条件：not name。
    # 分支条件：c.type == 'block'。
    # 分支条件：rt and rt not in _PRIMITIVES and (rt != 'void')。
    # 分支条件：rels。
    # 遍历边界：c ← node.children。
    # 返回路径：无返回值。
    # 写入目标：entry['body']。
    # 写入目标：entry['rels']。
    # 写入目标：nodes[sig]。
    # 直接调用：_text。
    # 直接调用：self._parse_params。
    # 直接调用：', '.join。
    # 直接调用：_node_line。
    # 直接调用：_decl_text_before_body。
    # 直接调用：_strip_type。
    # 直接调用：rels.append。
    # 表达式分流：ret_type；真值与假值分支分别计算，不同时执行。
    # [/maint-note:4c4f21cdb065]
    def _parse_method(self, src: bytes, node, nodes: Dict[str, Dict], owner: str):
        # method_declaration: modifiers? type_parameters? type? identifier formal_parameters ...
        name = None
        ret_type = None

        for c in node.children:
            if c.type == "identifier" and name is None:
                name = _text(src, c)
            if c.type in ("type", "integral_type", "floating_point_type", "boolean_type", "void_type") and ret_type is None:
                ret_type = _text(src, c)

        if not name:
            return

        params = self._parse_params(src, node)
        sig = f"{owner}#{name}({', '.join(params)})"
        sline = _node_line(node)

        method_def = _decl_text_before_body(src, node, ("block",))
        entry = {
            "type": "Method",
            "def": method_def,
            "sline": sline,
            "in_class": owner,
        }
        for c in node.children:
            if c.type == "block":
                entry["body"] = _text(src, c)
                break
        rels = []
        rt = _strip_type(ret_type) if ret_type else None
        if rt and rt not in _PRIMITIVES and rt != "void":
            rels.append([rt, "Typeof"])
        # You can add "Use" edges by walking method body, but type parsing is priority
        if rels:
            entry["rels"] = rels
        nodes[sig] = entry

    # [maint-note:cbd058bd03e8]
    # 维护导航：_parse_ctor；下列为本函数源码中的接口与路径索引。
    # 输入 src：bytes。
    # 输入 node：类型由调用方与函数体约束。
    # 输入 nodes：Dict[str, Dict]。
    # 输入 owner：str。
    # 分支条件：c.type == 'identifier'。
    # 分支条件：c.type in ('constructor_body', 'block')。
    # 遍历边界：c ← node.children。
    # 写入目标：entry['body']。
    # 写入目标：nodes[sig]。
    # 直接调用：_text。
    # 直接调用：self._parse_params。
    # 直接调用：', '.join。
    # 直接调用：_node_line。
    # 直接调用：_decl_text_before_body。
    # [/maint-note:cbd058bd03e8]
    def _parse_ctor(self, src: bytes, node, nodes: Dict[str, Dict], owner: str):
        # constructor_declaration has identifier = class name
        name = None
        for c in node.children:
            if c.type == "identifier":
                name = _text(src, c)
                break
        params = self._parse_params(src, node)
        sig = f"{owner}#{name}({', '.join(params)})"
        sline = _node_line(node)
        ctor_def = _decl_text_before_body(src, node, ("constructor_body", "block"))
        entry = {
            "type": "Constructor",
            "def": ctor_def,
            "sline": sline,
            "in_class": owner,
        }
        for c in node.children:
            if c.type in ("constructor_body", "block"):
                entry["body"] = _text(src, c)
                break
        nodes[sig] = entry

    # [maint-note:c0fceabdf895]
    # 维护导航：_parse_params；下列为本函数源码中的接口与路径索引。
    # 输入 src：bytes。
    # 输入 node：类型由调用方与函数体约束。
    # 返回契约：List[str]。
    # 分支条件：n.type == 'formal_parameters'。
    # 分支条件：r。
    # 分支条件：not fp。
    # 分支条件：c.type in ('formal_parameter', 'receiver_parameter', 'spread_parameter')。
    # 分支条件：cc.type in ('type', 'integral_type', 'floating_point_type', 'boolean_type', 'void_type')。
    # 分支条件：tnode。
    # 遍历边界：c ← n.children。
    # 遍历边界：c ← fp.children。
    # 遍历边界：cc ← c.children。
    # 返回路径：n。
    # 返回路径：r。
    # 返回路径：None。
    # 返回路径：params。
    # 直接调用：find_formal_parameters。
    # 直接调用：params.append。
    # 直接调用：_strip_type。
    # 直接调用：_text。
    # [/maint-note:c0fceabdf895]
    def _parse_params(self, src: bytes, node) -> List[str]:
        # formal_parameters contains formal_parameter nodes
        params: List[str] = []
        # [maint-note:c606c745e294]
        # 维护导航：find_formal_parameters；下列为本函数源码中的接口与路径索引。
        # 输入 n：类型由调用方与函数体约束。
        # 分支条件：n.type == 'formal_parameters'。
        # 分支条件：r。
        # 遍历边界：c ← n.children。
        # 返回路径：n。
        # 返回路径：r。
        # 返回路径：None。
        # 直接调用：find_formal_parameters。
        # [/maint-note:c606c745e294]
        def find_formal_parameters(n):
            if n.type == "formal_parameters":
                return n
            for c in n.children:
                r = find_formal_parameters(c)
                if r:
                    return r
            return None

        fp = find_formal_parameters(node)
        if not fp:
            return params

        for c in fp.children:
            if c.type in ("formal_parameter", "receiver_parameter", "spread_parameter"):
                # grab its type portion
                tnode = None
                for cc in c.children:
                    if cc.type in ("type", "integral_type", "floating_point_type", "boolean_type", "void_type"):
                        tnode = cc
                        break
                if tnode:
                    params.append(_strip_type(_text(src, tnode)))
        return params
