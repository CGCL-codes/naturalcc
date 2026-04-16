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

def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _text_span(src: bytes, start: int, end: int) -> str:
    return src[start:end].decode("utf-8", errors="ignore")

def _node_line(node) -> int:
    # tree-sitter point 是 0-based
    return int(node.start_point[0]) + 1

def _strip_type(type_str: str) -> str:
    t = type_str.strip()
    t = _GENERIC_CLEAN.sub("", t)
    t = t.replace("...", "[]")  # varargs
    t = t.replace("@", " ")     # 粗略去注解
    t = " ".join(t.split())
    return t

def _simple_name(type_str: str) -> str:
    t = _strip_type(type_str)
    # 去数组
    t = _ARR_CLEAN.sub("", t).strip()
    # 取最后一段
    return t.split(".")[-1].strip()


def _decl_text_before_body(src: bytes, node, body_types) -> str:
    for c in node.children:
        if c.type in body_types:
            return _text_span(src, node.start_byte, c.start_byte).rstrip()
    return _text(src, node).strip()

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
    def __init__(self):
        self.parser = self._build_parser()

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

    def _parse_params(self, src: bytes, node) -> List[str]:
        # formal_parameters contains formal_parameter nodes
        params: List[str] = []
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
