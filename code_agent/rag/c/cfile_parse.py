import os
import sys
from pathlib import Path
from ctypes.util import find_library

from clang import cindex


def _iter_libclang_candidates():
    override = os.environ.get("LIBCLANG_PATH")
    if override:
        override_path = Path(override)
        if override_path.is_file():
            yield override_path
        elif override_path.is_dir():
            if sys.platform == "win32":
                yield override_path / "libclang.dll"
                for dll in sorted(override_path.glob("libclang-*.dll"), reverse=True):
                    yield dll
                yield override_path / "clang.dll"
            elif sys.platform == "darwin":
                yield override_path / "libclang.dylib"
            else:
                yield override_path / "libclang.so"
                yield override_path / "libclang.so.1"

    prefixes = [Path(sys.prefix)]
    for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(env_name)
        if prefix:
            prefixes.append(Path(prefix))

    seen = set()
    for prefix in prefixes:
        prefix_str = str(prefix.resolve()) if prefix.exists() else str(prefix)
        if prefix_str in seen:
            continue
        seen.add(prefix_str)

        if sys.platform == "win32":
            win_dirs = [
                prefix / "Library" / "bin",
                prefix / "Library" / "lib",
                prefix / "bin",
            ]
            for win_dir in win_dirs:
                yield win_dir / "libclang.dll"
                for dll in sorted(win_dir.glob("libclang-*.dll"), reverse=True):
                    yield dll
                yield win_dir / "clang.dll"
        elif sys.platform == "darwin":
            yield prefix / "lib" / "libclang.dylib"
        else:
            yield prefix / "lib" / "libclang.so"
            yield prefix / "lib" / "libclang.so.1"
            yield prefix / "lib64" / "libclang.so"
            yield prefix / "lib64" / "libclang.so.1"

    for lib_name in ("clang", "libclang"):
        found = find_library(lib_name)
        if found:
            yield Path(found)


def _configure_libclang():
    if cindex.Config.loaded:
        return

    for candidate in _iter_libclang_candidates():
        try:
            if candidate.exists() or not candidate.parent:
                cindex.Config.set_library_file(str(candidate))
                return
        except OSError:
            continue

    env_prefix = sys.prefix
    raise RuntimeError(
        "libclang not found. "
        f"Checked under environment: {env_prefix}. "
        "Install libclang into the active environment, for example:\n"
        "  conda install -n naturalcc -c conda-forge libclang\n"
        "or set LIBCLANG_PATH to the full library path."
    )


_configure_libclang()
from clang.cindex import Index, CursorKind, TranslationUnit
# 利用 libclang 解析 C 文件，提取文件中的函数、变量、结构体、typedef、include 等语法元素，
# 并保存为结构化字典，方便后续做静态分析、代码索引、文档生成或知识图谱构建。

class CAstVisitor(object):
    def __init__(self):
        self.node_info = {}
        # source code
        self.source_code = None
        self.file_path = None
        
        self.COMMENT_KINDS = {CursorKind.MACRO_DEFINITION, CursorKind.INCLUSION_DIRECTIVE}
    

    def clear(self):
        self.node_info = {}
        self.source_code = None
        self.file_path = None


    def get_info(self):
        return self.node_info
    
    
    def set_code(self, source_code, file_path):
        self.source_code = source_code
        self.file_path = file_path


    def _get_code(self, start_location, end_location):
        if not self.source_code:
            return ""

        start_line = start_location.line - 1 
        start_col = start_location.column - 1
        end_line = end_location.line - 1
        end_col = end_location.column - 1

        lines = self.source_code.decode('utf-8', errors='ignore').splitlines()
        
        if start_line >= len(lines):
            return ""
        
        if start_line == end_line:

            if end_col > len(lines[start_line]):
                end_col = len(lines[start_line])
            return lines[start_line][start_col:end_col]
        else:

            result = [lines[start_line][start_col:]]
            for line in range(start_line + 1, end_line):
                if line < len(lines):
                    result.append(lines[line])
            if end_line < len(lines):
                result.append(lines[end_line][:end_col])
            return '\n'.join(result)


    def _save_include_info(self, cursor):
        lineno = cursor.location.line
        include_stmt = cursor.displayname
        
        include_file = cursor.displayname
        
        is_system = include_file.startswith('<') and include_file.endswith('>')
        
        if is_system:
            # <stdio.h> -> stdio.h
            header_name = include_file.strip('<>')
        else:
            # "myheader.h" -> myheader.h
            header_name = include_file.strip('"\'')
            
        header_base = os.path.splitext(os.path.basename(header_name))[0]
        
        if header_base:
            include_stmt = f"#include {include_file}"
            self.node_info[header_base] = {
                "type": "Variable",
                "def": include_stmt,
                "sline": lineno,
                "include": [header_name]
            }


    def _get_docstring(self, cursor):
        try:
            has_comment_support = hasattr(CursorKind, 'COMMENT') or hasattr(cursor.kind, 'is_comment')
            
            if has_comment_support:
                comments = []
                if hasattr(cursor.kind, 'is_comment'):
                    for c in cursor.get_children():
                        if c.kind.is_comment():
                            comments.append(c.spelling)
                    
                    tokens = list(cursor.get_tokens())
                    if tokens:
                        first_token = tokens[0]
                        for c in cursor.translation_unit.cursor.get_children():
                            if c.kind.is_comment() and c.location.line < first_token.location.line:
                                comments.append(c.spelling)
                elif hasattr(CursorKind, 'COMMENT'):
                    for c in cursor.get_children():
                        if c.kind == CursorKind.COMMENT:
                            comments.append(c.spelling)
                    
                    tokens = list(cursor.get_tokens())
                    if tokens:
                        first_token = tokens[0]
                        for c in cursor.translation_unit.cursor.get_children():
                            if c.kind == CursorKind.COMMENT and c.location.line < first_token.location.line:
                                comments.append(c.spelling)
                
                if comments:
                    return "\n".join(comments)
            
            tokens = list(cursor.get_tokens())
            if tokens and self.source_code:
                first_token = tokens[0]
                line_no = first_token.location.line
                
                lines = self.source_code.decode('utf-8', errors='ignore').splitlines()
                comments = []
                
                for i in range(line_no - 2, max(0, line_no - 5), -1):
                    if i < len(lines):
                        line = lines[i].strip()
                        if line.startswith('/*') or line.startswith('//'):
                            comments.append(line)
                        else:
                            if comments:
                                break
                
                if comments:
                    return "\n".join(reversed(comments))
                
            return None
        except Exception as e:
            print(f"Warning: Error getting docstring: {e}")
            return None


    def _get_all_identifiers(self, cursor, save_set):
        if cursor.kind == CursorKind.DECL_REF_EXPR:
            save_set.add(cursor.spelling)
        
        for child in cursor.get_children():
            self._get_all_identifiers(child, save_set)


    def _get_type_info(self, cursor):
        type_name = None
        type_set = set()
        
        if cursor.type and cursor.type.spelling:
            type_name = cursor.type.spelling
            parts = type_name.split()
            for part in parts:
                if part not in ('const', 'volatile', 'static', '*', '&'):
                    type_set.add(part)
        
        return type_name, type_set


    def _process_struct_declaration(self, cursor, struct_type):
        struct_name = cursor.spelling
        if not struct_name:
            struct_name = f"anon_{struct_type}_{cursor.hash}"
            
        lineno = cursor.location.line
        def_content = self._get_code(cursor.extent.start, cursor.extent.end)
        docstring = self._get_docstring(cursor)
        
        for field in cursor.get_children():
            if field.kind == CursorKind.FIELD_DECL:
                field_name = field.spelling
                if field_name:
                    type_name, type_set = self._get_type_info(field)
                    
                    qualified_name = f"{struct_name}.{field_name}"
                    field_stmt = self._get_code(field.extent.start, field.extent.end)
                    
                    self.node_info[qualified_name] = {
                        "type": "Variable",
                        "def": field_stmt,
                        "sline": field.location.line,
                        "in_struct": struct_name
                    }
                    
                    if type_name:
                        self.node_info[qualified_name]["rels"] = [[type_name, "Typeof"]]
        
        self.node_info[struct_name] = {
            "type": struct_type,
            "def": def_content,
            "sline": lineno
        }
        
        if docstring:
            self.node_info[struct_name]["docstring"] = docstring
            
        body_content = ""
        for child in cursor.get_children():
            if child.kind in (CursorKind.FIELD_DECL, CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR):
                body_content += self._get_code(child.extent.start, child.extent.end) + "\n"
        
        if body_content:
            self.node_info[struct_name]["body"] = body_content
            
        return struct_name


    def _process_function_declaration(self, cursor, has_body=False):
        func_name = cursor.spelling
        if not func_name:
            return None
            
        lineno = cursor.location.line
        
        return_type = cursor.result_type.spelling
        
        if has_body:
            docstring = self._get_docstring(cursor)
            
            body_start = None
            for child in cursor.get_children():
                if child.kind == CursorKind.COMPOUND_STMT:
                    body_start = child.extent.start
                    body_content = self._get_code(child.extent.start, child.extent.end)
                    break

            if body_start:
                def_content = self._get_code(cursor.extent.start, body_start).rstrip()
            else:
                def_content = self._get_code(cursor.extent.start, cursor.extent.end)
            
            self.node_info[func_name] = {
                "type": "Function",
                "def": def_content,
                "sline": lineno
            }
            
            if docstring:
                self.node_info[func_name]["docstring"] = docstring
                
            if body_content:
                self.node_info[func_name]["body"] = body_content
                
            if return_type and return_type != "void":
                self.node_info[func_name]["rels"] = [[return_type, "Typeof"]]
                
            for child in cursor.get_children():
                if child.kind == CursorKind.COMPOUND_STMT:
                    self._process_function_body(child, func_name)
                    
        else:
            def_content = self._get_code(cursor.extent.start, cursor.extent.end)
            self.node_info[func_name] = {
                "type": "Function",
                "def": def_content,
                "sline": lineno
            }
            
            if return_type and return_type != "void":
                self.node_info[func_name]["rels"] = [[return_type, "Typeof"]]
                
        return func_name


    def _process_function_body(self, body_cursor, func_name):
        self._process_declarations_in_block(body_cursor, func_name)


    def _process_declarations_in_block(self, block_cursor, func_name):
        for child in block_cursor.get_children():
            if child.kind == CursorKind.VAR_DECL:
                self._process_variable_declaration(child, func_name)
            elif child.kind == CursorKind.COMPOUND_STMT:
                self._process_declarations_in_block(child, func_name)


    def _process_variable_declaration(self, cursor, func_name=None):
        var_name = cursor.spelling
        if not var_name:
            return
            
        lineno = cursor.location.line
        var_def = self._get_code(cursor.extent.start, cursor.extent.end)
        
        type_name, type_set = self._get_type_info(cursor)
        
        entry = {
            "type": "Variable",
            "def": var_def,
            "sline": lineno
        }
        
        if func_name:
            entry["in_function"] = func_name
            
        if type_name:
            entry["rels"] = [[type_name, "Typeof"]]
            
        for child in cursor.get_children():
            if child.kind == CursorKind.INTEGER_LITERAL or child.kind == CursorKind.FLOATING_LITERAL or child.kind == CursorKind.STRING_LITERAL:
                continue
            elif child.kind == CursorKind.INIT_LIST_EXPR:
                continue
            else:
                referred_ids = set()
                self._get_all_identifiers(child, referred_ids)
                
                if referred_ids:
                    if "rels" not in entry:
                        entry["rels"] = []
                    for ref_id in referred_ids:
                        if ref_id != var_name: 
                            entry["rels"].append([ref_id, "Assign"])
        
        self.node_info[var_name] = entry


    def _process_typedef(self, cursor):
        new_type_name = cursor.spelling
        if not new_type_name:
            return
            
        underlying_type = cursor.underlying_typedef_type.spelling
        
        lineno = cursor.location.line
        typedef_def = self._get_code(cursor.extent.start, cursor.extent.end)
        docstring = self._get_docstring(cursor)
        
        entry = {
            "type": "Variable", 
            "def": typedef_def,
            "sline": lineno
        }
        
        if docstring:
            entry["docstring"] = docstring
            
        if underlying_type and underlying_type != new_type_name:
            entry["rels"] = [[underlying_type, "Typeof"]]
            
        self.node_info[new_type_name] = entry
        
        for child in cursor.get_children():
            if child.kind == CursorKind.STRUCT_DECL or child.kind == CursorKind.UNION_DECL or child.kind == CursorKind.ENUM_DECL:
                struct_type = {CursorKind.STRUCT_DECL: 'Struct', 
                              CursorKind.UNION_DECL: 'Union', 
                              CursorKind.ENUM_DECL: 'Enum'}[child.kind]

                if not child.spelling:
                    struct_name = new_type_name
                    self.node_info[struct_name] = {
                        "type": struct_type,
                        "def": self._get_code(child.extent.start, child.extent.end),
                        "sline": child.location.line
                    }
                    
                    for field in child.get_children():
                        if field.kind == CursorKind.FIELD_DECL:
                            field_name = field.spelling
                            if field_name:
                                qualified_name = f"{struct_name}.{field_name}"
                                self.node_info[qualified_name] = {
                                    "type": "Variable",
                                    "def": self._get_code(field.extent.start, field.extent.end),
                                    "sline": field.location.line,
                                    "in_struct": struct_name
                                }
                                
                                field_type = field.type.spelling
                                if field_type:
                                    self.node_info[qualified_name]["rels"] = [[field_type, "Typeof"]]


    def visit_root(self, cursor):
        file_path = ""  
        self.node_info[file_path] = {"type": "Module"}

        for child in cursor.get_children():
            if child.kind == CursorKind.MACRO_DEFINITION and child.location.file and child.location.file.name == self.file_path:
                tokens = list(child.get_tokens())
                if tokens and tokens[0].spelling.startswith('/*'):
                    self.node_info[file_path]["docstring"] = tokens[0].spelling
                break
        
        for child in cursor.get_children():
            if not child.location.file or child.location.file.name != self.file_path:
                continue
                
            if child.kind == CursorKind.INCLUSION_DIRECTIVE:
                self._save_include_info(child)
                
            elif child.kind == CursorKind.VAR_DECL:
                self._process_variable_declaration(child)
                
            elif child.kind == CursorKind.FUNCTION_DECL:
                has_body = any(c.kind == CursorKind.COMPOUND_STMT for c in child.get_children())
                self._process_function_declaration(child, has_body)
                
            elif child.kind == CursorKind.TYPEDEF_DECL:
                self._process_typedef(child)
                
            elif child.kind == CursorKind.STRUCT_DECL:
                self._process_struct_declaration(child, 'Struct')
                
            elif child.kind == CursorKind.UNION_DECL:
                self._process_struct_declaration(child, 'Union')
                
            elif child.kind == CursorKind.ENUM_DECL:
                self._process_struct_declaration(child, 'Enum')


class CParser(object):
    def __init__(self):
        
        self.index = Index.create()
        self.visitor = CAstVisitor()
        self.file_path = None
    
    def set_file_path(self, file_path):
        self.file_path = file_path
    
    def parse(self, c_file):
        try:
            with open(c_file, 'rb') as f:
                source_code = f.read()
            
            args = ['-x', 'c', 
                   '-Xclang', '-detailed-preprocessing-record'] 
            tu = self.index.parse(c_file, args=args, unsaved_files=[(c_file, source_code.decode('utf-8', errors='ignore'))])
            
            if not tu:
                print(f"Error parsing file: {c_file}, Could not create translation unit")
                return {}
            
            self.visitor.clear()
            self.visitor.set_code(source_code, c_file)

            self.visitor.visit_root(tu.cursor)
            
            file_info = self.visitor.get_info()
            
            if "" in file_info:
                file_info[""]["file_path"] = os.path.abspath(c_file)
            
            return file_info
            
        except Exception as e:
            print(f"Error parsing file: {c_file}, Error: {e}")
            import traceback
            traceback.print_exc()
            return {} 
