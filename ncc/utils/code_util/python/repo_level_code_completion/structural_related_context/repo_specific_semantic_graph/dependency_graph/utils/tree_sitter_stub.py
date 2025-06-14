from textwrap import dedent

from tree_sitter import Language, Parser, Tree, Node

from dependency_graph.graph_generator.tree_sitter_generator.load_lib import (
    get_builtin_lib_path,
    TS_LIB_PATH,
)

SPECIAL_CHAR = b" "


def _post_process(text: str) -> str:
    """Post process the code to rstrip the trailing whitespaces and remove the lines having only whitespace"""
    code = "\n".join([c.rstrip() for c in text.splitlines() if set(c) != {" "}])
    return code


def generate_java_stub(code: str, include_comments: bool = True) -> str:
    lib_path = get_builtin_lib_path(TS_LIB_PATH)

    # Initialize the Tree-sitter language
    language = Language(str(lib_path.absolute()), "java")
    parser = Parser()
    parser.set_language(language)

    code_bytes = code.encode()
    tree: Tree = parser.parse(code_bytes)

    code_has_changed = False
    code_bytes_arr = bytearray(code_bytes)

    # Remove import declaration
    query = language.query(
        dedent(
            """
            (import_declaration) @import
            """
        )
    )
    captures = query.captures(tree.root_node)
    for node, _ in captures:
        code_has_changed = True
        code_bytes_arr[node.start_byte : node.end_byte] = SPECIAL_CHAR * (
            node.end_byte - node.start_byte
        )

    # Remove method body
    query = language.query(
        dedent(
            """
            [
              (constructor_declaration) @decl
              (method_declaration) @decl
            ]
            """
        )
    )
    captures = query.captures(tree.root_node)

    for node, _ in captures:
        body: Node = node.child_by_field_name("body")
        if body:
            code_has_changed = True
            code_bytes_arr[body.start_byte : body.end_byte] = SPECIAL_CHAR * (
                body.end_byte - body.start_byte
            )

    # Remove comment
    if not include_comments:
        query = language.query(
            dedent(
                """
                [
                  (line_comment) @comment
                  (block_comment) @comment
                ]
                """
            )
        )
        captures = query.captures(tree.root_node)
        for node, _ in captures:
            code_has_changed = True
            code_bytes_arr[node.start_byte : node.end_byte] = SPECIAL_CHAR * (
                node.end_byte - node.start_byte
            )

    if code_has_changed:
        code = _post_process(code_bytes_arr.decode())

    return code


def generate_c_sharp_stub(code: str, include_comments: bool = True) -> str:
    lib_path = get_builtin_lib_path(TS_LIB_PATH)

    # Initialize the Tree-sitter language
    language = Language(str(lib_path.absolute()), "c_sharp")
    parser = Parser()
    parser.set_language(language)

    code_bytes = code.encode()
    tree: Tree = parser.parse(code_bytes)

    code_has_changed = False
    code_bytes_arr = bytearray(code_bytes)

    # Remove using directive
    query = language.query(
        dedent(
            """
            (using_directive) @using
            """
        )
    )
    captures = query.captures(tree.root_node)
    for node, _ in captures:
        code_has_changed = True
        code_bytes_arr[node.start_byte : node.end_byte] = SPECIAL_CHAR * (
            node.end_byte - node.start_byte
        )

    # Remove method body
    query = language.query(
        dedent(
            """
            [
              (method_declaration) @decl
              (constructor_declaration) @decl
              (destructor_declaration) @decl
              (property_declaration) @decl
              (indexer_declaration) @decl
            ]
            """
        )
    )
    captures = query.captures(tree.root_node)

    for node, _ in captures:
        field_name_to_remove = ("accessors", "body")
        for field_name in field_name_to_remove:
            body: Node = node.child_by_field_name(field_name)
            if body:
                code_has_changed = True
                code_bytes_arr[body.start_byte : body.end_byte] = SPECIAL_CHAR * (
                    body.end_byte - body.start_byte
                )

    # Remove comment
    if not include_comments:
        query = language.query(
            dedent(
                """
                [
                  (comment) @comment
                ]
                """
            )
        )
        captures = query.captures(tree.root_node)
        for node, _ in captures:
            code_has_changed = True
            code_bytes_arr[node.start_byte : node.end_byte] = SPECIAL_CHAR * (
                node.end_byte - node.start_byte
            )

    if code_has_changed:
        code = _post_process(code_bytes_arr.decode())

    return code


def generate_ts_js_stub(code: str, include_comments: bool = True) -> str:
    lib_path = get_builtin_lib_path(TS_LIB_PATH)

    # Initialize the Tree-sitter language
    language = Language(str(lib_path.absolute()), "typescript")
    parser = Parser()
    parser.set_language(language)

    code_bytes = code.encode()
    tree: Tree = parser.parse(code_bytes)

    code_has_changed = False
    code_bytes_arr = bytearray(code_bytes)

    # Remove import statement
    query = language.query(
        dedent(
            """
            (import_statement) @import
            """
        )
    )
    captures = query.captures(tree.root_node)
    for node, _ in captures:
        code_has_changed = True
        code_bytes_arr[node.start_byte : node.end_byte] = SPECIAL_CHAR * (
            node.end_byte - node.start_byte
        )

    # Remove method body
    # TODO should we remove arrow_function too?
    query = language.query(
        dedent(
            """
            [
              (method_definition) @def
              (function_declaration) @def
              (generator_function_declaration) @def
            ]
            """
        )
    )
    captures = query.captures(tree.root_node)

    for node, _ in captures:
        body: Node = node.child_by_field_name("body")
        if body:
            code_has_changed = True
            code_bytes_arr[body.start_byte : body.end_byte] = SPECIAL_CHAR * (
                body.end_byte - body.start_byte
            )

    # Remove comment
    if not include_comments:
        query = language.query(
            dedent(
                """
                [
                  (comment) @comment
                ]
                """
            )
        )
        captures = query.captures(tree.root_node)
        for node, _ in captures:
            code_has_changed = True
            code_bytes_arr[node.start_byte : node.end_byte] = SPECIAL_CHAR * (
                node.end_byte - node.start_byte
            )

    if code_has_changed:
        code = _post_process(code_bytes_arr.decode())

    return code
