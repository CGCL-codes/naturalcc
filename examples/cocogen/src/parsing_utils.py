from abc import ABC, abstractmethod
import tree_sitter


parser = None
IDENTIFIER_QUERIES = {
    "java": """
    (identifier) @identifier
    """,
    "python": """
    (identifier) @identifier
    """
}

TYPE_QUERIES = {
    "java": """
    """,
    "python": """
    """
}

class Parser(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def parse(self, code, fileObject=None):
        pass

    @abstractmethod
    def query(self, node, query_str):
        pass


class JavaParser(Parser):
    def __init__(self):
        super().__init__()
        self.language_p = tree_sitter.Language('../../third_party/parsers/tree-sitter-java-0.20.2/java_parser', 'java')
        self.language = "java"
        self.parser = tree_sitter.Parser()
        self.parser.set_language(self.language_p)

    def parse(self, code):
        print(f"Parsing Java Code\n-----------------------\n{code}\n-----------------------")
        return self.parser.parse(bytes(code, "utf8"))

    def query(self, node, query, result_position = 0):
        if(type(query) == str):
            query = self.language_p.query(query)
            return query.captures(node)
        else:
            raise AssertionError("Internal Error.")

class PythonParser(Parser):
    def __init__(self):
        super().__init__()
        self.language_p = tree_sitter.Language('../../third_party/parsers/tree-sitter-python-0.20.4/python_parser', 'python')
        self.language = "python"
        self.parser = tree_sitter.Parser()
        self.parser.set_language(self.language_p)

    def parse(self, code):
        # print(f"Parsing Python Code\n-----------------------\n{code}\n-----------------------")
        return self.parser.parse(bytes(code, "utf8"))

    def query(self, node, query, result_position = 0):
        if(type(query) == str):
            query = self.language_p.query(query)
            return query.captures(node)
        else:
            raise AssertionError("Internal Error.")

PARSERS = {
    "python": PythonParser(),
    "java": JavaParser()
}

def dedup(li):
    dedup_li = []
    # Initialize an empty set to track seen elements
    seen = set()
    for item in li:
        if item not in seen:
            seen.add(item)
            dedup_li.append(item)
    return dedup_li

def syntax_checking(code, language):
    parser = PARSERS[language]
    tree = parser.parse(code)

    # Function to recursively check each node
    def check_node(node):
        # Check if the current node is an error
        if node.type == 'ERROR':
            return False
        # Recursively check all children of the current node
        for child in node.children:
            if not check_node(child):
                return False
        return True

    # Start the recursion from the root node
    return check_node(tree.root_node)


def get_identifiers_from_code(code, language):
    parser = PARSERS[language]
    tree = parser.parse(code)
    query = IDENTIFIER_QUERIES[language]
    captures = parser.query(tree.root_node, query)
    idents = [x[0].text for x in captures]
    print(f"Idents: {idents}")
    idents = dedup(idents)
    return idents
    pass

def substitute_function_in_code(code, function_name, new_body, language):
    # Initialize the parser for the specified language
    parser = PARSERS[language]
    tree = parser.parse(code)

    # Walk the AST to find the function
    cursor = tree.walk()

    # Function to recursively search for the function_definition node
    def find_function_definition(cursor):
        if cursor.node.type == 'decorated_definition':
            # Check if this decorated_definition contains the function we're looking for
            function_node = cursor.node.child_by_field_name('definition')
            if function_node:
                first_child = function_node.child_by_field_name('name')
                if first_child and first_child.text.decode('utf8') == function_name:
                    return cursor.node
        elif cursor.node.type == 'function_definition':
            # Check for function_definition without decorators
            first_child = cursor.node.child_by_field_name('name')
            if first_child and first_child.text.decode('utf8') == function_name:
                return cursor.node
        # Recursively search in each child
        if cursor.goto_first_child():
            while True:
                found = find_function_definition(cursor)
                if found:
                    return found
                if not cursor.goto_next_sibling():
                    break
            cursor.goto_parent()
        return None

    function_node = find_function_definition(cursor)

    if function_node is None:
        raise ValueError("Function not found")
    # Error 1. Not considering decorator
    # Error 2. Not considering blank line (I should use AST to determine the indentation)
    # Get start and end byte positions of the old function
    start_byte = function_node.start_byte
    end_byte = function_node.end_byte
    # Determine the indentation level
    bcode = bytes(code, 'utf-8')
    start_line_index = bcode.rfind(ord('\n'), 0, start_byte) + 1
    indent = ''
    while start_line_index < len(bcode) and chr(bcode[start_line_index]) in ' \t':
        indent += chr(bcode[start_line_index])
        start_line_index += 1

    # Adjust the indentation of the new body
    indented_new_body = '\n'.join(indent + line for line in new_body.split('\n')).strip()

    # Replace the old function text with the new body
    new_code = bcode[:start_byte] + bytes(indented_new_body, 'utf-8') + bcode[end_byte:]
    new_code = new_code.decode('utf-8')
    # Calculate the start and end line numbers
    start_line_num = bcode.count(ord('\n'), 0, start_byte) + 1
    end_line_num = start_line_num + indented_new_body.count('\n')
    return new_code, start_line_num, end_line_num

    #
    # # Replace the old function text with the new body
    # new_code = code[:start_byte] + new_body + code[end_byte:]
    #
    # return new_code


# def substitute_function_in_code(code, function_name, new_body, language):
#     parser = PARSERS[language]
#     tree = parser.parse(code)


