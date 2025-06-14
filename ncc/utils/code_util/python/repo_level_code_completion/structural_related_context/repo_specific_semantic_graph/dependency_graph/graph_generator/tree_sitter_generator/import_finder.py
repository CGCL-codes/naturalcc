import re
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import List, Union, Optional

from tree_sitter import Parser, Language as TS_Language, Tree

from dependency_graph.graph_generator.tree_sitter_generator.info import (
    RegexInfo,
    ParseTreeInfo,
)
from dependency_graph.graph_generator.tree_sitter_generator.load_lib import (
    get_builtin_lib_path,
)
from dependency_graph.models.language import Language
from dependency_graph.utils.read_file import read_file_to_string
from dependency_graph.utils.run_in_subprocess import SubprocessRunner

"""
Tree-sitter query to find all the imports in a code. The captured import name should be named as `import_name`.
"""
FIND_IMPORT_QUERY = {
    # Language.Python: dedent(
    #     """
    #     [
    #       (import_from_statement
    #         module_name: [
    #             (dotted_name) @import_name
    #             (relative_import) @import_name
    #         ]
    #       )
    #       (import_statement
    #         name: [
    #             (dotted_name) @import_name
    #             (aliased_import
    #                 name: (dotted_name) @import_name
    #             )
    #         ]
    #       )
    #     ]
    #     """
    # ),
    # For python, we need the whole import statement to analyze the import symbol
    Language.Python: dedent(
        """
        [
          (import_from_statement) @import_name
          (import_statement) @import_name
        ]
        """
    ),
    Language.Java: dedent(
        """
        (import_declaration
        [
          (identifier) @import_name
          (scoped_identifier) @import_name
        ])
        """
    ),
    Language.Kotlin: dedent(
        """
        (import_header (identifier) @import_name)
        """
    ),
    Language.CSharp: dedent(
        """
        (using_directive
        [
          (qualified_name) @import_name
          (identifier) @import_name
        ])
        """
    ),
    Language.TypeScript: dedent(
        """
        [
            (import_statement (string (string_fragment) @import_name))
            (call_expression
              function: ((identifier) @require_name
                            (#eq? @require_name "require"))
              arguments: (arguments (string (string_fragment) @import_name))
            )
        ]
        """
    ),
    Language.JavaScript: dedent(
        """
        [
            (import_statement (string (string_fragment) @import_name))
            (call_expression
              function: ((identifier) @require_name
                            (#eq? @require_name "require"))
              arguments: (arguments (string (string_fragment) @import_name))
            )
        ]
        """
    ),
    Language.PHP: dedent(
        """
        [
          (require_once_expression (string) @import_name)
          (require_expression (string) @import_name)
          (include_expression (string) @import_name)
        ]
        """
    ),
    Language.Ruby: dedent(
        """
        (call
            method: ((identifier) @require_name
                (#match? @require_name "require_relative|require")
            )
            arguments: (argument_list) @import_name
        )
        """
    ),
    Language.C: dedent(
        """
        (preproc_include path: 
            [
                (string_literal) @import_name
                (system_lib_string) @import_name
            ]
        )
        """
    ),
    Language.CPP: dedent(
        """
        (preproc_include path: 
            [
                (string_literal) @import_name
                (system_lib_string) @import_name
            ]
        )
        """
    ),
    Language.Go: dedent(
        """
        (import_declaration
            [
                (import_spec path: (interpreted_string_literal) @import_name)
                (import_spec_list (import_spec path: (interpreted_string_literal) @import_name))
            ]
        )
        """
    ),
    Language.Swift: dedent(
        """
        (import_declaration (identifier) @import_name)
        """
    ),
    Language.Rust: dedent(
        """
        [
            (use_declaration argument: [(scoped_identifier)(use_wildcard)] @import_name)
            (use_declaration argument: (use_as_clause path: (scoped_identifier) @import_name))
            (use_declaration argument: (scoped_use_list) @import_name)
        ]
        """
    ),
    Language.Lua: dedent(
        """
        (call
            function:
                (variable
                    name: ((identifier) @require_name)
                            (#match? @require_name "require|dofile|loadfile"))
            arguments:
                (argument_list [(expression_list)(string)] @import_name)
        )
        """
    ),
    Language.Bash: dedent(
        """
        (command
            name: ((command_name) @command_name
                    (#match? @command_name "\\\\.|source|bash|zsh|ksh|zsh|csh|dash"))
            argument: (word) @import_name
        )
        """
    ),
    Language.R: dedent(
        """
        (call
            function: ((identifier) @source_name)
                       (#eq? @source_name "source")
            arguments: (arguments (argument) @import_name)
        )
        """
    ),
}

"""
Tree-sitter query to find all the package in a code. The captured pacakge name should be named as `package_name`.
Note that not all languages have packages declared in code.
"""
FIND_PACKAGE_QUERY = {
    Language.Java: dedent(
        """
        (package_declaration
        [
          (identifier) @package_name
          (scoped_identifier) @package_name
        ])
        """
    ),
    Language.Kotlin: dedent(
        """
        (package_header (identifier) @package_name)
        """
    ),
    Language.CSharp: dedent(
        """
        (namespace_declaration
        [
          (qualified_name) @package_name
          (identifier) @package_name
        ])
        """
    ),
    Language.Go: dedent(
        """
        (package_clause (package_identifier) @package_name)
        """
    ),
}

"""
Regex pattern to find all the imports in a code. The captured import name should be matched in group 1
"""
REGEX_FIND_IMPORT_PATTERN = {
    Language.Lua: r"^\s*(?!--).*(?:require|dofile|loadfile)\s*\((.+)\)$",
    Language.R: r"^\s*(?<!#)(?:source)\s*\((.+)\)$",
    Language.Bash: r"^\s*(?<!#)(?:\.|source|bash|zsh|ksh|csh|dash)\s+[\"\']?([^\"\s]+)[\"\']?",
    Language.Swift: r"^\s*(?!\/\/|\/\*).*?\bimport\s+(?:typealias|struct|class|enum|protocol|let|var|func\s+)?(.+?)(?:\s*;)?$",
}


class ImportFinder:
    languages_using_regex = tuple(REGEX_FIND_IMPORT_PATTERN.keys())

    def __init__(self, language: Language):
        lib_path = get_builtin_lib_path()
        self.language = language
        # Initialize the Tree-sitter language
        self.parser = Parser()
        self.ts_language = TS_Language(str(lib_path.absolute()), str(language))
        self.parser.set_language(self.ts_language)
        self._timeout = 5  # seconds

    def _query_and_captures(
        self, code: str, query: str, capture_name: str = "import_name"
    ) -> List[ParseTreeInfo]:
        """
        Query the Tree-sitter language and get the nodes that match the query
        :param code: The code to be parsed
        :param query: The query to be matched
        :param capture_name: The name of the capture group to be matched
        :return: The nodes that match the query
        """
        tree: Tree = self.parser.parse(code.encode())
        query = self.ts_language.query(query)
        captures = query.captures(tree.root_node)
        nodes = [node for node, captured in captures if captured == capture_name]
        info_list = []
        for n in nodes:
            info = ParseTreeInfo(n.start_point, n.end_point, n.text.decode(), n.type)
            if n.parent:
                info.parent = ParseTreeInfo(
                    n.parent.start_point,
                    n.parent.end_point,
                    n.parent.text.decode(),
                    n.parent.type,
                )
            info_list.append(info)
        return info_list

    def _query_and_captures_in_subprocess(
        self, code: str, query: str, capture_name: str = "import_name"
    ):
        """
        Use SubprocessRunner to query the Tree-sitter language and get the nodes that match the query
        to avoid the Tree-sitter deadlock/segmentation fault issue
        """
        captures = SubprocessRunner(
            self._query_and_captures, code, query, capture_name
        ).run(self._timeout)
        return captures

    def _regex_find_imports(self, code: str, pattern: str) -> List[RegexInfo]:
        matches = []
        for match in re.finditer(pattern, code, re.MULTILINE):
            module_name = match.group(1)
            start_index = match.start(1)
            end_index = match.end(1)

            # Calculate line and column number
            start_line = code.count("\n", 0, start_index)
            start_column = start_index - code.rfind("\n", 0, start_index) - 1

            end_line = code.count("\n", 0, end_index)
            end_column = end_index - code.rfind("\n", 0, end_index) - 1

            matches.append(
                RegexInfo(
                    start_point=(start_line, start_column),
                    end_point=(end_line, end_column),
                    text=module_name,
                )
            )
        return matches

    @lru_cache(maxsize=128)
    def find_imports(
        self,
        code: str,
    ) -> Union[List[RegexInfo], List[ParseTreeInfo]]:
        if self.language in self.languages_using_regex:
            return self._regex_find_imports(
                code, REGEX_FIND_IMPORT_PATTERN[self.language]
            )
        else:
            return self._query_and_captures_in_subprocess(
                code, FIND_IMPORT_QUERY[self.language]
            )

    @lru_cache(maxsize=256)
    def find_module_name(self, file_path: Path) -> Optional[str]:
        """
        Find the name of the module of the current file.
        This term is broad enough to encompass the different ways in which these languages organize and reference code units
        In Java, it is the name of the package.
        In C#, it is the name of the namespace.
        In JavaScript/TypeScript, it is the name of the file.
        """
        # Use read_file_to_string here to avoid non-UTF8 decoding issue
        code = read_file_to_string(file_path)
        if self.language in (Language.Java, Language.Kotlin):
            captures = self._query_and_captures_in_subprocess(
                code, FIND_PACKAGE_QUERY[self.language], "package_name"
            )

            if len(captures) > 0:
                node = captures[0]
                package_name = node.text
                module_name = f"{package_name}.{file_path.stem}"
                return module_name

        elif self.language in (Language.CSharp, Language.Go):
            captures = self._query_and_captures_in_subprocess(
                code, FIND_PACKAGE_QUERY[self.language], "package_name"
            )

            if len(captures) > 0:
                node = captures[0]
                package_name = node.text
                return package_name

        elif self.language in (
            Language.TypeScript,
            Language.JavaScript,
            Language.Python,
            Language.Ruby,
            Language.Rust,
            Language.Lua,
            Language.R,
        ):
            return file_path.stem

        elif self.language in (Language.PHP, Language.C, Language.CPP, Language.Bash):
            return file_path.name

        elif self.language == Language.Swift:
            # Swift module name is its parent directory
            return file_path.parent.name

        else:
            raise NotImplementedError(f"Language {self.language} is not supported")
