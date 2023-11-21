import keyword
import logging
import re
from collections import defaultdict, Counter
from symtable import symtable, Symbol
from typing import Any, Dict, Optional, Set, Union, List, FrozenSet

from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.utils import run_and_debug
from typed_ast.ast3 import (
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
    LShift,
    RShift,
    BitOr,
    BitAnd,
    BitXor,
    Pow,
    MatMult,
    ExtSlice,
    Index,
    Compare,
    Await,
    Lambda,
    arg,
    Global,
    Nonlocal,
    arguments,
    Name,
    comprehension,
    alias,
    withitem,
    JoinedStr,
    Assign,
    AnnAssign,
    AugAssign,
    FormattedValue,
    TypeIgnore,
    Attribute,
    Module,
    ImportFrom,
)
from typed_ast.ast3 import And, Or
from typed_ast.ast3 import Eq, Gt, GtE, In, Is, IsNot, Lt, LtE, NotEq, NotIn
from typed_ast.ast3 import Invert, Not, UAdd, USub
from typed_ast.ast3 import (
    NodeVisitor,
    parse,
    AST,
    FunctionDef,
    AsyncFunctionDef,
    Return,
    Yield,
    Subscript,
    Str,
    YieldFrom,
    Starred,
    Delete,
    Break,
    Continue,
    If,
    For,
    AsyncFor,
    While,
    Try,
    Assert,
    With,
    AsyncWith,
    Raise,
    IfExp,
    Call,
)

from .dataflowpass import DataflowPass
from .graphgenutils import EdgeType, TokenNode, StrSymbol, SymbolInformation
from .type_lattice_generator import TypeLatticeGenerator
from .typeparsing import (
    parse_type_annotation_node,
    parse_type_comment,
    TypeAnnotationNode,
)


class AstGraphGenerator(NodeVisitor):
    def __init__(self, source: str, type_graph: TypeLatticeGenerator):
        self.__type_graph = type_graph
        self.__node_to_id: Dict[Any, int] = {}
        self.__id_to_node: List[Any] = []

        self.__symbol_to_supernode_id: Dict[Symbol, int] = {}

        self.__edges: Dict[EdgeType, Dict[int, Set[int]]] = {
            e: defaultdict(set) for e in EdgeType
        }

        self.__ast = parse(source)
        self.__scope_symtable = [symtable(source, "file.py", "exec")]
        self.__symtable_usage_count = Counter()

        self.__imported_symbols = (
            {}
        )  # type: Dict[TypeAnnotationNode, TypeAnnotationNode]

        # For the CHILD edges
        self.__current_parent_node: Optional[AST] = None

        # For the NEXT_TOKEN edges
        self.__backbone_sequence: List[TokenNode] = []
        self.__prev_token_node: Optional[TokenNode] = None

        # For the RETURNS_TO edge
        self.__return_scope: Optional[AST] = None

        # For the OCCURRENCE_OF and Supernodes
        self.__variable_like_symbols: Dict[Any, SymbolInformation] = {}

        # Last Lexical Use
        self.__last_lexical_use: Dict[Any, Any] = {}

    # region Constants
    INDENT = "<INDENT>"
    DEDENT = "<DEDENT>"
    NLINE = "<NL>"

    BOOLOP_SYMBOLS = {And: "and", Or: "or"}

    BINOP_SYMBOLS = {
        Add: "+",
        Sub: "-",
        Mult: "*",
        Div: "/",
        FloorDiv: "//",
        Mod: "%",
        LShift: "<<",
        RShift: ">>",
        BitOr: "|",
        BitAnd: "&",
        BitXor: "^",
        Pow: "**",
        MatMult: "*.",
    }

    CMPOP_SYMBOLS = {
        Eq: "==",
        Gt: ">",
        GtE: ">=",
        In: "in",
        Is: "is",
        IsNot: "is not",
        Lt: "<",
        LtE: "<=",
        NotEq: "!=",
        NotIn: "not in",
    }

    UNARYOP_SYMBOLS = {Invert: "~", Not: "not", UAdd: "+", USub: "-"}

    IDENTIFER_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    BUILT_IN_METHODS_TO_KEEP = frozenset(
        {"__getitem__", "__setitem__", "__enter__", "__call__"}
    )

    # endregion

    def build(self):
        self.visit(self.__ast)
        self.__add_subtoken_of_edges()

        dataflow = DataflowPass(self)
        dataflow.visit(self.__ast)

        def parse_symbol_info(sinfo: SymbolInformation) -> Dict[str, Any]:
            has_annotation = any(
                s is not None for s in sinfo.annotatable_locations.values()
            )

            if has_annotation:
                first_annotatable_location = min(
                    k for k, v in sinfo.annotatable_locations.items() if v is not None
                )
                annotation_str = str(
                    sinfo.annotatable_locations[first_annotatable_location]
                )
            else:
                first_annotatable_location = min(
                    k for k, v in sinfo.annotatable_locations.items()
                )
                annotation_str = None

            return {
                "name": sinfo.name,
                "annotation": None if not has_annotation else annotation_str,
                "location": first_annotatable_location,
                "type": sinfo.symbol_type,
            }

        def is_annotation_worthy(sinfo: SymbolInformation) -> bool:
            if sinfo.name == "self":
                return False  # "self" by convention is not annotated
            elif (
                sinfo.name.startswith("__")
                and sinfo.name.endswith("__")
                and sinfo.name not in self.BUILT_IN_METHODS_TO_KEEP
            ):
                return False  # Build in methods have fixed conventions
            elif any(v == "None" for k, v in sinfo.annotatable_locations.items()):
                return False  # 'None' is deterministically computable
            return True

        return {
            "nodes": [self.node_to_label(n) for n in self.__id_to_node],
            "edges": {
                e.name: {f: list(t) for f, t in v.items() if len(t) > 0}
                for e, v in self.__edges.items()
                if len(v) > 0
            },
            "token-sequence": [self.__node_to_id[t] for t in self.__backbone_sequence],
            "supernodes": {
                self.__node_to_id[node]: parse_symbol_info(symbol_info)
                for node, symbol_info in self.__variable_like_symbols.items()
                if len(symbol_info.annotatable_locations) > 0
                and is_annotation_worthy(symbol_info)
            },
        }

    def __add_subtoken_of_edges(self):
        def is_identifier_node(n):
            if not isinstance(n, str) and not isinstance(n, TokenNode):
                return False
            if not self.IDENTIFER_REGEX.fullmatch(str(n)):
                return False
            if keyword.iskeyword(str(n)):
                return False
            if n == self.INDENT or n == self.DEDENT or n == self.NLINE:
                return False
            return True

        all_identifier_like_nodes: Set[TokenNode] = {
            n for n in self.__node_to_id if is_identifier_node(n)
        }
        subtoken_nodes: Dict[str, TokenNode] = {}

        for node in all_identifier_like_nodes:
            for subtoken in split_identifier_into_parts(str(node)):
                if subtoken == "_":
                    continue
                subtoken_dummy_node = subtoken_nodes.get(subtoken)
                if subtoken_dummy_node is None:
                    subtoken_dummy_node = TokenNode(subtoken)
                    subtoken_nodes[subtoken] = subtoken_dummy_node
                self._add_edge(subtoken_dummy_node, node, EdgeType.SUBTOKEN_OF)

    def __node_id(self, node: Union[AST, TokenNode]) -> int:
        assert not isinstance(node, int), "Node should be an object not its int id"
        idx = self.__node_to_id.get(node)
        if idx is None:
            idx = len(self.__node_to_id)
            assert len(self.__id_to_node) == len(self.__node_to_id)
            self.__node_to_id[node] = idx
            self.__id_to_node.append(node)
        return idx

    def _get_node(self, node_id: int):
        return self.__id_to_node[node_id]

    def _add_edge(
        self,
        from_node: Union[AST, TokenNode],
        to_node: Union[AST, TokenNode],
        edge_type: EdgeType,
    ) -> None:
        from_node_idx = self.__node_id(from_node)
        to_node_idx = self.__node_id(to_node)
        self.__edges[edge_type][from_node_idx].add(to_node_idx)

    def _get_edge_targets(self, from_node, edge_type: EdgeType) -> FrozenSet:
        from_node_idx = self.__node_id(from_node)
        return frozenset(
            self._get_node(n) for n in self.__edges[edge_type][from_node_idx]
        )

    def visit(self, node: AST):
        """Visit a node adding the Child edge."""
        if self.__current_parent_node is not None:
            assert self.__current_parent_node in self.__node_to_id or isinstance(
                self.__current_parent_node, Module
            ), self.__current_parent_node
            self._add_edge(self.__current_parent_node, node, EdgeType.CHILD)
        parent = self.__current_parent_node
        self.__current_parent_node = node
        try:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            if visitor == self.generic_visit:
                logging.warning("Unvisited AST type: %s", node.__class__.__name__)
            return visitor(node)
        finally:
            self.__current_parent_node = parent

    def add_terminal(self, token_node: TokenNode):
        self._add_edge(self.__current_parent_node, token_node, EdgeType.CHILD)
        if self.__prev_token_node is not None:
            self._add_edge(self.__prev_token_node, token_node, EdgeType.NEXT)
        self.__backbone_sequence.append(token_node)
        self.__prev_token_node = token_node

    def __visit_statement_block(self, stmts: List):
        self.add_terminal(TokenNode(self.INDENT))  # Skip ":" since it is implied
        for i, statement in enumerate(stmts):
            self.visit(statement)
            if i < len(stmts) - 1:
                self.add_terminal(TokenNode(self.NLINE))
            if i > 0:
                self._add_edge(stmts[i - 1], statement, edge_type=EdgeType.NEXT)

        self.add_terminal(TokenNode(self.DEDENT))

    def visit_Name_annotatable(
        self,
        node: Name,
        lineno: int,
        col_offset: int,
        can_annotate_here: Optional[bool],
        type_annotation: TypeAnnotationNode = None,
    ):
        self._add_edge(self.__current_parent_node, node, EdgeType.CHILD)
        parent = self.__current_parent_node
        self.__current_parent_node = node
        try:
            return self.__visit_variable_like(
                node.id,
                node.lineno,
                node.col_offset,
                can_annotate_here=can_annotate_here,
                type_annotation=type_annotation,
            )
        finally:
            self.__current_parent_node = parent

    def __visit_variable_like(
        self,
        name: Union[str, AST],
        lineno: int,
        col_offset: int,
        can_annotate_here: Optional[bool],
        type_annotation: TypeAnnotationNode = None,
    ):
        if isinstance(name, Name):
            # Transfer any annotation to the name directly.
            self.visit_Name_annotatable(
                name, lineno, col_offset, can_annotate_here, type_annotation
            )
            return
        name, node, symbol, symbol_type = self.__get_symbol_for_name(
            name, lineno, col_offset
        )

        if type_annotation is not None:
            type_annotation = self.__type_graph.canonicalize_annotation(
                type_annotation, self.__imported_symbols
            )

        if symbol is not None:
            self._add_edge(node, symbol, edge_type=EdgeType.OCCURRENCE_OF)
            symbol_info = self.__variable_like_symbols.get(symbol)
            if symbol_info is None:
                symbol_info = SymbolInformation.create(name, symbol_type)
                self.__variable_like_symbols[symbol] = symbol_info
            symbol_info.locations.append((lineno, col_offset))
            if can_annotate_here:
                symbol_info.annotatable_locations[
                    (lineno, col_offset)
                ] = type_annotation

            # Last lexical use
            last_lexical_use_node = self.__last_lexical_use.get(symbol)
            if last_lexical_use_node is not None:
                self._add_edge(last_lexical_use_node, node, EdgeType.LAST_LEXICAL_USE)
            self.__last_lexical_use[symbol] = node

        if type_annotation is not None:
            self.__type_graph.add_type(type_annotation, self.__imported_symbols)

    def __get_symbol_for_name(self, name, lineno, col_offset):
        """
        Aside from strings and Attribute nodes, name can be Subscript. Skip annotation in this case.
        """
        if isinstance(name, str):
            node = TokenNode(name, lineno, col_offset)
            self.add_terminal(node)

            if (
                self.__scope_symtable[-1].get_type() == "class"
                and name.startswith("__")
                and not name.endswith("__")
            ):
                name = "_" + self.__scope_symtable[-1].get_name() + name

            current_idx = len(self.__scope_symtable) - 1
            while current_idx >= 0:
                try:
                    symbol = self.__scope_symtable[current_idx].lookup(name)
                    break
                except KeyError:
                    current_idx -= 1
            else:
                logging.warning(f'Symbol "{name}"@{lineno}:{col_offset} Not Found!')
                symbol = None
        elif isinstance(name, Attribute):
            node = name
            # Heuristic: create symbols only for attributes of the form X.Y and X.Y.Z
            self.visit(node.value)
            self.add_terminal(TokenNode(".", node.lineno, node.col_offset))
            self.add_terminal(TokenNode(node.attr, node.lineno, node.col_offset))
            if isinstance(node.value, Name):
                name = f"{node.value.id}.{node.attr}"
                symbol = StrSymbol(name)
            elif isinstance(node.value, Attribute) and isinstance(
                node.value.value, Name
            ):
                name = f"{node.value.value.id}.{node.value.attr}.{node.attr}"
                symbol = StrSymbol(name)
            else:
                symbol = None
        else:
            node = name
            symbol = None
            self.visit(node)

        if isinstance(symbol, StrSymbol):
            symbol_type = "variable"
        elif isinstance(symbol, Symbol):
            if symbol.is_namespace():
                symbol_type = "class-or-function"
            elif symbol.is_parameter():
                symbol_type = "parameter"
            elif symbol.is_imported():
                symbol_type = "imported"
            else:
                symbol_type = "variable"
        else:
            symbol_type = None
        return name, node, symbol, symbol_type

    def visit_Name(self, node: Name):
        self.__visit_variable_like(
            node.id, node.lineno, node.col_offset, can_annotate_here=None
        )

    def __enter_child_symbol_table(
        self, symtable_type: str, name: str, lineno: Optional[int] = None
    ):
        """
        When there are function decorators, the lineno in symtable points to the line with `def`, while passed lineno
        refers to the very first decorator. To resolve it, when there are available symbols with mismatched lineno,
        we pick the nearest successor.

        For the comprehension-like objects, the error can be the opposite and we should pick the predecessor.

        When there are several alike elements at the same level (e.g., multiple lambdas) it's hard to choose between
        them and we need some tweaks: we pick the one which was used less times, among them -- the first one.
        """
        occurrences = []
        for child_symtable in self.__scope_symtable[-1].get_children():
            if (
                child_symtable.get_type() == symtable_type
                and child_symtable.get_name() == name
            ):
                occurrences.append(child_symtable)

        if len(occurrences) == 0:
            raise ValueError(
                f"Symbol Table for {name} of type {symtable_type} at {lineno} not found"
            )

        should_reverse = name in ["listcomp", "dictcomp", "setcomp", "genexpr"]
        occurrences.sort(key=lambda table: table.get_lineno(), reverse=should_reverse)

        closest_matching = []
        for child_symtable in occurrences:
            if lineno is not None and (
                (not should_reverse and child_symtable.get_lineno() >= lineno)
                or (should_reverse and child_symtable.get_lineno() <= lineno)
            ):
                # Pick all the closest symtables in the right direction
                if (
                    len(closest_matching) == 0
                    or closest_matching[0].get_lineno() == child_symtable.get_lineno()
                ):
                    closest_matching.append(child_symtable)

        if len(closest_matching) == 0:
            self.__scope_symtable.append(occurrences[-1])
            self.__symtable_usage_count[occurrences[-1].get_id()] += 1
        else:
            # If there are multiple matching symtables (e.g., [lambda x: x, lambda: lambda x: x]), select the one that
            # was used less times, since the order is right.
            selected_table = min(
                closest_matching,
                key=lambda table: self.__symtable_usage_count[table.get_id()],
            )
            self.__scope_symtable.append(selected_table)
            self.__symtable_usage_count[selected_table.get_id()] += 1

    # region Function Parsing

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self.__visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        self.__visit_function(node, is_async=True)

    def __visit_function(
        self, node: Union[FunctionDef, AsyncFunctionDef], is_async: bool
    ):
        for decorator in node.decorator_list:
            self.add_terminal(TokenNode("@"))
            self.visit(decorator)

        if is_async:
            self.add_terminal(TokenNode("async"))
        self.add_terminal(TokenNode("def"))

        t = None
        if node.returns is not None:
            t = parse_type_annotation_node(node.returns)
        elif node.type_comment is not None and "->" in node.type_comment:
            # TODO: Add support for argument types
            t = node.type_comment
            t = t.split("->")[-1].strip()
            t = parse_type_comment(t)

        symbol_name = node.name
        self.__visit_variable_like(
            symbol_name,
            node.lineno,
            node.col_offset,
            can_annotate_here=True,
            type_annotation=t,
        )

        old_return_scope = self.__return_scope
        self.__enter_child_symbol_table("function", node.name, node.lineno)
        try:

            self.add_terminal(TokenNode("("))
            self.visit(node.args)
            self.add_terminal(TokenNode(")"))

            self.__return_scope = node
            self.__visit_statement_block(node.body)
        finally:
            self.__return_scope = old_return_scope
            self.__scope_symtable.pop()

    def visit_Yield(self, node: Yield):
        self._add_edge(node, self.__return_scope, EdgeType.RETURNS_TO)
        self.add_terminal(TokenNode("yield"))
        if node.value is not None:
            self.visit(node.value)

    def visit_YieldFrom(self, node: YieldFrom):
        self._add_edge(node, self.__return_scope, EdgeType.RETURNS_TO)
        self.add_terminal(TokenNode("yield"))
        self.add_terminal(TokenNode("from"))
        self.visit(node.value)

    # endregion

    # region ControlFlow
    def visit_Break(self, node: Break):
        self.add_terminal(TokenNode("break"))

    def visit_Continue(self, node: Continue):
        self.add_terminal(TokenNode("continue"))

    def visit_For(self, node: For):
        self.__visit_for(node, False)

    def visit_AsyncFor(self, node: AsyncFor):
        self.__visit_for(node, True)

    def __visit_for(self, node, is_async: bool):
        if is_async:
            self.add_terminal(TokenNode("async"))
        self.add_terminal(TokenNode("for"))
        self.visit(node.target)
        self.add_terminal(TokenNode("in"))
        self.visit(node.iter)
        self._add_edge(node.target, node.iter, EdgeType.COMPUTED_FROM)

        self.__visit_statement_block(node.body)

        if node.orelse is not None:
            self.add_terminal(TokenNode("else"))
            self.__visit_statement_block(node.orelse)

    def visit_If(self, node: If):
        self.add_terminal(TokenNode("if"))
        self.visit(node.test)
        self.__visit_statement_block(node.body)

        if node.orelse is None:
            return

        self.add_terminal(TokenNode("else"))
        self.__visit_statement_block(node.orelse)

    def visit_IfExp(self, node: IfExp):
        self.visit(node.body)
        self.add_terminal(TokenNode("if"))
        self.visit(node.test)
        self.add_terminal(TokenNode("else"))
        self.visit(node.orelse)

    def visit_Raise(self, node: Raise):
        self._add_edge(node, self.__return_scope, EdgeType.RETURNS_TO)
        self.add_terminal(TokenNode("raise"))
        if node.exc is not None:
            self.visit(node.exc)
            if node.cause is not None:
                self.add_terminal(TokenNode("from"))
                self.visit(node.cause)

    def visit_Return(self, node: Return):
        self._add_edge(node, self.__return_scope, EdgeType.RETURNS_TO)
        self.add_terminal(TokenNode("return"))
        if node.value is not None:
            self.visit(node.value)

    def visit_Try(self, node: Try):
        self.add_terminal(TokenNode("try"))
        self.__visit_statement_block(node.body)
        for i, exc_handler in enumerate(node.handlers):
            self.visit(exc_handler)
            if i > 0:
                self._add_edge(node.handlers[i - 1], exc_handler, EdgeType.NEXT)

        if node.orelse:
            self.add_terminal(TokenNode("else"))
            self.__visit_statement_block(node.orelse)
        if node.finalbody:
            self.add_terminal(TokenNode("finally"))
            self.__visit_statement_block(node.finalbody)

    def visit_ExceptHandler(self, node):
        self.add_terminal(TokenNode("except"))
        if node.type:
            self.visit(node.type)
            if node.name:
                self.__visit_variable_like(
                    node.name, node.lineno, node.col_offset, can_annotate_here=False
                )
        self.__visit_statement_block(node.body)

    def visit_While(self, node: While):
        self.add_terminal(TokenNode("while"))
        self.visit(node.test)
        self.__visit_statement_block(node.body)

        if node.orelse is None:
            return
        self.add_terminal(TokenNode("else"))
        self.__visit_statement_block(node.orelse)

    def visit_With(self, node: With):
        self.__visit_with(node, False)

    def visit_AsyncWith(self, node: AsyncWith):
        self.__visit_with(node, True)

    def __visit_with(self, node: Union[With, AsyncWith], is_asyc: bool):
        # TODO: There is a type comment here! node.type_comment
        if is_asyc:
            self.add_terminal(TokenNode("async"))
        self.add_terminal(TokenNode("with"))
        for i, w_item in enumerate(node.items):
            self.visit(w_item)
            if i < len(node.items) - 1:
                self.add_terminal(TokenNode(","))
        self.__visit_statement_block(node.body)

    def visit_withitem(self, node: withitem):
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.add_terminal(TokenNode("as"))
            self.visit(node.optional_vars)

    # endregion

    # region ClassDef
    def visit_ClassDef(self, node):
        # Add class inheritance (if any)
        self.__type_graph.add_class(
            node.name,
            [
                self.__type_graph.canonicalize_annotation(
                    parse_type_annotation_node(parent), self.__imported_symbols
                )
                for parent in node.bases
            ],
        )
        if len(node.bases) == 0:
            self.__type_graph.add_class(
                node.name, [parse_type_annotation_node("object")]
            )

        for decorator in node.decorator_list:
            self.add_terminal(TokenNode("@"))
            self.visit(decorator)

        self.add_terminal(TokenNode("class"))
        self.add_terminal(TokenNode(node.name, node.lineno, node.col_offset))
        if len(node.bases) > 0:
            self.add_terminal(TokenNode("("))
            for i, base in enumerate(node.bases):
                self.visit(base)
                if i < len(node.bases) - 1:
                    self.add_terminal(TokenNode(","))

            self.add_terminal(TokenNode(")"))

        self.__enter_child_symbol_table("class", node.name, node.lineno)
        try:
            self.__visit_statement_block(node.body)
        finally:
            self.__scope_symtable.pop()

    # endregion

    def visit_Assign(self, node: Assign):
        if (
            hasattr(node, "value")
            and hasattr(node.value, "func")
            and hasattr(node.value.func, "id")
            and node.value.func.id == "NewType"
            and hasattr(node, "value")
            and hasattr(node.value, "args")
            and len(node.value.args) == 2
        ):
            self.__type_graph.add_type_alias(
                parse_type_annotation_node(node.value.args[0]),
                parse_type_annotation_node(node.value.args[1]),
            )

        # TODO: Type aliases are of the form Vector=List[float] how do we parse these?

        # Fails for a chained assignment:
        # a = b = c = expression # Type annotation
        # Here the type annotation seems valid, while there are 3 assignment targets
        # I think that the graph processing should not fail in this case, uncomment if you think the opposite
        # if node.type_comment is not None and len(node.targets) != 1:
        #     assert False

        # When there are several targets, they represent a chained assignment, so we put `=` between them
        # Tuple assignment (like a, b, c = [1, 2, 3]) will be represented by a single Tuple-target
        for i, target in enumerate(node.targets):
            if isinstance(target, Attribute) or isinstance(target, Name):
                self.__visit_variable_like(
                    target,
                    target.lineno,
                    target.col_offset,
                    can_annotate_here=True,
                    type_annotation=parse_type_comment(node.type_comment)
                    if node.type_comment is not None
                    else None,
                )
            else:
                self.visit(target)
            if i > 0:
                self._add_edge(node.targets[i - 1], target, EdgeType.NEXT)

            self._add_edge(target, node.value, EdgeType.COMPUTED_FROM)
            if i < len(node.targets) - 1:
                self.add_terminal(TokenNode("="))

        self.add_terminal(TokenNode("="))
        self.visit(node.value)

    def visit_AugAssign(self, node: AugAssign):
        if isinstance(node.target, Name) or isinstance(node.target, Attribute):
            self.__visit_variable_like(
                node.target, node.lineno, node.col_offset, can_annotate_here=False
            )
        else:
            self.visit(node.target)
        self._add_edge(node.target, node.value, EdgeType.COMPUTED_FROM)

        self.add_terminal(TokenNode(self.BINOP_SYMBOLS[type(node.op)] + "="))
        self.visit(node.value)

    def visit_AnnAssign(self, node: AnnAssign):
        self.__visit_variable_like(
            node.target,
            node.target.lineno,
            node.target.col_offset,
            can_annotate_here=True,
            type_annotation=parse_type_annotation_node(node.annotation),
        )

        if node.value is not None:
            self.add_terminal(TokenNode("="))
            self.visit(node.value)

        self._add_edge(node.target, node.value, EdgeType.COMPUTED_FROM)

    # Resolve imports. Since 99.9% of imports are global, don't account for scoping for simplicity.
    def visit_Import(self, node):
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ImportFrom):
        for alias in node.names:
            if node.module is not None:
                name = parse_type_annotation_node(node.module + "." + alias.name)
            else:
                name = parse_type_annotation_node(alias.name)
            if alias.asname:
                self.__imported_symbols[parse_type_annotation_node(alias.asname)] = name
            elif node.module is not None:
                self.__imported_symbols[parse_type_annotation_node(alias.name)] = name

    # region Ignored
    def visit_Module(self, node):
        self.generic_visit(node)

    def visit_Load(self, node):
        self.generic_visit(node)

    def visit_Store(self, node):
        self.generic_visit(node)

    def visit_TypeIgnore(self, node: TypeIgnore):
        pass

    # endregion

    def visit_Call(self, node: Call):
        self.visit(node.func)
        self.add_terminal(TokenNode("("))
        num_args = len(node.args) + len(node.keywords)
        num_args_added = 0
        for arg in node.args:
            self.visit(arg)
            num_args_added += 1
            if num_args_added < num_args:
                self.add_terminal(TokenNode(","))

        for arg in node.keywords:
            self.visit(arg)
            num_args_added += 1
            if num_args_added < num_args:
                self.add_terminal(TokenNode(","))

        self.add_terminal(TokenNode(")"))

    def visit_Lambda(self, node: Lambda):
        self.add_terminal(TokenNode("lambda"))

        self.__enter_child_symbol_table("function", "lambda", node.lineno)
        self.visit(node.args)
        self.add_terminal(TokenNode(":"))
        self.visit(node.body)
        self.__scope_symtable.pop()

    def visit_arg(self, node: arg):
        type_annotation = None
        if node.annotation is not None:
            type_annotation = parse_type_annotation_node(node.annotation)
        elif node.type_comment is not None:
            type_annotation = parse_type_comment(node.type_comment)

        self.__visit_variable_like(
            node.arg,
            node.lineno,
            node.col_offset,
            can_annotate_here=True,
            type_annotation=type_annotation,
        )

    def visit_arguments(self, node: arguments):
        defaults = [None] * (len(node.args) - len(node.defaults)) + node.defaults
        for i, (argument, default) in enumerate(zip(node.args, defaults)):
            self.visit(argument)
            if default is not None:
                self.add_terminal(TokenNode("="))
                inner_symtable = self.__scope_symtable.pop()
                self.visit(default)
                self.__scope_symtable.append(inner_symtable)
                self._add_edge(argument, default, EdgeType.COMPUTED_FROM)
            self.add_terminal(TokenNode(","))
            if i > 0:
                self._add_edge(node.args[i - 1], argument, EdgeType.NEXT)

        if node.vararg is not None:
            self.add_terminal(TokenNode("*"))
            self.visit(node.vararg)
            self.add_terminal(TokenNode(","))

        if node.kwarg is not None:
            self.add_terminal(TokenNode("**"))
            self.visit(node.kwarg)

        if len(node.kwonlyargs) > 0:
            self.add_terminal(TokenNode("*"))
            self.add_terminal(TokenNode(","))
            defaults = [None] * (
                len(node.kwonlyargs) - len(node.kw_defaults)
            ) + node.kw_defaults
            for argument, default in zip(node.kwonlyargs, defaults):
                self.visit(argument)
                if default is not None:
                    self.add_terminal(TokenNode("="))
                    inner_symtable = self.__scope_symtable.pop()
                    self.visit(default)
                    self.__scope_symtable.append(inner_symtable)
                    self._add_edge(argument, default, EdgeType.COMPUTED_FROM)
                self.add_terminal(TokenNode(","))

    def visit_keyword(self, node):
        if node.arg is not None:
            self.add_terminal(TokenNode(node.arg))
            self.add_terminal(TokenNode("="))
        self.visit(node.value)

    # region Comprehensions
    def visit_comprehension(self, node: comprehension):
        if node.is_async:
            self.add_terminal(TokenNode("async"))
        self.add_terminal(TokenNode("for"))
        self.visit(node.target)

        self.add_terminal(TokenNode("in"))
        inner_symtable = self.__scope_symtable.pop()
        self.visit(node.iter)
        self.__scope_symtable.append(inner_symtable)

        for if_ in node.ifs:
            self.add_terminal(TokenNode("if"))
            self.visit(if_)

    def visit_ListComp(self, node):
        self.__enter_child_symbol_table("function", "listcomp", node.lineno)
        try:
            self.add_terminal(TokenNode("["))
            self.visit(node.elt)
            for i, generator in enumerate(node.generators):
                if i > 0:
                    # When we have multiple generators, then the symbol table of the iter is in the listcomp symboltable.
                    # Reasonable, but I don't see any other
                    self.__scope_symtable.append(self.__scope_symtable[-1])
                self.visit(generator)
                if i > 0:
                    self.__scope_symtable.pop()
            self.add_terminal(TokenNode("]"))
        finally:
            self.__scope_symtable.pop()

    def visit_GeneratorExp(self, node):
        self.__enter_child_symbol_table("function", "genexpr", node.lineno)
        try:
            self.add_terminal(TokenNode("("))
            self.visit(node.elt)
            for i, generator in enumerate(node.generators):
                if i > 0:
                    # When we have multiple generators, then the symbol table of the iter is in the genexpr symboltable.
                    # Reasonable, but I don't see any other
                    self.__scope_symtable.append(self.__scope_symtable[-1])
                self.visit(generator)
                if i > 0:
                    self.__scope_symtable.pop()
            self.add_terminal(TokenNode(")"))
        finally:
            self.__scope_symtable.pop()

    def visit_SetComp(self, node):
        self.__enter_child_symbol_table("function", "setcomp", node.lineno)
        try:
            self.add_terminal(TokenNode("{"))
            self.visit(node.elt)
            for i, generator in enumerate(node.generators):
                if i > 0:
                    # When we have multiple generators, then the symbol table of the iter is in the setcomp symboltable.
                    # Reasonable, but I don't see any other
                    self.__scope_symtable.append(self.__scope_symtable[-1])
                self.visit(generator)
                if i > 0:
                    self.__scope_symtable.pop()
            self.add_terminal(TokenNode("}"))
        finally:
            self.__scope_symtable.pop()

    def visit_DictComp(self, node):
        self.__enter_child_symbol_table("function", "dictcomp", node.lineno)
        try:
            self.add_terminal(TokenNode("{"))
            self.visit(node.key)
            self.add_terminal(TokenNode(":"))
            self.visit(node.value)

            for i, generator in enumerate(node.generators):
                if i > 0:
                    # When we have multiple generators, then the symbol table of the iter is in the dictcomp symboltable.
                    # Reasonable, but I don't see any other
                    self.__scope_symtable.append(self.__scope_symtable[-1])
                self.visit(generator)
                if i > 0:
                    self.__scope_symtable.pop()
            self.add_terminal(TokenNode("}"))
        finally:
            self.__scope_symtable.pop()

    # endregion

    # region Simple Expressions
    def visit_alias(self, node: alias):
        if node.asname is not None:
            self.__imported_symbols[
                parse_type_annotation_node(node.asname)
            ] = parse_type_annotation_node(node.name)

    def visit_Attribute(self, node: Attribute):
        self.__visit_variable_like(node, node.lineno, node.col_offset, False)
        # self.visit(node.value)

    def visit_Assert(self, node: Assert):
        self.add_terminal(TokenNode("assert"))
        self.visit(node.test)
        if node.msg is not None:
            self.add_terminal(TokenNode(","))
            self.visit(node.msg)

    def visit_Await(self, node: Await):
        self.add_terminal(TokenNode("await"))
        self.visit(node.value)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.add_terminal(TokenNode(self.BINOP_SYMBOLS[type(node.op)]))
        self.visit(node.right)

    def visit_BoolOp(self, node):
        for idx, value in enumerate(node.values):
            self.visit(value)
            if idx < len(node.values) - 1:
                self.add_terminal(TokenNode(self.BOOLOP_SYMBOLS[type(node.op)]))

    def visit_Compare(self, node: Compare):
        self.visit(node.left)
        for i, (op, right) in enumerate(zip(node.ops, node.comparators)):
            self.add_terminal(TokenNode(self.CMPOP_SYMBOLS[type(op)]))
            self.visit(right)

    def visit_Delete(self, node: Delete):
        self.add_terminal(TokenNode("del"))
        for i, target in enumerate(node.targets):
            self.visit(target)
            if i < len(node.targets) - 1:
                self.add_terminal(TokenNode(","))

    def visit_Ellipsis(self, node):
        self.add_terminal(TokenNode("..."))

    def visit_ExtSlice(self, node: ExtSlice):
        for i, value in enumerate(node.dims):
            self.visit(value)
            if i < len(node.dims) - 1:
                self.add_terminal(TokenNode(","))

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_Global(self, node: Global):
        self.add_terminal(TokenNode("global"))
        for name in node.names:
            self.__visit_variable_like(
                name, node.lineno, node.col_offset, can_annotate_here=False
            )

    def visit_Index(self, node: Index):
        self.visit(node.value)

    def visit_Nonlocal(self, node: Nonlocal):
        self.add_terminal(TokenNode("nonlocal"))
        for name in node.names:
            self.__visit_variable_like(
                name, node.lineno, node.col_offset, can_annotate_here=False
            )

    def visit_Pass(self, node):
        self.add_terminal(TokenNode("pass"))

    def visit_Slice(self, node):
        self.add_terminal(TokenNode("["))
        if node.lower is not None:
            self.visit(node.lower)
        self.add_terminal(TokenNode(":"))
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self.add_terminal(TokenNode(":"))
            self.visit(node.step)
        self.add_terminal(TokenNode("]"))

    def visit_Subscript(self, node: Subscript):
        self.visit(node.value)
        self.add_terminal(TokenNode("["))
        self.visit(node.slice)
        self.add_terminal(TokenNode("]"))

    def visit_Starred(self, node: Starred):
        self.add_terminal(TokenNode("*"))
        self.visit(node.value)

    def visit_UnaryOp(self, node):
        op = self.UNARYOP_SYMBOLS[type(node.op)]
        self.add_terminal(TokenNode(op))
        self.visit(node.operand)

    # endregion

    # region Data Structure Constructors
    def visit_Dict(self, node):
        self.add_terminal(TokenNode("{"))
        for idx, (key, value) in enumerate(zip(node.keys, node.values)):
            if key is None:
                self.add_terminal(TokenNode("None"))
            else:
                self.visit(key)
            self.add_terminal(TokenNode(":"))
            self.visit(value)
            if idx < len(node.keys) - 1:
                self.add_terminal(TokenNode(","))
        self.add_terminal(TokenNode("}"))

    def visit_FormattedValue(self, node: FormattedValue):
        self.add_terminal(TokenNode(str('f"')))
        self.visit(node.value)
        if node.format_spec is not None:
            self.add_terminal(TokenNode(str(":")))
            self.visit(node.format_spec)
        self.add_terminal(TokenNode(str('"')))

    def visit_List(self, node):
        self.__sequence_datastruct_visit(node, "[", "]")

    def visit_Set(self, node):
        self.__sequence_datastruct_visit(node, "{", "}")

    def visit_Tuple(self, node):
        self.__sequence_datastruct_visit(node, "(", ")")

    def __sequence_datastruct_visit(self, node, open_brace: str, close_brace: str):
        self.add_terminal(TokenNode(open_brace))
        for idx, element in enumerate(node.elts):
            self.visit(element)
            self.add_terminal(
                TokenNode(",")
            )  # Always add , this is always correct and useful for len one tuples.
        self.add_terminal(TokenNode(close_brace))

    # endregion

    # region literals and constructor-likes
    def visit_Bytes(self, node):
        self.add_terminal(TokenNode(repr(node.s)))

    def visit_JoinedStr(self, node: JoinedStr):
        for v in node.values:
            self.visit(v)

    def visit_NameConstant(self, node):
        self.add_terminal(TokenNode(str(node.value)))

    def visit_Num(self, node):
        self.add_terminal(TokenNode(repr(node.n)))

    def visit_Str(self, node: Str):
        self.add_terminal(
            TokenNode('"' + node.s + '"')
        )  # Approximate quote addition, but should be good enough.

    # endregion

    # region Visualization
    def node_to_label(self, node: Any) -> str:
        if isinstance(node, str):
            return node.replace("\n", "").replace('"', "")
        elif isinstance(node, TokenNode):
            return node.token.replace("\n", "").replace('"', "")
        elif isinstance(node, AST):
            return node.__class__.__name__
        elif isinstance(node, Symbol):
            return node.get_name()
        elif isinstance(node, StrSymbol):
            return node.name
        elif node is None:
            return "None"
        else:
            raise Exception("Unrecognized node type %s" % type(node))

    def to_dot(
        self,
        filename: str,
        initial_comment: str = "",
        draw_only_edge_types: Optional[Set[EdgeType]] = None,
    ) -> None:
        nodes_to_be_drawn = set()

        for edge_type, edges in self.__edges.items():
            if (
                draw_only_edge_types is not None
                and edge_type not in draw_only_edge_types
            ):
                continue
            for from_idx, to_idxs in edges.items():
                nodes_to_be_drawn.add(from_idx)
                for to_idx in to_idxs:
                    nodes_to_be_drawn.add(to_idx)

        with open(filename, "w") as f:
            if len(initial_comment) > 0:
                f.write("#" + initial_comment)
                f.write("\n")
            f.write("digraph program {\n")
            for node, node_idx in self.__node_to_id.items():
                if node_idx not in nodes_to_be_drawn:
                    continue
                node_lbl = self.node_to_label(node)
                if len(node_lbl) > 15:
                    node_lbl = node_lbl[:15]
                if hasattr(node, "lineno"):
                    node_lbl += f":L{node.lineno}:{node.col_offset if hasattr(node, 'col_offset') else -1}"
                f.write(f'\t node{node_idx}[shape="rectangle", label="{node_lbl}"];\n')

            for edge_type, edges in self.__edges.items():
                if (
                    draw_only_edge_types is not None
                    and edge_type not in draw_only_edge_types
                ):
                    continue
                for from_idx, to_idxs in edges.items():
                    for to_idx in to_idxs:
                        f.write(
                            f'\tnode{from_idx} -> node{to_idx} [label="{edge_type.name}"];\n'
                        )
            f.write("}\n")  # graph

    # endregion


def test_on_self():
    from glob import iglob
    import os

    lattice = TypeLatticeGenerator("../metadata/typingRules.json")
    for fname in iglob("./testfiles/*.py", recursive=True):
        # for fname in iglob('/mnt/c/Users/t-mialla/Source/Repos/**/*.py', recursive=True):
        if os.path.isdir(fname):
            continue
        print(fname)

        with open(fname) as f:
            try:
                b = AstGraphGenerator(f.read(), lattice)
                b.build()
                b.to_dot(
                    fname + ".dot"
                )  # , draw_only_edge_types={EdgeType.NEXT_USE, EdgeType.OCCURRENCE_OF})
                # import pdb; pdb.set_trace()
            except SyntaxError:
                pass


if __name__ == "__main__":
    run_and_debug(test_on_self, True)
