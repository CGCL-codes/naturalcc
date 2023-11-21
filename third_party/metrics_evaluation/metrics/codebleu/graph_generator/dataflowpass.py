from collections import defaultdict
from itertools import chain
from symtable import SymbolTable
from typing import Any, Dict, Optional, Set, Union, List


from typed_ast.ast3 import (
    Mod,
    Compare,
    Lambda,
    arg,
    Global,
    Nonlocal,
    arguments,
    Name,
    comprehension,
    withitem,
    Assign,
    AnnAssign,
    AugAssign,
    FormattedValue,
    Attribute,
    dump,
)
from typed_ast.ast3 import (
    NodeVisitor,
    AST,
    FunctionDef,
    AsyncFunctionDef,
    Return,
    Subscript,
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
)

from .graphgenutils import EdgeType


class DataflowPass(NodeVisitor):
    def __init__(self, ast_graph_generator):
        self.__graph_generator = ast_graph_generator

        # Last Use
        self.__last_use: Dict[Any, Set[Any]] = defaultdict(set)

        self.__break_uses: Dict[Any, Set[Any]] = defaultdict(set)
        self.__continue_uses: Dict[Any, Set[Any]] = defaultdict(set)
        self.__return_uses: Dict[Any, Set[Any]] = defaultdict(set)

    def __visit_variable_like(self, name: Union[str, AST], parent_node: Optional[AST]):
        if isinstance(name, Name):
            self.visit(name)
            return

        if isinstance(name, AST):
            node = name
        else:
            # We need to find the relevant node in the graph
            assert parent_node is not None
            candidate_children = self.__graph_generator._get_edge_targets(
                parent_node, EdgeType.CHILD
            )
            for child in candidate_children:
                if str(child) == name:
                    node = child
                    break
            else:
                assert False

        # Find relevant symbol (OCCURRENCE_OF)
        candidate_symbols = self.__graph_generator._get_edge_targets(
            node, EdgeType.OCCURRENCE_OF
        )
        if len(candidate_symbols) == 0:
            return
        assert len(candidate_symbols) == 1
        symbol = next(iter(candidate_symbols))
        self.__record_next_use(symbol, node)

    def visit_Name(self, node: Name):
        self.__visit_variable_like(node.id, node)

    def __visit_statement_block(self, stmts: List):
        for statement in stmts:
            self.visit(statement)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self.__visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        self.__visit_function(node, is_async=True)

    def __visit_function(
        self, node: Union[FunctionDef, AsyncFunctionDef], is_async: bool
    ):
        outer_return_uses = self.__return_uses
        self.__return_uses = defaultdict(set)

        before_function_uses = self.__clone_last_uses()
        self.visit(node.args)
        self.__visit_statement_block(node.body)

        # Merge used variables in a dummy return value *only* if they have been accessed within the function.
        self.__last_use = self.__merge_uses(self.__last_use, self.__return_uses)
        for symbol, last_uses in self.__last_use.items():
            if before_function_uses[symbol] == last_uses:
                continue

        self.__return_uses = outer_return_uses

    # endregion

    # region ControlFlow

    def __record_next_use(self, symbol, node) -> None:
        for last_node_used in self.__last_use[symbol]:
            self.__graph_generator._add_edge(last_node_used, node, EdgeType.NEXT_USE)
        self.__last_use[symbol] = {node}

    def __merge_uses(
        self, use_set1: Dict[Any, Set[Any]], use_set2: Dict[Any, Set[Any]]
    ) -> Dict[Any, Set[Any]]:
        merged = defaultdict(set)
        for symbol, last_used_nodes in chain(use_set1.items(), use_set2.items()):
            merged[symbol] |= last_used_nodes
        return merged

    def __clone_last_uses(self) -> Dict[Any, Set[Any]]:
        cloned = defaultdict(set)
        for symbol, last_used_nodes in self.__last_use.items():
            cloned[symbol] = set(last_used_nodes)
        return cloned

    def __loop_back_after(
        self,
        last_uses_at_end_of_loop: Dict[Any, Set[Any]],
        last_uses_just_before_looping_point: Dict[Any, Set[Any]],
    ) -> None:
        for (
            symbol,
            last_use_before_looping_point,
        ) in last_uses_just_before_looping_point.items():
            first_use_after_looping_point = set(
                chain(
                    *(
                        self.__graph_generator._get_edge_targets(
                            node, EdgeType.NEXT_USE
                        )
                        for node in last_use_before_looping_point
                    )
                )
            )

            for from_node in last_uses_at_end_of_loop[symbol]:
                for to_node in first_use_after_looping_point:
                    self.__graph_generator._add_edge(
                        from_node, to_node, EdgeType.NEXT_USE
                    )

    def visit_Break(self, node: Break):
        self.__break_uses = self.__merge_uses(self.__break_uses, self.__last_use)
        self.__last_use = defaultdict(set)

    def visit_Continue(self, node: Continue):
        self.__continue_uses = self.__merge_uses(self.__last_use, self.__continue_uses)
        self.__last_use = defaultdict(set)

    def visit_For(self, node: For):
        self.__visit_for(node, False)

    def visit_AsyncFor(self, node: AsyncFor):
        self.__visit_for(node, True)

    def __visit_for(self, node, is_async: bool):
        self.visit(node.iter)
        last_use_before_loop = self.__clone_last_uses()

        outer_break, outer_continue = self.__break_uses, self.__continue_uses

        self.visit(node.target)
        self.__visit_statement_block(node.body)
        self.__loop_back_after(
            self.__merge_uses(self.__last_use, self.__continue_uses),
            last_use_before_loop,
        )
        self.__last_use = self.__merge_uses(
            self.__last_use,
            self.__merge_uses(last_use_before_loop, self.__continue_uses),
        )

        if node.orelse is not None:
            last_use_after_loop = self.__clone_last_uses()
            self.__visit_statement_block(node.orelse)
            self.__last_use = self.__merge_uses(self.__last_use, last_use_after_loop)

        self.__last_use = self.__merge_uses(
            self.__last_use, self.__break_uses
        )  # Break doesn't go through the else

        self.__break_uses, self.__continue_uses = outer_break, outer_continue

    def visit_If(self, node: If):
        self.visit(node.test)

        last_uses_before_body = self.__clone_last_uses()
        self.__visit_statement_block(node.body)

        if node.orelse is None:
            self.__last_use = self.__merge_uses(self.__last_use, last_uses_before_body)
            return

        last_uses_after_then_body = self.__last_use
        self.__last_use = last_uses_before_body
        self.__visit_statement_block(node.orelse)
        self.__last_use = self.__merge_uses(self.__last_use, last_uses_after_then_body)

    def visit_IfExp(self, node: IfExp):
        self.visit(node.test)

        last_uses_before_body = self.__clone_last_uses()
        self.visit(node.body)
        last_uses_after_body = self.__last_use

        self.__last_use = last_uses_before_body
        self.visit(node.orelse)
        self.__last_use = self.__merge_uses(self.__last_use, last_uses_after_body)

    def visit_Raise(self, node: Raise):
        if node.exc is not None:
            self.visit(node.exc)
            if node.cause is not None:
                self.visit(node.cause)
        self.__visit_return_like()

    def visit_Return(self, node: Return):
        if node.value is not None:
            self.visit(node.value)
        self.__visit_return_like()

    def __visit_return_like(self):
        self.__return_uses = self.__merge_uses(self.__return_uses, self.__last_use)
        self.__last_use = defaultdict(set)

    def visit_Try(self, node: Try):
        # Heuristic: each handler is an if-like statement
        self.__visit_statement_block(node.body)

        before_exec_handlers = self.__clone_last_uses()
        after_exec_handlers = self.__clone_last_uses()
        for i, exc_handler in enumerate(node.handlers):
            self.visit(exc_handler)
            after_exec_handlers = self.__merge_uses(
                after_exec_handlers, self.__last_use
            )
            self.__last_use = before_exec_handlers
            before_exec_handlers = self.__clone_last_uses()

        if node.orelse:
            before_uses = self.__clone_last_uses()
            self.__visit_statement_block(node.orelse)
            self.__last_use = self.__merge_uses(before_uses, self.__last_use)
        if node.finalbody:
            self.__visit_statement_block(node.finalbody)

    def visit_ExceptHandler(self, node):
        if node.type:
            self.visit(node.type)
            if node.name:
                self.__visit_variable_like(node.name, node)
        self.__visit_statement_block(node.body)

    def visit_While(self, node: While):
        last_use_before_loop = self.__clone_last_uses()
        self.visit(node.test)
        last_use_after_loop_test = self.__clone_last_uses()

        outer_break, outer_continue = self.__break_uses, self.__continue_uses

        self.__visit_statement_block(node.body)
        self.__loop_back_after(
            self.__merge_uses(self.__last_use, self.__continue_uses),
            last_use_before_loop,
        )

        self.__last_use = self.__merge_uses(
            self.__last_use,
            self.__merge_uses(last_use_after_loop_test, self.__continue_uses),
        )
        if node.orelse is not None:
            last_use_before_branch = self.__clone_last_uses()
            self.__visit_statement_block(node.orelse)
            self.__last_use = self.__merge_uses(last_use_before_branch, self.__last_use)

        self.__last_use = self.__merge_uses(self.__break_uses, self.__last_use)

        self.__break_uses, self.__continue_uses = outer_break, outer_continue

    def visit_With(self, node: With):
        self.__visit_with(node)

    def visit_AsyncWith(self, node: AsyncWith):
        self.__visit_with(node)

    def __visit_with(self, node: Union[With, AsyncWith]):
        for i, w_item in enumerate(node.items):
            self.visit(w_item)
        self.__visit_statement_block(node.body)

    def visit_withitem(self, node: withitem):
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.visit(node.optional_vars)

    # endregion

    def visit_ClassDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)

        last_uses_before = self.__clone_last_uses()
        last_uses_after = self.__clone_last_uses()
        for statement in node.body:
            self.__last_use = last_uses_before
            last_uses_before = self.__clone_last_uses()
            self.visit(statement)
            last_uses_after = self.__merge_uses(self.__last_use, last_uses_after)
        self.__last_use = last_uses_after

    def visit_Assign(self, node: Assign):
        self.visit(node.value)

        for target in node.targets:
            if isinstance(target, Attribute) or isinstance(target, Name):
                self.__visit_variable_like(target, node)
            else:
                self.visit(target)

    def visit_AugAssign(self, node: AugAssign):
        self.visit(node.value)
        if isinstance(node.target, Name) or isinstance(node.target, Attribute):
            self.__visit_variable_like(node.target, node)
        else:
            self.visit(node.target)

    def visit_AnnAssign(self, node: AnnAssign):
        if node.value is not None:
            self.visit(node.value)
        self.__visit_variable_like(node.target, node)

    def visit_Call(self, node):
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

        for arg in node.keywords:
            self.visit(arg)

    def visit_Lambda(self, node: Lambda):
        self.visit(node.args)
        self.visit(node.body)

    def visit_arg(self, node: arg):
        self.__visit_variable_like(node.arg, node)

    def visit_arguments(self, node: arguments):
        defaults = [None] * (len(node.args) - len(node.defaults)) + node.defaults

        for argument, default in zip(node.args, defaults):
            self.visit(argument)
            if default is not None:
                self.visit(default)

        if node.vararg is not None:
            self.visit(node.vararg)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        if len(node.kwonlyargs) > 0:
            defaults = [None] * (
                len(node.kwonlyargs) - len(node.kw_defaults)
            ) + node.kw_defaults
            for argument, default in zip(node.kwonlyargs, defaults):
                self.visit(argument)
                if default is not None:
                    self.visit(default)

    def visit_keyword(self, node):
        self.visit(node.value)

    # region Comprehensions
    def visit_comprehension(self, node: comprehension):
        self.visit(node.iter)

        before_comp = self.__clone_last_uses()
        for if_ in node.ifs:
            self.visit(if_)

        self.visit(node.target)
        self.__last_use = self.__merge_uses(before_comp, self.__last_use)

    def visit_ListComp(self, node):
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_GeneratorExp(self, node):
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_SetComp(self, node):
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_DictComp(self, node):
        self.visit(node.key)
        self.visit(node.value)

        for generator in node.generators:
            self.visit(generator)

    # endregion

    def visit_Attribute(self, node: Attribute):
        self.__visit_variable_like(node, None)
        self.visit(node.value)

    def visit_Assert(self, node: Assert):
        self.visit(node.test)
        if node.msg is not None:
            self.visit(node.msg)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_BoolOp(self, node):
        for idx, value in enumerate(node.values):
            self.visit(value)

    def visit_Compare(self, node: Compare):
        self.visit(node.left)
        for i, (op, right) in enumerate(zip(node.ops, node.comparators)):
            self.visit(right)

    def visit_Delete(self, node: Delete):
        for i, target in enumerate(node.targets):
            self.visit(target)

    def visit_Global(self, node: Global):
        for name in node.names:
            self.__visit_variable_like(name, node)

    def visit_Nonlocal(self, node: Nonlocal):
        for name in node.names:
            self.__visit_variable_like(name, node)

    def visit_Slice(self, node):
        if node.lower is not None:
            self.visit(node.lower)
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self.visit(node.step)

    def visit_Subscript(self, node: Subscript):
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Starred(self, node: Starred):
        self.visit(node.value)

    def visit_UnaryOp(self, node):
        self.visit(node.operand)

    # endregion

    # region Data Structure Constructors
    def visit_Dict(self, node):
        for idx, (key, value) in enumerate(zip(node.keys, node.values)):
            if key is not None:
                self.visit(key)
            self.visit(value)

    def visit_FormattedValue(self, node: FormattedValue):
        self.visit(node.value)
        if node.format_spec is not None:
            self.visit(node.format_spec)

    def visit_List(self, node):
        self.__sequence_datastruct_visit(node)

    def visit_Set(self, node):
        self.__sequence_datastruct_visit(node)

    def visit_Tuple(self, node):
        self.__sequence_datastruct_visit(node)

    def __sequence_datastruct_visit(self, node):
        for idx, element in enumerate(node.elts):
            self.visit(element)

    # endregion
