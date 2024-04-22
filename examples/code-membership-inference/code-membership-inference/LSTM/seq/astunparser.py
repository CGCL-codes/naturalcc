#!/usr/bin/env python2
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import sys

import six
from six import StringIO


# Large float and imaginary literals get turned into infinities in the AST.
# We unparse those infinities to INFSTR.
INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)


def interleave(inter, f, seq):
    """Call f on each item in seq, calling inter() in between.
    """
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)


class Unparser:
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded. """

    def __init__(self, tree, file=sys.stdout):
        """Unparser(tree, file=sys.stdout) -> None.
         Print the source for tree to file."""
        self.f = file
        self.future_imports = []
        self._indent = 0
        self.dispatch(tree)
        self.f.write("\n")
        # print("", file=self.f)
        self.f.flush()

    def fill(self, text=""):
        "Indent a piece of text, according to the current indentation level"
        self.f.write("\n" + "    " * self._indent + text)

    def write(self, text, type=None):
        "Append a piece of text to the current line."
        self.f.write(six.text_type(text), type)

    def enter(self):
        "Print ':', and increase the indentation."
        self.write(":")
        self._indent += 1

    def leave(self):
        "Decrease the indentation level."
        self._indent -= 1

    def dispatch(self, tree):
        "Dispatcher function, dispatching tree type T to method _T."
        if isinstance(tree, list):
            for t in tree:
                self.dispatch(t)
            return
        meth = getattr(self, "_" + tree.__class__.__name__)
        meth(tree)

    ############### Unparsing methods ######################
    # There should be one method per concrete grammar type #
    # Constructors should be grouped by sum type. Ideally, #
    # this would follow the order in the grammar, but      #
    # currently doesn't.                                   #
    ########################################################

    def _Module(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Interactive(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Expression(self, tree):
        self.dispatch(tree.body)

    # stmt
    def _Expr(self, tree):
        self.fill()
        self.dispatch(tree.value)

    def _NamedExpr(self, tree):
        self.write("(")
        self.dispatch(tree.target)
        self.write(" := ")
        self.dispatch(tree.value)
        self.write(")")

    def _Import(self, t):
        self.fill("import ")
        interleave(lambda: self.write(", "), self.dispatch, t.names)

    def _ImportFrom(self, t):
        # A from __future__ import may affect unparsing, so record it.
        if t.module and t.module == "__future__":
            self.future_imports.extend(n.name for n in t.names)

        self.fill("from ")
        self.write("." * t.level)
        # NOTE: `level` is not stored as values in py150 trees.
        #   This will make `from ..package.name import sth` appear the same as
        #   `from .package.name import sth` or `from package.name import sth`.
        if t.module:
            # NOTE: Reason: Use class name. `parse_python.py:L66-69`.
            self.write(t.module, type="ImportFrom")
        self.write(" import ")
        interleave(lambda: self.write(", "), self.dispatch, t.names)

    def _Assign(self, t):
        self.fill()
        for target in t.targets:
            self.dispatch(target)
            self.write(" = ")
        self.dispatch(t.value)

    def _AugAssign(self, t):
        self.fill()
        self.dispatch(t.target)
        self.write(" ")
        self.write(self.binop[t.op.__class__.__name__])
        self.write("= ")
        self.dispatch(t.value)

    def _AnnAssign(self, t):
        self.fill()
        if not t.simple and isinstance(t.target, ast.Name):
            self.write("(")
        self.dispatch(t.target)
        if not t.simple and isinstance(t.target, ast.Name):
            self.write(")")
        self.write(": ")
        self.dispatch(t.annotation)
        if t.value:
            self.write(" = ")
            self.dispatch(t.value)

    def _Return(self, t):
        self.fill("return")
        if t.value:
            self.write(" ")
            self.dispatch(t.value)

    def _Pass(self, t):
        self.fill("pass")

    def _Break(self, t):
        self.fill("break")

    def _Continue(self, t):
        self.fill("continue")

    def _Delete(self, t):
        self.fill("del ")
        interleave(lambda: self.write(", "), self.dispatch, t.targets)

    def _Assert(self, t):
        self.fill("assert ")
        self.dispatch(t.test)
        if t.msg:
            self.write(", ")
            self.dispatch(t.msg)

    def _Exec(self, t):
        self.fill("exec ")
        self.dispatch(t.body)
        if t.globals:
            self.write(" in ")
            self.dispatch(t.globals)
        if t.locals:
            self.write(", ")
            self.dispatch(t.locals)

    def _Print(self, t):
        self.fill("print ")
        do_comma = False
        if t.dest:
            self.write(">>")
            self.dispatch(t.dest)
            do_comma = True
        for e in t.values:
            if do_comma:
                self.write(", ")
            else:
                do_comma = True
            self.dispatch(e)
        if not t.nl:
            self.write(",")

    def _Global(self, t):
        self.fill("global ")
        # NOTE: Reason: Use "identifier". `parse_python.py:L71`.
        interleave(lambda: self.write(", "), lambda x: self.write(x, type="identifier"), t.names)

    def _Nonlocal(self, t):
        # NOTE: This is not part of PY2.
        # NOTE: Reason: Use "identifier". Similar to the case of "Global".
        self.fill("nonlocal ")
        interleave(lambda: self.write(", "), lambda x: self.write(x, type="identifier"), t.names)

    def _Await(self, t):
        self.write("(")
        self.write("await")
        if t.value:
            self.write(" ")
            self.dispatch(t.value)
        self.write(")")

    def _Yield(self, t):
        self.write("(")
        self.write("yield")
        if t.value:
            self.write(" ")
            self.dispatch(t.value)
        self.write(")")

    def _YieldFrom(self, t):
        self.write("(")
        self.write("yield from")
        if t.value:
            self.write(" ")
            self.dispatch(t.value)
        self.write(")")

    def _Raise(self, t):
        self.fill("raise")
        if six.PY3:
            if not t.exc:
                assert not t.cause
                return
            self.write(" ")
            self.dispatch(t.exc)
            if t.cause:
                self.write(" from ")
                self.dispatch(t.cause)
        else:
            self.write(" ")
            if t.type:
                self.dispatch(t.type)
            if t.inst:
                self.write(", ")
                self.dispatch(t.inst)
            if t.tback:
                self.write(", ")
                self.dispatch(t.tback)

    def _Try(self, t):
        self.fill("try")
        self.enter()
        self.dispatch(t.body)
        self.leave()
        for ex in t.handlers:
            self.dispatch(ex)
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()
        if t.finalbody:
            self.fill("finally")
            self.enter()
            self.dispatch(t.finalbody)
            self.leave()

    def _TryExcept(self, t):
        self.fill("try")
        self.enter()
        self.dispatch(t.body)
        self.leave()

        for ex in t.handlers:
            self.dispatch(ex)
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _TryFinally(self, t):
        if len(t.body) == 1 and isinstance(t.body[0], ast.TryExcept):
            # try-except-finally
            self.dispatch(t.body)
        else:
            self.fill("try")
            self.enter()
            self.dispatch(t.body)
            self.leave()

        self.fill("finally")
        self.enter()
        self.dispatch(t.finalbody)
        self.leave()

    def _ExceptHandler(self, t):
        self.fill("except")
        if t.type:
            self.write(" ")
            self.dispatch(t.type)
        if t.name:
            self.write(" as ")
            if six.PY3:
                # NOTE: This is not part of PY2.
                # NOTE: Reason: Use "identifier". Similar to the case of "Global".
                self.write(t.name, type="identifier")
            else:
                self.dispatch(t.name)
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _ClassDef(self, t):
        self.write("\n")
        for deco in t.decorator_list:
            self.fill("@")
            self.dispatch(deco)
        self.fill("class ")
        # NOTE: Reason: Use class name. `parse_python.py:L64-65`.
        self.write(t.name, type="ClassDef")
        if six.PY3:
            self.write("(")
            comma = False
            for e in t.bases:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.dispatch(e)
            for e in t.keywords:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.dispatch(e)
            if sys.version_info[:2] < (3, 5):
                if t.starargs:
                    if comma:
                        self.write(", ")
                    else:
                        comma = True
                    self.write("*")
                    self.dispatch(t.starargs)
                if t.kwargs:
                    if comma:
                        self.write(", ")
                    else:
                        comma = True
                    self.write("**")
                    self.dispatch(t.kwargs)
            self.write(")")
        elif t.bases:
            self.write("(")
            for a in t.bases:
                self.dispatch(a)
                self.write(", ")
            self.write(")")
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _FunctionDef(self, t):
        self.__FunctionDef_helper(t, "def")

    def _AsyncFunctionDef(self, t):
        self.__FunctionDef_helper(t, "async def")

    def __FunctionDef_helper(self, t, fill_suffix):
        self.write("\n")
        for deco in t.decorator_list:
            self.fill("@")
            self.dispatch(deco)
        self.fill(fill_suffix)
        self.write(" ")
        # NOTE: Reason: Use class name. `parse_python.py:L62-63`
        self.write(t.name, type="FunctionDef")
        self.write("(")
        self.dispatch(t.args)
        self.write(")")
        if getattr(t, "returns", False):
            self.write(" -> ")
            self.dispatch(t.returns)
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _For(self, t):
        self.__For_helper("for ", t)

    def _AsyncFor(self, t):
        self.__For_helper("async for ", t)

    def __For_helper(self, fill, t):
        self.fill(fill)
        self.dispatch(t.target)
        self.write(" in ")
        self.dispatch(t.iter)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _If(self, t):
        self.fill("if ")
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        # collapse nested ifs into equivalent elifs.
        while t.orelse and len(t.orelse) == 1 and isinstance(t.orelse[0], ast.If):
            t = t.orelse[0]
            self.fill("elif ")
            self.dispatch(t.test)
            self.enter()
            self.dispatch(t.body)
            self.leave()
        # final else
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _While(self, t):
        self.fill("while ")
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _generic_With(self, t, async_=False):
        self.fill("async with " if async_ else "with ")
        if hasattr(t, "items"):
            interleave(lambda: self.write(", "), self.dispatch, t.items)
        else:
            self.dispatch(t.context_expr)
            if t.optional_vars:
                self.write(" as ")
                self.dispatch(t.optional_vars)
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _With(self, t):
        self._generic_With(t)

    def _AsyncWith(self, t):
        self._generic_With(t, async_=True)

    # expr
    def _Bytes(self, t):
        # NOTE: This is not part of PY2 and will be removed in PY3.8+.
        # NOTE: Reason: Use class name. Similar to the case of "Str".
        self.write(repr(t.s), type="Bytes")

    def _Str(self, tree):
        # NOTE: This will be removed in PY3.8+.
        # NOTE: Reason: Use class name. `parse_python.py:L56-57`.
        if six.PY3:
            self.write(repr(tree.s), type="Str")
        else:
            # NOTE: py150 nodes will keep string in value form, not repr form.
            #   We keep this part as it is to preserve consistency after training.
            # -----
            # if from __future__ import unicode_literals is in effect,
            # then we want to output string literals using a 'b' prefix
            # and unicode literals with no prefix.
            if "unicode_literals" not in self.future_imports:
                self.write(repr(tree.s), type="Str")
            elif isinstance(tree.s, str):
                self.write("b" + repr(tree.s), type="Str")
            elif isinstance(tree.s, unicode):
                self.write(repr(tree.s).lstrip("u"), type="Str")
            else:
                assert False, "shouldn't get here"

    def _JoinedStr(self, t):
        # NOTE: This is not part of PY2.
        # JoinedStr(expr* values)
        self.write("f")
        string = StringIO()
        self._fstring_JoinedStr(t, string.write)
        # Deviation from `unparse.py`: Try to find an unused quote.
        # This change is made to handle _very_ complex f-strings.
        v = string.getvalue()
        if "\n" in v or "\r" in v:
            quote_types = ["'''", '"""']
        else:
            quote_types = ["'", '"', '"""', "'''"]
        for quote_type in quote_types:
            if quote_type not in v:
                v = "{quote_type}{v}{quote_type}".format(quote_type=quote_type, v=v)
                break
        else:
            v = repr(v)
        # NOTE: Reason: Use class name. Similar to the case of "Str".
        self.write(v, type="JoinedStr")

    def _FormattedValue(self, t):
        # NOTE: This is not part of PY2.
        # FormattedValue(expr value, int? conversion, expr? format_spec)
        self.write("f")
        string = StringIO()
        self._fstring_JoinedStr(t, string.write)
        # NOTE: Reason: Use class name. Similar to the case of "Str".
        self.write(repr(string.getvalue()), type="FormattedValue")

    def _fstring_JoinedStr(self, t, write):
        for value in t.values:
            meth = getattr(self, "_fstring_" + type(value).__name__)
            meth(value, write)

    def _fstring_Str(self, t, write):
        value = t.s.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_Constant(self, t, write):
        assert isinstance(t.value, str)
        value = t.value.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_FormattedValue(self, t, write):
        write("{")
        expr = StringIO()
        Unparser(t.value, expr)
        expr = expr.getvalue().rstrip("\n")
        if expr.startswith("{"):
            write(" ")  # Separate pair of opening brackets as "{ {"
        write(expr)
        if t.conversion != -1:
            conversion = chr(t.conversion)
            assert conversion in "sra"
            write("!{conversion}".format(conversion=conversion))
        if t.format_spec:
            write(":")
            meth = getattr(self, "_fstring_" + type(t.format_spec).__name__)
            meth(t.format_spec, write)
        write("}")

    def _Name(self, t):
        # NOTE: PY2, PY3 grammar: Name(identifier id, expr_context ctx)
        # NOTE: Reason: Use class name + context name. `parse_python.py:L125-127`.
        #   ETH parser merges the value of `expr_context` into its parent node.
        #   From [PY2 grammar](https://docs.python.org/2/library/ast.html#abstract-grammar):
        #   ```
        #       expr_context = Load | Store | Del | AugLoad | AugStore | Param
        #   ```
        self.write(t.id, type="Name" + t.ctx.__class__.__name__)

    def _NameConstant(self, t):
        # NOTE: This is not part of PY2 and will be removed PY3.8+.
        # NOTE: Use class name. Similar to the case of "str".
        self.write(repr(t.value), type="NameConstant")

    def _Repr(self, t):
        self.write("`")
        self.dispatch(t.value)
        self.write("`")

    def _write_constant(self, value):
        # NOTE: Reason: Use class name. Similar to the case of "Str".
        if isinstance(value, (float, complex)):
            # Substitute overflowing decimal literal for AST infinities.
            self.write(repr(value).replace("inf", INFSTR), type="Constant")
        else:
            self.write(repr(value), type="Constant")

    def _Constant(self, t):
        # NOTE: This is not part of PY2 and will be removed PY3.8+.
        value = t.value
        if isinstance(value, tuple):
            self.write("(")
            if len(value) == 1:
                self._write_constant(value[0])
                self.write(",")
            else:
                interleave(lambda: self.write(", "), self._write_constant, value)
            self.write(")")
        elif value is Ellipsis:  # instead of `...` for Py2 compatibility
            self.write("...")
        else:
            if t.kind == "u":
                self.write("u")
            self._write_constant(t.value)

    def _Num(self, t):
        # NOTE: Reason: Use class name. `parse_python.py:L54-55`.
        # NOTE: Here we use `repr()` while `parse_python.py` uses `unicode()`.
        #   This causes disparity such as:
        #       | value | repr() | str() | unicode() | notes |
        #       | ----- | ------ | ----- | --------- | ----- |
        #       | 1L | '1L' | '1' | u'1' | long int |
        #       | 3e13 | '30000000000000.0' | '3e+13' | u'3e+13' | exponent notation |
        #       | 1254213006.517507 | '1254213006.517507' | '1254213006.52' | u'1254213006.52' | floating point precision |
        #   Here we keep the part as it is to preserve consistency.
        repr_n = repr(t.n)
        if six.PY3:
            self.write(repr_n.replace("inf", INFSTR), type="Num")
        else:
            # Parenthesize negative numbers, to avoid turning (-1)**2 into -1**2.
            if repr_n.startswith("-"):
                self.write("(")
            if "inf" in repr_n and repr_n.endswith("*j"):
                repr_n = repr_n.replace("*j", "j")
            # Substitute overflowing decimal literal for AST infinities.
            self.write(repr_n.replace("inf", INFSTR), type="Num")
            if repr_n.startswith("-"):
                self.write(")")

    def _List(self, t):
        self.write("[")
        interleave(lambda: self.write(", "), self.dispatch, t.elts)
        self.write("]")

    def _ListComp(self, t):
        self.write("[")
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("]")

    def _GeneratorExp(self, t):
        self.write("(")
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write(")")

    def _SetComp(self, t):
        self.write("{")
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("}")

    def _DictComp(self, t):
        self.write("{")
        self.dispatch(t.key)
        self.write(": ")
        self.dispatch(t.value)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("}")

    def _comprehension(self, t):
        if getattr(t, "is_async", False):
            self.write(" async for ")
        else:
            self.write(" for ")
        self.dispatch(t.target)
        self.write(" in ")
        self.dispatch(t.iter)
        for if_clause in t.ifs:
            self.write(" if ")
            self.dispatch(if_clause)

    def _IfExp(self, t):
        self.write("(")
        self.dispatch(t.body)
        self.write(" if ")
        self.dispatch(t.test)
        self.write(" else ")
        self.dispatch(t.orelse)
        self.write(")")

    def _Set(self, t):
        assert t.elts  # should be at least one element
        self.write("{")
        interleave(lambda: self.write(", "), self.dispatch, t.elts)
        self.write("}")

    def _Dict(self, t):
        self.write("{")

        def write_key_value_pair(k, v):
            self.dispatch(k)
            self.write(": ")
            self.dispatch(v)

        def write_item(item):
            k, v = item
            if k is None:
                # for dictionary unpacking operator in dicts {**{'y': 2}}
                # see PEP 448 for details
                self.write("**")
                self.dispatch(v)
            else:
                write_key_value_pair(k, v)

        interleave(lambda: self.write(", "), write_item, zip(t.keys, t.values))
        self.write("}")

    def _Tuple(self, t):
        self.write("(")
        if len(t.elts) == 1:
            elt = t.elts[0]
            self.dispatch(elt)
            self.write(",")
        else:
            interleave(lambda: self.write(", "), self.dispatch, t.elts)
        self.write(")")

    unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}

    def _UnaryOp(self, t):
        self.write("(")
        self.write(self.unop[t.op.__class__.__name__])
        self.write(" ")
        if six.PY2 and isinstance(t.op, ast.USub) and isinstance(t.operand, ast.Num):
            # If we're applying unary minus to a number, parenthesize the number.
            # This is necessary: -2147483648 is different from -(2147483648) on
            # a 32-bit machine (the first is an int, the second a long), and
            # -7j is different from -(7j).  (The first has real part 0.0, the second
            # has real part -0.0.)
            self.write("(")
            self.dispatch(t.operand)
            self.write(")")
        else:
            self.dispatch(t.operand)
        self.write(")")

    binop = {
        "Add": "+",
        "Sub": "-",
        "Mult": "*",
        "MatMult": "@",
        "Div": "/",
        "Mod": "%",
        "LShift": "<<",
        "RShift": ">>",
        "BitOr": "|",
        "BitXor": "^",
        "BitAnd": "&",
        "FloorDiv": "//",
        "Pow": "**",
    }

    def _BinOp(self, t):
        self.write("(")
        self.dispatch(t.left)
        self.write(" ")
        self.write(self.binop[t.op.__class__.__name__])
        self.write(" ")
        self.dispatch(t.right)
        self.write(")")

    cmpops = {
        "Eq": "==",
        "NotEq": "!=",
        "Lt": "<",
        "LtE": "<=",
        "Gt": ">",
        "GtE": ">=",
        "Is": "is",
        "IsNot": "is not",
        "In": "in",
        "NotIn": "not in",
    }

    def _Compare(self, t):
        self.write("(")
        self.dispatch(t.left)
        for o, e in zip(t.ops, t.comparators):
            self.write(" ")
            self.write(self.cmpops[o.__class__.__name__])
            self.write(" ")
            self.dispatch(e)
        self.write(")")

    boolops = {ast.And: "and", ast.Or: "or"}

    def _BoolOp(self, t):
        self.write("(")
        s = " %s " % self.boolops[t.op.__class__]
        interleave(lambda: self.write(s), self.dispatch, t.values)
        self.write(")")

    def _Attribute(self, t):
        self.dispatch(t.value)
        # Special case: 3.__abs__() is a syntax error, so if t.value
        # is an integer literal then we need to either parenthesize
        # it or add an extra space to get 3 .__abs__().
        if isinstance(
            t.value, getattr(ast, "Constant", getattr(ast, "Num", None))
        ) and isinstance(t.value.n, int):
            self.write(" ")
        self.write(".")
        # NOTE: Reason: Special case. `parse_python.py:L131-132`.
        self.write(t.attr, type="attr")

    def _Call(self, t):
        self.dispatch(t.func)
        self.write("(")
        comma = False
        for e in t.args:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.dispatch(e)
        for e in t.keywords:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.dispatch(e)
        if sys.version_info[:2] < (3, 5):
            if t.starargs:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.write("*")
                self.dispatch(t.starargs)
            if t.kwargs:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.write("**")
                self.dispatch(t.kwargs)
        self.write(")")

    def _Subscript(self, t):
        self.dispatch(t.value)
        self.write("[")
        self.dispatch(t.slice)
        self.write("]")

    def _Starred(self, t):
        self.write("*")
        self.dispatch(t.value)

    # slice
    def _Ellipsis(self, t):
        self.write("...")

    def _Index(self, t):
        self.dispatch(t.value)

    def _Slice(self, t):
        if t.lower:
            self.dispatch(t.lower)
        self.write(":")
        if t.upper:
            self.dispatch(t.upper)
        if t.step:
            self.write(":")
            self.dispatch(t.step)

    def _ExtSlice(self, t):
        interleave(lambda: self.write(", "), self.dispatch, t.dims)

    # argument
    def _arg(self, t):
        # NOTE: This is not part of PY2.
        # NOTE: Reason: Use class name. Default behaviour of `parse_python.py`.
        self.write(t.arg, type="arg")
        if t.annotation:
            self.write(": ")
            self.dispatch(t.annotation)

    # others
    def _arguments(self, t):
        # NOTE: PY2 grammar: arguments = (
        #   expr* args, identifier? vararg,
        #   identifier? kwarg, expr* defaults)
        # NOTE: PY3.7 grammar: arguments = (
        #   arg* args, arg? vararg, arg* kwonlyargs,
        #   expr* kw_defaults, arg? kwarg, expr* defaults)
        # NOTE: PY3.8 grammar: arguments = (
        #   arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
        #   expr* kw_defaults, arg? kwarg, expr* defaults)
        first = True
        # normal arguments
        # NOTE: `posonlyargs` is not part of PY2 and appears only starting PY3.8.
        all_args = getattr(t, "posonlyargs", []) + t.args
        defaults = [None] * (len(all_args) - len(t.defaults)) + t.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:
                first = False
            else:
                self.write(", ")
            self.dispatch(a)
            if d:
                self.write("=")
                self.dispatch(d)
            if index == len(getattr(t, "posonlyargs", ())):
                self.write(", ")
                self.write("/")

        # varargs, or bare '*' if no varargs but keyword-only arguments present
        # NOTE: `kwonlyargs` is not part of PY2.
        if t.vararg or getattr(t, "kwonlyargs", False):
            if first:
                first = False
            else:
                self.write(", ")
            self.write("*")
            if t.vararg:
                if hasattr(t.vararg, "arg"):
                    # NOTE: This is not part of PY2.
                    # NOTE: Reason: Special case. Following the case of "vararg".
                    self.write(t.vararg.arg, type="vararg")
                    if t.vararg.annotation:
                        self.write(": ")
                        self.dispatch(t.vararg.annotation)
                else:
                    # NOTE: Reason: Special case. `parse_python.py:L105`.
                    self.write(t.vararg, type="vararg")
                    if getattr(t, "varargannotation", None):
                        self.write(": ")
                        self.dispatch(t.varargannotation)

        # keyword-only arguments
        if getattr(t, "kwonlyargs", False):
            for a, d in zip(t.kwonlyargs, t.kw_defaults):
                if first:
                    first = False
                else:
                    self.write(", ")
                self.dispatch(a),
                if d:
                    self.write("=")
                    self.dispatch(d)

        # kwargs
        if t.kwarg:
            if first:
                first = False
            else:
                self.write(", ")
            if hasattr(t.kwarg, "arg"):
                # NOTE: This is not part of PY2.
                self.write("**")
                # NOTE: Reason: Special case. Following the case of "kwarg".
                self.write(t.kwarg.arg, type="kwarg")
                if t.kwarg.annotation:
                    self.write(": ")
                    self.dispatch(t.kwarg.annotation)
            else:
                self.write("**")
                # NOTE: Reason: Special case. `parse_python.py:L107`.
                self.write(t.kwarg, type="kwarg")
                if getattr(t, "kwargannotation", None):
                    self.write(": ")
                    self.dispatch(t.kwargannotation)

    def _keyword(self, t):
        if t.arg is None:
            # starting from Python 3.5 this denotes a kwargs part of the invocation
            self.write("**")
        else:
            # NOTE: Reason: Use class name. `parse_python.py:L72-73`.
            self.write(t.arg, type="keyword")
            self.write("=")
        self.dispatch(t.value)

    def _Lambda(self, t):
        self.write("(")
        self.write("lambda ")
        self.dispatch(t.args)
        self.write(": ")
        self.dispatch(t.body)
        self.write(")")

    def _alias(self, t):
        # NOTE: Reason: Use class name. `parse_python.py:L59`.
        self.write(t.name, type="alias")
        if t.asname:
            self.write(" as ")
            # NOTE: Use "identifier". `parse_python.py:L61`.
            self.write(t.asname, type="identifier")

    def _withitem(self, t):
        self.dispatch(t.context_expr)
        if t.optional_vars:
            self.write(" as ")
            self.dispatch(t.optional_vars)
