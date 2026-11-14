"""Bounded, source-based C/C++ checks. No labels, builds or target execution.

These checks deliberately expose their evidence and limits: local string flow
into a shell, and direct global accesses in overlapping thread lifetimes.
"""
from __future__ import annotations

import re
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp

from code_agent.security_c_macros import neutralize, structural_groups
from code_agent.security_exam import switch_off

VERSION = "c-source-v4"
LIMITATIONS = (
    "Syntax analysis, not a compiler or proof (macros the grammar cannot expand are neutralised "
    "first; see security_c_macros): local string flow and direct global "
    "accesses in overlapping pthread/std::thread lifetimes. No general alias, macro, "
    "cross-file, condition/path, custom sanitizer or custom synchronization analysis. "
    "Review evidence; use instrumented tests for runtime confirmation."
)


# [maint-note:823dfe19dc42]
# 定位约束：宏中和仅替换字符，不移动原文件的行列与字节位置。
# 解析失败时尝试结构分支，选取错误更少的表示，而非宣布源码正确。
# 返回的语法树服务于有界规则；不代表完整编译和路径证明。
# 维护导航：parse_source；下列为本函数源码中的接口与路径索引。
# 输入 source：str。
# 输入 suffix：str。
# 输入 macros：类型由调用方与函数体约束；缺省表达式 None。
# 分支条件：not root.has_error。
# 分支条件：len(arms) < 2 or not _has_error_between(root, arms[0][0], end)。
# 分支条件：count < errors。
# 分支条件：not errors。
# 遍历边界：(number, (arms, start, end)) ← enumerate(groups)。
# 遍历边界：arm ← range(1, len(arms))。
# 返回路径：root。
# 直接调用：Parser。
# 直接调用：Language。
# 直接调用：language.language。
# 直接调用：neutralize。
# 直接调用：parser.parse。
# 直接调用：text.encode。
# 直接调用：_error_count。
# 直接调用：structural_groups。
# 直接调用：enumerate。
# 直接调用：len。
# 直接调用：_has_error_between。
# 直接调用：range。
# 直接调用：neutralize(source, object_like, function_like, trial).encode。
# 表达式分流：suffix in {'.cc', '.cpp', '.cxx', '.hpp'}；真值与假值分支分别计算，不同时执行。
# [/maint-note:823dfe19dc42]
def parse_source(source: str, suffix: str, macros=None):
    """Parse after blanking macro uses the grammar cannot model (see security_c_macros).

    The blanking keeps every offset, so node positions are positions in ``source``.
    ``macros`` is an optional (object_like, function_like) pair of extra attribute
    macro names, typically harvested from the project's own headers.
    """
    language = tree_sitter_cpp if suffix in {".cc", ".cpp", ".cxx", ".hpp"} else tree_sitter_c
    parser = Parser(Language(language.language()))
    object_like, function_like = macros or ((), ())
    choose: dict[int, int] = {}
    text = neutralize(source, object_like, function_like, choose)
    root = parser.parse(text.encode()).root_node
    if not root.has_error:
        return root
    # Keeping the first arm of a split #if is right for one configuration and wrong
    # for another (a newlib-only declaration form); let the grammar decide, one
    # group at a time, and keep a different arm only when it parses better.
    errors = _error_count(root)
    groups = structural_groups(source, object_like, function_like)
    for number, (arms, start, end) in enumerate(groups):
        if len(arms) < 2 or not _has_error_between(root, arms[0][0], end):
            continue
        for arm in range(1, len(arms)):
            trial = {**choose, number: arm}
            candidate = parser.parse(neutralize(source, object_like, function_like, trial).encode()).root_node
            count = _error_count(candidate)
            if count < errors:
                choose, root, errors = trial, candidate, count
        if not errors:
            break
    return root


# [maint-note:5e14247fba16]
# 维护导航：_error_nodes；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 分支条件：not node.has_error。
# 分支条件：node.type == 'ERROR' or node.is_missing。
# 遍历边界：child ← node.children。
# 返回路径：无返回值。
# 直接调用：_error_nodes。
# 生成器交付：node。
# 生成器交付：_error_nodes(child)。
# [/maint-note:5e14247fba16]
def _error_nodes(node):
    if not node.has_error:
        return
    if node.type == "ERROR" or node.is_missing:
        yield node
        return
    for child in node.children:
        yield from _error_nodes(child)


# [maint-note:c460200fe229]
# 维护导航：_error_count；下列为本函数源码中的接口与路径索引。
# 输入 root：类型由调用方与函数体约束。
# 返回契约：int。
# 返回路径：sum((1 for _ in _error_nodes(root)))。
# 直接调用：sum。
# 直接调用：_error_nodes。
# 集合派生：_ 来自 _error_nodes(root)。
# [/maint-note:c460200fe229]
def _error_count(root) -> int:
    return sum(1 for _ in _error_nodes(root))


# [maint-note:ee457e966f85]
# 维护导航：_has_error_between；下列为本函数源码中的接口与路径索引。
# 输入 root：类型由调用方与函数体约束。
# 输入 start：int。
# 输入 end：int。
# 返回契约：bool。
# 返回路径：any((start <= node.start_byte <= end for node in _error_nodes(root)))。
# 直接调用：any。
# 直接调用：_error_nodes。
# 集合派生：node 来自 _error_nodes(root)。
# [/maint-note:ee457e966f85]
def _has_error_between(root, start: int, end: int) -> bool:
    return any(start <= node.start_byte <= end for node in _error_nodes(root))


# [maint-note:b86f0d7db1e6]
# 维护导航：walk；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 遍历边界：child ← node.named_children。
# 直接调用：walk。
# 生成器交付：node。
# 生成器交付：walk(child)。
# [/maint-note:b86f0d7db1e6]
def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


# [maint-note:ce3da36f747a]
# 维护导航：text；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 返回路径：node.text.decode('utf-8', errors='replace') if node is not None else ''。
# 直接调用：node.text.decode。
# 表达式分流：node is not None；真值与假值分支分别计算，不同时执行。
# [/maint-note:ce3da36f747a]
def text(node):
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


# [maint-note:c3ae1f88ac04]
# 维护导航：declared_name；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 遍历边界：node is not None and node.type not in {'identifier', 'field_identifier'}。
# 返回路径：text(node)。
# 直接调用：node.child_by_field_name。
# 直接调用：text。
# [/maint-note:c3ae1f88ac04]
def declared_name(node):
    while node is not None and node.type not in {"identifier", "field_identifier"}:
        node = node.child_by_field_name("declarator")
    return text(node)


# [maint-note:9d05114ba577]
# 维护导航：call_parts；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 返回路径：(text(node.child_by_field_name('function')).removeprefix('std::'), list(node.child_by_field_name('arguments').named_children))。
# 直接调用：text(node.child_by_field_name('function')).removeprefix。
# 直接调用：text。
# 直接调用：node.child_by_field_name。
# 直接调用：list。
# [/maint-note:9d05114ba577]
def call_parts(node):
    return text(node.child_by_field_name("function")).removeprefix("std::"), list(
        node.child_by_field_name("arguments").named_children)


# [maint-note:f32ccdac2395]
# 维护导航：identifiers；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 返回路径：{text(n) for n in walk(node) if n.type == 'identifier'} if node is not None else set()。
# 直接调用：text。
# 直接调用：walk。
# 直接调用：set。
# 表达式分流：node is not None；真值与假值分支分别计算，不同时执行。
# 集合派生：n 来自 walk(node)；过滤 n.type == 'identifier'。
# [/maint-note:f32ccdac2395]
def identifiers(node):
    return {text(n) for n in walk(node) if n.type == "identifier"} if node is not None else set()


# [maint-note:8a33ddce6b1b]
# 维护导航：_command_flows；下列为本函数源码中的接口与路径索引。
# 输入 function：类型由调用方与函数体约束。
# 分支条件：node.type == 'parameter_declaration' and re.search('\\b(?:char|wchar_t)\\b', text(node))。
# 分支条件：name and ('*' in text(node) or '[' in text(node))。
# 分支条件：node is None or node.type in {'string_literal', 'char_literal', 'number_literal', 'sizeof_expression'}。
# 分支条件：node.type == 'call_expression'。
# 分支条件：name in {'getenv', 'gets'}。
# 分支条件：name in {'atoi', 'atol', 'strtol', 'strtoul', 'strlen'}。
# 分支条件：node.type in {'init_declarator', 'assignment_expression'}。
# 分支条件：name。
# 分支条件：value。
# 分支条件：name in {'system', 'popen', '_popen'} and args。
# 分支条件：name in {'fgets', 'gets', 'read', 'recv', 'scanf', 'fscanf', 'sscanf'} and args。
# 分支条件：name in {'scanf', 'fscanf', 'sscanf'}。
# 分支条件：name in {'strcpy', 'strncpy', 'strcat', 'strncat', 'memcpy', 'memmove', 'sprintf', 'snprintf'} and len(args) >= 2。
# 分支条件：name in {'sprintf', 'snprintf'}。
# 分支条件：name not in {'strcat', 'strncat'}。
# 分支条件：node.type == 'if_statement'。
# 分支条件：alternate is not None。
# 遍历边界：node ← walk(function.child_by_field_name('declarator'))。
# 遍历边界：arg ← destinations。
# 遍历边界：identifier ← identifiers(arg)。
# 遍历边界：child ← node.named_children。
# 返回路径：None。
# 返回路径：f'{name} at line {node.start_point.row + 1}'。
# 返回路径：next((tainted[n] for n in sorted(identifiers(node)) if n in tainted), None)。
# 返回路径：无返回值。
# 写入目标：tainted[name]。
# 写入目标：tainted[identifier]。
# 写入目标：tainted[destination]。
# 直接调用：walk。
# 直接调用：function.child_by_field_name。
# 直接调用：re.search。
# 直接调用：text。
# 直接调用：declared_name。
# 直接调用：node.child_by_field_name。
# 直接调用：call_parts。
# 直接调用：next。
# 直接调用：sorted。
# 直接调用：identifiers。
# 直接调用：origin。
# 直接调用：tainted.pop。
# 直接调用：len。
# 直接调用：visit。
# 直接调用：dict。
# 直接调用：tainted.clear。
# 直接调用：tainted.update。
# 生成器交付：visit(function.child_by_field_name('body'))。
# [/maint-note:8a33ddce6b1b]
def _command_flows(function):
    tainted = {}
    # String parameters are caller-controlled inputs, not proven external inputs.
    for node in walk(function.child_by_field_name("declarator")):
        if node.type == "parameter_declaration" and re.search(r"\b(?:char|wchar_t)\b", text(node)):
            name = declared_name(node.child_by_field_name("declarator"))
            if name and ("*" in text(node) or "[" in text(node)):
                tainted[name] = f"string parameter {name} at line {node.start_point.row + 1}"

    # [maint-note:a8cb8b7273dc]
    # 维护导航：origin；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 分支条件：node is None or node.type in {'string_literal', 'char_literal', 'number_literal', 'sizeof_expression'}。
    # 分支条件：node.type == 'call_expression'。
    # 分支条件：name in {'getenv', 'gets'}。
    # 分支条件：name in {'atoi', 'atol', 'strtol', 'strtoul', 'strlen'}。
    # 返回路径：None。
    # 返回路径：f'{name} at line {node.start_point.row + 1}'。
    # 返回路径：next((tainted[n] for n in sorted(identifiers(node)) if n in tainted), None)。
    # 直接调用：call_parts。
    # 直接调用：next。
    # 直接调用：sorted。
    # 直接调用：identifiers。
    # 集合派生：n 来自 sorted(identifiers(node))；过滤 n in tainted。
    # [/maint-note:a8cb8b7273dc]
    def origin(node):
        if node is None or node.type in {"string_literal", "char_literal", "number_literal", "sizeof_expression"}:
            return None
        if node.type == "call_expression":
            name, _ = call_parts(node)
            if name in {"getenv", "gets"}:
                return f"{name} at line {node.start_point.row + 1}"
            # Numeric conversion is not a string injection path.
            if name in {"atoi", "atol", "strtol", "strtoul", "strlen"}:
                return None
        return next((tainted[n] for n in sorted(identifiers(node)) if n in tainted), None)

    # [maint-note:22223bfbbfb7]
    # 维护导航：visit；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 分支条件：node.type in {'init_declarator', 'assignment_expression'}。
    # 分支条件：name。
    # 分支条件：value。
    # 分支条件：node.type == 'call_expression'。
    # 分支条件：name in {'system', 'popen', '_popen'} and args。
    # 分支条件：name in {'fgets', 'gets', 'read', 'recv', 'scanf', 'fscanf', 'sscanf'} and args。
    # 分支条件：name in {'scanf', 'fscanf', 'sscanf'}。
    # 分支条件：name in {'strcpy', 'strncpy', 'strcat', 'strncat', 'memcpy', 'memmove', 'sprintf', 'snprintf'} and len(args) >= 2。
    # 分支条件：name in {'sprintf', 'snprintf'}。
    # 分支条件：name not in {'strcat', 'strncat'}。
    # 分支条件：node.type == 'if_statement'。
    # 分支条件：alternate is not None。
    # 遍历边界：arg ← destinations。
    # 遍历边界：identifier ← identifiers(arg)。
    # 遍历边界：child ← node.named_children。
    # 返回路径：无返回值。
    # 写入目标：tainted[name]。
    # 写入目标：tainted[identifier]。
    # 写入目标：tainted[destination]。
    # 直接调用：node.child_by_field_name。
    # 直接调用：declared_name。
    # 直接调用：origin。
    # 直接调用：tainted.pop。
    # 直接调用：call_parts。
    # 直接调用：text。
    # 直接调用：len。
    # 直接调用：re.search。
    # 直接调用：identifiers。
    # 直接调用：next。
    # 直接调用：visit。
    # 直接调用：dict。
    # 直接调用：tainted.clear。
    # 直接调用：tainted.update。
    # 表达式分流：node.type == 'init_declarator'；真值与假值分支分别计算，不同时执行。
    # 生成器交付：(node, f'{value} -> {text(args[0])} -> {name} (shell interpretation)')。
    # 表达式分流：name in {'read', 'recv'}；真值与假值分支分别计算，不同时执行。
    # 表达式分流：name == 'scanf'；真值与假值分支分别计算，不同时执行。
    # 表达式分流：len(args) > index and re.search('%[^%]*[s\\[]', text(args[index]))；真值与假值分支分别计算，不同时执行。
    # 表达式分流：name == 'sprintf'；真值与假值分支分别计算，不同时执行。
    # 表达式分流：len(args) > fmt and re.search('%(?:[-+ #0\\d.*hlzjt]+)?s', text(args[fmt]))；真值与假值分支分别计算，不同时执行。
    # 集合派生：a 来自 values；过滤 (v := origin(a))。
    # 生成器交付：visit(node.child_by_field_name('condition'))。
    # 生成器交付：visit(node.child_by_field_name('consequence'))。
    # 生成器交付：visit(alternate)。
    # 生成器交付：visit(child)。
    # [/maint-note:22223bfbbfb7]
    def visit(node):
        if node.type in {"init_declarator", "assignment_expression"}:
            lhs = node.child_by_field_name("declarator" if node.type == "init_declarator" else "left")
            rhs = node.child_by_field_name("value" if node.type == "init_declarator" else "right")
            name = declared_name(lhs)
            value = origin(rhs)
            if name:
                if value:
                    tainted[name] = value
                else:
                    tainted.pop(name, None)
        if node.type == "call_expression":
            name, args = call_parts(node)
            if name in {"system", "popen", "_popen"} and args:
                value = origin(args[0])
                if value:
                    yield node, f"{value} -> {text(args[0])} -> {name} (shell interpretation)"
            if name in {"fgets", "gets", "read", "recv", "scanf", "fscanf", "sscanf"} and args:
                index = 1 if name in {"read", "recv"} else 0
                destinations = args[index:index + 1]
                if name in {"scanf", "fscanf", "sscanf"}:
                    index = 0 if name == "scanf" else 1
                    # Numeric conversions do not introduce shell metacharacters.
                    destinations = args[index + 1:] if len(args) > index and re.search(r"%[^%]*[s\[]", text(args[index])) else []
                for arg in destinations:
                    for identifier in identifiers(arg):
                        tainted[identifier] = f"{name} input at line {node.start_point.row + 1}"
            if name in {"strcpy", "strncpy", "strcat", "strncat", "memcpy", "memmove", "sprintf", "snprintf"} and len(args) >= 2:
                values = args[1:2]
                if name in {"sprintf", "snprintf"}:
                    fmt = 1 if name == "sprintf" else 2
                    values = args[fmt + 1:] if len(args) > fmt and re.search(r"%(?:[-+ #0\d.*hlzjt]+)?s", text(args[fmt])) else []
                value = next((v for a in values if (v := origin(a))), None)
                destination = text(args[0])
                if value:
                    tainted[destination] = value
                elif name not in {"strcat", "strncat"}:
                    tainted.pop(destination, None)
        if node.type == "if_statement":
            # Join branch states: a constant on only one branch cannot clear taint.
            yield from visit(node.child_by_field_name("condition"))
            before = dict(tainted)
            yield from visit(node.child_by_field_name("consequence"))
            after = dict(tainted)
            tainted.clear()
            tainted.update(before)
            alternate = node.child_by_field_name("alternative")
            if alternate is not None:
                yield from visit(alternate)
            tainted.update(after)
            return
        for child in node.named_children:
            yield from visit(child)

    yield from visit(function.child_by_field_name("body"))


# [maint-note:be70e25e3b40]
# 维护导航：_race_accesses；下列为本函数源码中的接口与路径索引。
# 输入 function：类型由调用方与函数体约束。
# 输入 globals_：类型由调用方与函数体约束。
# 分支条件：n.type == 'declaration'。
# 分支条件：node.type == 'if_statement'。
# 分支条件：guarded。
# 分支条件：alternate is not None。
# 分支条件：node.type == 'call_expression'。
# 分支条件：name in {'pthread_mutex_lock', 'pthread_mutex_unlock'} and args。
# 分支条件：name.endswith('unlock')。
# 分支条件：not conditional。
# 分支条件：node.type == 'declaration' and re.search('std::(?:lock_guard|scoped_lock)\\b', text(node)) and (not conditional)。
# 分支条件：match。
# 分支条件：node.type == 'identifier' and text(node) in shared。
# 分支条件：parent.type != 'pointer_expression' or not text(parent).startswith('&')。
# 分支条件：node.type == 'compound_statement'。
# 分支条件：child.type == 'declaration' and re.search('std::(?:lock_guard|scoped_lock)\\b', text(child))。
# 分支条件：match and match[1] not in before。
# 遍历边界：n ← walk(function)。
# 遍历边界：child ← node.named_children。
# 返回路径：无返回值。
# 返回路径：list(visit(function.child_by_field_name('body')))。
# 直接调用：declared_name。
# 直接调用：n.child_by_field_name。
# 直接调用：walk。
# 直接调用：local.update。
# 直接调用：n.children_by_field_name。
# 直接调用：set。
# 直接调用：re.sub('\\s', '', text(node.child_by_field_name('condition'))).strip。
# 直接调用：re.sub。
# 直接调用：text。
# 直接调用：node.child_by_field_name。
# 直接调用：re.fullmatch。
# 直接调用：locks.add。
# 直接调用：visit。
# 直接调用：locks.clear。
# 直接调用：locks.update。
# 直接调用：call_parts。
# 直接调用：name.endswith。
# 直接调用：locks.discard。
# 直接调用：re.search。
# 直接调用：parent.child_by_field_name。
# 直接调用：text(parent).startswith。
# 直接调用：list。
# 直接调用：function.child_by_field_name。
# 集合派生：n 来自 walk(function)；过滤 n.type in {'parameter_declaration', 'init_declarator'}。
# 集合派生：c 来自 n.children_by_field_name('declarator')。
# [/maint-note:be70e25e3b40]
def _race_accesses(function, globals_):
    """Direct accesses and lexical locksets; shadowed and atomic objects excluded."""
    local = {declared_name(n.child_by_field_name("declarator")) for n in walk(function)
             if n.type in {"parameter_declaration", "init_declarator"}}
    for n in walk(function):
        if n.type == "declaration":
            local.update(declared_name(c) for c in n.children_by_field_name("declarator"))
    shared = globals_ - local
    locks = set()

    # [maint-note:f6a1ed1cab98]
    # 维护导航：visit；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 输入 conditional：类型由调用方与函数体约束；缺省表达式 False。
    # 分支条件：node.type == 'if_statement'。
    # 分支条件：guarded。
    # 分支条件：alternate is not None。
    # 分支条件：node.type == 'call_expression'。
    # 分支条件：name in {'pthread_mutex_lock', 'pthread_mutex_unlock'} and args。
    # 分支条件：name.endswith('unlock')。
    # 分支条件：not conditional。
    # 分支条件：node.type == 'declaration' and re.search('std::(?:lock_guard|scoped_lock)\\b', text(node)) and (not conditional)。
    # 分支条件：match。
    # 分支条件：node.type == 'identifier' and text(node) in shared。
    # 分支条件：parent.type != 'pointer_expression' or not text(parent).startswith('&')。
    # 分支条件：node.type == 'compound_statement'。
    # 分支条件：child.type == 'declaration' and re.search('std::(?:lock_guard|scoped_lock)\\b', text(child))。
    # 分支条件：match and match[1] not in before。
    # 遍历边界：child ← node.named_children。
    # 返回路径：无返回值。
    # 直接调用：set。
    # 直接调用：re.sub('\\s', '', text(node.child_by_field_name('condition'))).strip。
    # 直接调用：re.sub。
    # 直接调用：text。
    # 直接调用：node.child_by_field_name。
    # 直接调用：re.fullmatch。
    # 直接调用：locks.add。
    # 直接调用：visit。
    # 直接调用：locks.clear。
    # 直接调用：locks.update。
    # 直接调用：call_parts。
    # 直接调用：name.endswith。
    # 直接调用：locks.discard。
    # 直接调用：re.search。
    # 直接调用：parent.child_by_field_name。
    # 直接调用：text(parent).startswith。
    # 生成器交付：visit(node.child_by_field_name('consequence'), conditional)。
    # 生成器交付：visit(alternate, True)。
    # 生成器交付：{'node': node, 'variable': text(node), 'write': write, 'locks': set(locks)}。
    # 生成器交付：visit(child, conditional)。
    # [/maint-note:f6a1ed1cab98]
    def visit(node, conditional=False):
        before = set(locks)
        if node.type == "if_statement":
            condition = re.sub(r"\s", "", text(node.child_by_field_name("condition"))).strip("()")
            guarded = re.fullmatch(r"pthread_mutex_lock\(&([A-Za-z_]\w*)\)==0", condition)
            if guarded:
                locks.add(guarded[1])
                yield from visit(node.child_by_field_name("consequence"), conditional)
                locks.clear()
                locks.update(before)
                alternate = node.child_by_field_name("alternative")
                if alternate is not None:
                    yield from visit(alternate, True)
                return
        conditional = conditional or node.type in {"if_statement", "switch_statement", "conditional_expression"}
        if node.type == "call_expression":
            name, args = call_parts(node)
            if name in {"pthread_mutex_lock", "pthread_mutex_unlock"} and args:
                lock = re.sub(r"[\s&()]", "", text(args[0]))
                if name.endswith("unlock"):
                    locks.discard(lock)
                elif not conditional:
                    locks.add(lock)
            # Address-taking is not an access; unknown helpers need deeper analysis.
            return
        if node.type == "declaration" and re.search(r"std::(?:lock_guard|scoped_lock)\b", text(node)) and not conditional:
            match = re.search(r"[({]\s*(\w+)\s*[)}]", text(node))
            if match:
                locks.add(match[1])
            return
        if node.type == "identifier" and text(node) in shared:
            parent = node.parent
            write = parent.type == "update_expression" or (
                parent.type == "assignment_expression" and parent.child_by_field_name("left") == node)
            if parent.type != "pointer_expression" or not text(parent).startswith("&"):
                yield {"node": node, "variable": text(node), "write": write, "locks": set(locks)}
        for child in node.named_children:
            yield from visit(child, conditional)
        if node.type == "compound_statement":
            # Only RAII guards end at a C++ block boundary; pthread locks persist.
            for child in node.named_children:
                if child.type == "declaration" and re.search(r"std::(?:lock_guard|scoped_lock)\b", text(child)):
                    match = re.search(r"[({]\s*(\w+)\s*[)}]", text(child))
                    if match and match[1] not in before:
                        locks.discard(match[1])

    return list(visit(function.child_by_field_name("body")))


# [maint-note:2f04d44005f7]
# 维护导航：_races；下列为本函数源码中的接口与路径索引。
# 输入 root：类型由调用方与函数体约束。
# 输入 functions：类型由调用方与函数体约束。
# 分支条件：node.type == 'declaration' and (not re.search('\\b(?:_Atomic|atomic|thread_local|_Thread_local|__thread|const)\\b', text(node)))。
# 分支条件：not any((n.type in {'pointer_declarator', 'array_declarator', 'function_declarator'} for n in walk(declaration)))。
# 分支条件：a['variable'] != b['variable'] or not (a['write'] or b['write']) or a['locks'] & b['locks']。
# 分支条件：key not in seen。
# 分支条件：node.type == 'call_expression'。
# 分支条件：call == 'pthread_create' and len(args) >= 3。
# 分支条件：call == 'pthread_join' and args。
# 分支条件：call.endswith('.join')。
# 分支条件：node.type == 'declaration' and text(node.child_by_field_name('type')) in {'std::thread', 'std::jthread'}。
# 分支条件：match。
# 分支条件：kind == 'start'。
# 分支条件：kind == 'join'。
# 遍历边界：node ← root.named_children。
# 遍历边界：declaration ← node.children_by_field_name('declarator')。
# 遍历边界：a ← left。
# 遍历边界：b ← right。
# 遍历边界：(name, fn) ← functions.items()。
# 遍历边界：node ← walk(fn.child_by_field_name('body'))。
# 遍历边界：(_, kind, event) ← sorted(events, key=lambda row: row[0])。
# 遍历边界：other ← active.values()。
# 写入目标：active[handle]。
# 直接调用：set。
# 直接调用：re.search。
# 直接调用：text。
# 直接调用：node.children_by_field_name。
# 直接调用：any。
# 直接调用：walk。
# 直接调用：globals_.add。
# 直接调用：declared_name。
# 直接调用：_race_accesses。
# 直接调用：functions.items。
# 直接调用：min。
# 直接调用：max。
# 直接调用：seen.add。
# 直接调用：fn.child_by_field_name。
# 直接调用：call_parts。
# 直接调用：len。
# 直接调用：events.append。
# 直接调用：text(args[0]).strip。
# 直接调用：text(args[2]).strip。
# 直接调用：call.endswith。
# 直接调用：node.child_by_field_name。
# 直接调用：sorted。
# 直接调用：accesses.get。
# 直接调用：active.values。
# 直接调用：compare。
# 直接调用：active.pop。
# 集合派生：n 来自 walk(declaration)。
# 集合派生：(name, fn) 来自 functions.items()。
# 集合派生：a 来自 accesses[name]。
# 生成器交付：compare(other, current)。
# 生成器交付：compare(other, [event])。
# [/maint-note:2f04d44005f7]
def _races(root, functions):
    globals_ = set()
    for node in root.named_children:
        if node.type == "declaration" and not re.search(r"\b(?:_Atomic|atomic|thread_local|_Thread_local|__thread|const)\b", text(node)):
            for declaration in node.children_by_field_name("declarator"):
                # Pointer/array aliases need a different analysis.
                if not any(n.type in {"pointer_declarator", "array_declarator", "function_declarator"} for n in walk(declaration)):
                    globals_.add(declared_name(declaration))
    accesses = {name: _race_accesses(fn, globals_) for name, fn in functions.items()}
    seen = set()

    # [maint-note:811e28a0ade8]
    # 维护导航：compare；下列为本函数源码中的接口与路径索引。
    # 输入 left：类型由调用方与函数体约束。
    # 输入 right：类型由调用方与函数体约束。
    # 分支条件：a['variable'] != b['variable'] or not (a['write'] or b['write']) or a['locks'] & b['locks']。
    # 分支条件：key not in seen。
    # 遍历边界：a ← left。
    # 遍历边界：b ← right。
    # 直接调用：min。
    # 直接调用：max。
    # 直接调用：seen.add。
    # 生成器交付：(a['node'], f"Overlapping threads access global {a['variable']}; conflicting lines {a['node'].start_point.row + 1} and {b['node'].start_point.row + 1}; at least one write and no common recognized mutex")。
    # [/maint-note:811e28a0ade8]
    def compare(left, right):
        for a in left:
            for b in right:
                if a["variable"] != b["variable"] or not (a["write"] or b["write"]) or a["locks"] & b["locks"]:
                    continue
                key = (a["variable"], min(a["node"].start_byte, b["node"].start_byte), max(a["node"].start_byte, b["node"].start_byte))
                if key not in seen:
                    seen.add(key)
                    yield a["node"], (f"Overlapping threads access global {a['variable']}; "
                        f"conflicting lines {a['node'].start_point.row + 1} and {b['node'].start_point.row + 1}; "
                        "at least one write and no common recognized mutex")

    for name, fn in functions.items():
        active = {}
        events = [(a["node"].start_byte, "access", a) for a in accesses[name]]
        for node in walk(fn.child_by_field_name("body")):
            if node.type == "call_expression":
                call, args = call_parts(node)
                if call == "pthread_create" and len(args) >= 3:
                    events.append((node.start_byte, "start", (text(args[0]).strip("& "), text(args[2]).strip("& "))))
                elif call == "pthread_join" and args:
                    events.append((node.start_byte, "join", text(args[0])))
                elif call.endswith(".join"):
                    events.append((node.start_byte, "join", call[:-5]))
            elif node.type == "declaration" and text(node.child_by_field_name("type")) in {"std::thread", "std::jthread"}:
                declaration = node.child_by_field_name("declarator")
                match = re.search(r"[({]\s*&?(\w+)\s*[,)}]", text(declaration))
                if match:
                    events.append((node.start_byte, "start", (declared_name(declaration), match[1])))
        for _, kind, event in sorted(events, key=lambda row: row[0]):
            if kind == "start":
                handle, worker = event
                current = accesses.get(worker, [])
                for other in active.values():
                    yield from compare(other, current)
                active[handle] = current
            elif kind == "join":
                active.pop(event, None)
            else:
                for other in active.values():
                    yield from compare(other, [event])


_INT_EXPR = re.compile(r"^[\d\s+\-*/()]+$")


# [maint-note:1de75bc3fddb]
# 维护导航：_int_value；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 分支条件：not expression or not _INT_EXPR.match(expression)。
# 返回路径：None。
# 返回路径：int(value) if isinstance(value, (int, float)) and value >= 0 else None。
# 捕获异常：(SyntaxError, ZeroDivisionError, TypeError)。
# 直接调用：text(node).strip().replace('(size_t)', '').replace。
# 直接调用：text(node).strip().replace。
# 直接调用：text(node).strip。
# 直接调用：text。
# 直接调用：_INT_EXPR.match。
# 直接调用：eval。
# 直接调用：isinstance。
# 直接调用：int。
# 表达式分流：isinstance(value, (int, float)) and value >= 0；真值与假值分支分别计算，不同时执行。
# [/maint-note:1de75bc3fddb]
def _int_value(node):
    """Value of a constant integer expression such as 100-1, else None."""
    expression = text(node).strip().replace("(size_t)", "").replace("(int)", "")
    if not expression or not _INT_EXPR.match(expression):
        return None
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # digits and + - * / ( ) only
    except (SyntaxError, ZeroDivisionError, TypeError):
        return None
    return int(value) if isinstance(value, (int, float)) and value >= 0 else None


# [maint-note:867407fe8714]
# 维护导航：_literal_length；下列为本函数源码中的接口与路径索引。
# 输入 node：类型由调用方与函数体约束。
# 分支条件：node is None or node.type != 'string_literal'。
# 返回路径：None。
# 返回路径：len(re.sub('\\\\(x[0-9a-fA-F]+|[0-7]{1,3}|.)', 'x', body))。
# 直接调用：text。
# 直接调用：len。
# 直接调用：re.sub。
# [/maint-note:867407fe8714]
def _literal_length(node):
    if node is None or node.type != "string_literal":
        return None
    body = text(node)[1:-1]
    return len(re.sub(r"\\(x[0-9a-fA-F]+|[0-7]{1,3}|.)", "x", body))


# [maint-note:0409581704a3]
# 容量与字符串长度分别跟踪：容量包含可写空间，strlen 不包含结尾 NUL。
# strcpy 的写入需求为源长度加一；strcat 还需加上已有前缀长度。
# strncpy 的长度参数限定写入字节数，不能假定总会自动追加终止符。
# 局部别名解析需要记录已访问名称，防止循环别名导致无限循环。
# 跨函数只传播已知容量和长度；没有证据时保留未知而非猜测。
# 递归传播有深度限制，不是任意跨文件或无界调用图分析。
# 调用处归因让缺陷落到传入错误容量的业务函数，便于逐函数核查。
# 教学开关 C 关闭时保留调用处归因缺口；完整参考实现启用该规则。
# 规则按语法顺序更新事实，存在路径不敏感限制，结果仍需审查。
# 维护导航：_string_writes；下列为本函数源码中的接口与路径索引。
# 输入 function：类型由调用方与函数体约束。
# 输入 functions：类型由调用方与函数体约束。
# 输入 facts：类型由调用方与函数体约束；缺省表达式 None。
# 输入 depth：类型由调用方与函数体约束；缺省表达式 0。
# 分支条件：value is not None。
# 分支条件：node.type == 'call_expression'。
# 分支条件：name == 'strlen' and arguments。
# 分支条件：node.type == 'binary_expression' and text(node.child_by_field_name('operator')) == '+'。
# 分支条件：node.type == 'sizeof_expression'。
# 分支条件：target is not None。
# 分支条件：node.type == 'declaration' and 'char' in text(node.child_by_field_name('type'))。
# 分支条件：target is not None and target.type == 'array_declarator'。
# 分支条件：size is not None and _int_value(size) is not None。
# 分支条件：value is not None and value.type == 'string_literal'。
# 分支条件：target is not None and target.type == 'pointer_declarator' and (value is not None) and (value.type == 'identifier')。
# 分支条件：node.type == 'assignment_expression' and text(node.child_by_field_name('operator')) == '='。
# 分支条件：left.type == 'identifier' and right.type == 'identifier' and (buffer(right) in capacity or text(right) in alias)。
# 分支条件：left.type == 'subscript_expression' and text(right) in {"'\\0'", '0', "'\\x00'"}。
# 分支条件：index is not None。
# 分支条件：name in functions and depth < 3 and (functions[name] is not function)。
# 分支条件：source in capacity。
# 分支条件：strlen_of(argument) is not None。
# 分支条件：inherited['capacity'] or inherited['length']。
# 分支条件：not switch_off('C')。
# 分支条件：not arguments。
# 分支条件：name == 'strcpy' and len(arguments) >= 2。
# 分支条件：name == 'strcat' and len(arguments) >= 2。
# 分支条件：name == 'strncat' and len(arguments) >= 3。
# 分支条件：name == 'strncpy' and len(arguments) >= 3。
# 分支条件：name == 'snprintf' and len(arguments) >= 2。
# 分支条件：room is not None and written is not None and (written > room)。
# 分支条件：name in {'strcpy', 'strncpy'} and len(arguments) >= 2 and (strlen_of(arguments[1]) is not None)。
# 遍历边界：name in alias and name not in seen。
# 遍历边界：node ← walk(function)。
# 遍历边界：declarator ← node.named_children。
# 遍历边界：(parameter, argument) ← zip(parameters, arguments)。
# 遍历边界：(_, evidence) ← _string_writes(functions[name], functions, inherited, depth + 1)。
# 返回路径：name。
# 返回路径：literal if literal is not None else length.get(buffer(node))。
# 返回路径：value。
# 返回路径：strlen_of(arguments[0])。
# 返回路径：left + right if left is not None and right is not None else None。
# 返回路径：capacity.get(buffer(inner))。
# 返回路径：None。
# 返回路径：results。
# 写入目标：capacity[name]。
# 写入目标：length[name]。
# 写入目标：alias[declared_name(target)]。
# 写入目标：alias[text(left)]。
# 写入目标：length[buffer(left.child_by_field_name('argument'))]。
# 写入目标：inherited['capacity'][parameter]。
# 写入目标：inherited['length'][parameter]。
# 写入目标：length[destination]。
# 直接调用：dict(facts or {}).get。
# 直接调用：dict。
# 直接调用：(facts or {}).get。
# 直接调用：text(node).strip。
# 直接调用：text。
# 直接调用：set。
# 直接调用：seen.add。
# 直接调用：_literal_length。
# 直接调用：length.get。
# 直接调用：buffer。
# 直接调用：_int_value。
# 直接调用：call_parts。
# 直接调用：strlen_of。
# 直接调用：node.child_by_field_name。
# 直接调用：size_of。
# 直接调用：capacity.get。
# 直接调用：walk。
# 直接调用：declarator.child_by_field_name。
# 直接调用：target.child_by_field_name。
# 直接调用：declared_name。
# 直接调用：capacity.setdefault。
# 直接调用：left.child_by_field_name。
# 直接调用：functions[name].child_by_field_name。
# 直接调用：zip。
# 直接调用：_string_writes。
# 直接调用：switch_off。
# 直接调用：results.append。
# 直接调用：len。
# 直接调用：min。
# 表达式分流：declarator.type == 'init_declarator'；真值与假值分支分别计算，不同时执行。
# 集合派生：p 来自 walk(functions[name].child_by_field_name('declarator'))；过滤 p.type == 'parameter_declaration'。
# 集合派生：item 来自 evidence。
# 表达式分流：source is None；真值与假值分支分别计算，不同时执行。
# 表达式分流：prefix is None or source is None；真值与假值分支分别计算，不同时执行。
# 表达式分流：source is not None or limit is not None；真值与假值分支分别计算，不同时执行。
# 集合派生：v 来自 (source, limit)；过滤 v is not None。
# 表达式分流：prefix is None or appended is None；真值与假值分支分别计算，不同时执行。
# [/maint-note:0409581704a3]
def _string_writes(function, functions, facts=None, depth=0):
    """Track char-array capacities, aliases and known string lengths in one function
    (path-insensitive, last write wins) and flag string writes that cannot fit.
    Facts follow buffer arguments one call level into functions of the same file."""
    capacity, length, alias = dict(facts or {}).get("capacity", {}), {}, {}
    capacity, length, alias = dict(capacity), dict((facts or {}).get("length", {})), dict((facts or {}).get("alias", {}))

    # [maint-note:e76e210e5bae]
    # 维护导航：buffer；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 遍历边界：name in alias and name not in seen。
    # 返回路径：name。
    # 直接调用：text(node).strip。
    # 直接调用：text。
    # 直接调用：set。
    # 直接调用：seen.add。
    # [/maint-note:e76e210e5bae]
    def buffer(node):
        name = text(node).strip()
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    # [maint-note:d906200f6785]
    # 维护导航：strlen_of；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 返回路径：literal if literal is not None else length.get(buffer(node))。
    # 直接调用：_literal_length。
    # 直接调用：length.get。
    # 直接调用：buffer。
    # 表达式分流：literal is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:d906200f6785]
    def strlen_of(node):
        literal = _literal_length(node)
        return literal if literal is not None else length.get(buffer(node))

    # [maint-note:da9fe875514e]
    # 维护导航：size_of；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 分支条件：value is not None。
    # 分支条件：node.type == 'call_expression'。
    # 分支条件：name == 'strlen' and arguments。
    # 分支条件：node.type == 'binary_expression' and text(node.child_by_field_name('operator')) == '+'。
    # 分支条件：node.type == 'sizeof_expression'。
    # 分支条件：target is not None。
    # 返回路径：value。
    # 返回路径：strlen_of(arguments[0])。
    # 返回路径：left + right if left is not None and right is not None else None。
    # 返回路径：capacity.get(buffer(inner))。
    # 返回路径：None。
    # 直接调用：_int_value。
    # 直接调用：call_parts。
    # 直接调用：strlen_of。
    # 直接调用：text。
    # 直接调用：node.child_by_field_name。
    # 直接调用：size_of。
    # 直接调用：capacity.get。
    # 直接调用：buffer。
    # 表达式分流：left is not None and right is not None；真值与假值分支分别计算，不同时执行。
    # 表达式分流：node.named_children；真值与假值分支分别计算，不同时执行。
    # 表达式分流：target.type == 'parenthesized_expression' and target.named_children；真值与假值分支分别计算，不同时执行。
    # [/maint-note:da9fe875514e]
    def size_of(node):
        value = _int_value(node)
        if value is not None:
            return value
        if node.type == "call_expression":
            name, arguments = call_parts(node)
            if name == "strlen" and arguments:
                return strlen_of(arguments[0])
        if node.type == "binary_expression" and text(node.child_by_field_name("operator")) == "+":
            left, right = size_of(node.child_by_field_name("left")), size_of(node.child_by_field_name("right"))
            return left + right if left is not None and right is not None else None
        if node.type == "sizeof_expression":
            target = node.child_by_field_name("value") or (node.named_children[0] if node.named_children else None)
            if target is not None:
                inner = target.named_children[0] if target.type == "parenthesized_expression" and target.named_children else target
                return capacity.get(buffer(inner))
        return None

    results = []
    for node in walk(function):
        if node.type == "declaration" and "char" in text(node.child_by_field_name("type")):
            for declarator in node.named_children:
                value = declarator.child_by_field_name("value") if declarator.type == "init_declarator" else None
                target = declarator.child_by_field_name("declarator") if declarator.type == "init_declarator" else declarator
                if target is not None and target.type == "array_declarator":
                    size = target.child_by_field_name("size")
                    name = declared_name(target)
                    if size is not None and _int_value(size) is not None:
                        capacity[name] = _int_value(size)
                    if value is not None and value.type == "string_literal":
                        length[name] = _literal_length(value)
                        capacity.setdefault(name, length[name] + 1)
                elif target is not None and target.type == "pointer_declarator" and value is not None and value.type == "identifier":
                    alias[declared_name(target)] = text(value)
        elif node.type == "assignment_expression" and text(node.child_by_field_name("operator")) == "=":
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            if left.type == "identifier" and right.type == "identifier" and (buffer(right) in capacity or text(right) in alias):
                alias[text(left)] = text(right)
            elif left.type == "subscript_expression" and text(right) in {"'\\0'", "0", "'\\x00'"}:
                index = _int_value(left.child_by_field_name("index"))
                if index is not None:
                    length[buffer(left.child_by_field_name("argument"))] = index
        elif node.type == "call_expression":
            name, arguments = call_parts(node)
            if name in functions and depth < 3 and functions[name] is not function:
                parameters = [declared_name(p) for p in walk(functions[name].child_by_field_name("declarator"))
                              if p.type == "parameter_declaration"]
                inherited = {"capacity": {}, "length": {}, "alias": {}}
                for parameter, argument in zip(parameters, arguments):
                    source = buffer(argument)
                    if source in capacity:
                        inherited["capacity"][parameter] = capacity[source]
                    if strlen_of(argument) is not None:
                        inherited["length"][parameter] = strlen_of(argument)
                if inherited["capacity"] or inherited["length"]:
                    # Attribute a callee overflow to this call, which supplies the sizes.
                    for _, evidence in _string_writes(functions[name], functions, inherited, depth + 1):
                        if not switch_off("C"):  # Contract-3 switch C (security_exam)
                            results.append((node, [f"via call to {name}(): " + item for item in evidence]))
                continue
            if not arguments:
                continue
            destination = buffer(arguments[0])
            room = capacity.get(destination)
            written, evidence = None, None
            if name == "strcpy" and len(arguments) >= 2:
                source = strlen_of(arguments[1])
                written = None if source is None else source + 1
                evidence = f"strcpy copies a {source}-character string plus NUL"
            elif name == "strcat" and len(arguments) >= 2:
                prefix, source = length.get(destination), strlen_of(arguments[1])
                written = None if prefix is None or source is None else prefix + source + 1
                evidence = f"strcat appends {source} characters after {prefix} plus NUL"
            elif name == "strncat" and len(arguments) >= 3:
                prefix, source, limit = length.get(destination), strlen_of(arguments[1]), size_of(arguments[2])
                appended = min(v for v in (source, limit) if v is not None) if source is not None or limit is not None else None
                written = None if prefix is None or appended is None else prefix + appended + 1
                evidence = f"strncat appends {appended} characters after {prefix} plus NUL"
            elif name == "strncpy" and len(arguments) >= 3:
                written = size_of(arguments[2])
                evidence = f"strncpy writes {written} bytes"
            elif name == "snprintf" and len(arguments) >= 2:
                written = size_of(arguments[1])
                evidence = f"snprintf is allowed to write {written} bytes"
            # I is a deliberate student boundary error: NUL is already in written.
            # Full rules accept equality; the exercise incorrectly demands spare space.
            exact_fit_gap = switch_off("I") and name in {"strcpy", "strcat"} and written == room
            if room is not None and written is not None and (written > room or exact_fit_gap):
                results.append((node, [f"{evidence} into '{destination}' of capacity {room}"]))
            if name in {"strcpy", "strncpy"} and len(arguments) >= 2 and strlen_of(arguments[1]) is not None:
                length[destination] = strlen_of(arguments[1])
    return results


_NULL_TOKENS = {"0", "NULL", "nullptr", "((void *)0)", "(void *)0"}


# [maint-note:47e6c6df123f]
# 维护导航：_unchecked_dereferences；下列为本函数源码中的接口与路径索引。
# 输入 function：类型由调用方与函数体约束。
# 输入 parameter：类型由调用方与函数体约束。
# 分支条件：node.type in {'if_statement', 'conditional_expression', 'binary_expression'} and re.search(f'\\b{re.escape(parameter)}\\b\\s*(==|!=)|(==|!=)\\s*\\b{re.escape(parameter)}\\b|^\\s*!\\s*{re.escape(parameter)}\\b|\\(\\s*!?\\s*{re.escape(paramet。
# 分支条件：node.type == 'pointer_expression' and text(node.child_by_field_name('argument')) == parameter and source.startswith('*')。
# 分支条件：node.type == 'field_expression' and text(node.child_by_field_name('operator')) == '->' and (text(node.child_by_field_name('argument')) == parameter)。
# 分支条件：node.type == 'subscript_expression' and text(node.child_by_field_name('argument')) == parameter。
# 遍历边界：node ← walk(function.child_by_field_name('body'))。
# 返回路径：False。
# 返回路径：True。
# 直接调用：walk。
# 直接调用：function.child_by_field_name。
# 直接调用：text。
# 直接调用：re.search。
# 直接调用：re.escape。
# 直接调用：node.child_by_field_name。
# 直接调用：source.startswith。
# [/maint-note:47e6c6df123f]
def _unchecked_dereferences(function, parameter):
    """True if the parameter is dereferenced and never compared against NULL first."""
    for node in walk(function.child_by_field_name("body")):
        source = text(node)
        if node.type in {"if_statement", "conditional_expression", "binary_expression"} and re.search(
                rf"\b{re.escape(parameter)}\b\s*(==|!=)|(==|!=)\s*\b{re.escape(parameter)}\b|^\s*!\s*{re.escape(parameter)}\b|\(\s*!?\s*{re.escape(parameter)}\s*\)", source):
            return False
        if node.type == "pointer_expression" and text(node.child_by_field_name("argument")) == parameter and source.startswith("*"):
            return True
        if node.type == "field_expression" and text(node.child_by_field_name("operator")) == "->" and text(node.child_by_field_name("argument")) == parameter:
            return True
        if node.type == "subscript_expression" and text(node.child_by_field_name("argument")) == parameter:
            return True
    return False


# [maint-note:aa1e6dce78a3]
# 维护导航：_null_arguments；下列为本函数源码中的接口与路径索引。
# 输入 function：类型由调用方与函数体约束。
# 输入 functions：类型由调用方与函数体约束。
# 分支条件：node.type != 'call_expression'。
# 分支条件：callee is None or callee is function。
# 分支条件：text(argument).replace(' ', '') in {t.replace(' ', '') for t in _NULL_TOKENS} and _unchecked_dereferences(callee, parameter)。
# 遍历边界：node ← walk(function)。
# 遍历边界：(index, (parameter, argument)) ← enumerate(zip(parameters, arguments))。
# 返回路径：results。
# 直接调用：walk。
# 直接调用：call_parts。
# 直接调用：functions.get。
# 直接调用：declared_name。
# 直接调用：callee.child_by_field_name。
# 直接调用：enumerate。
# 直接调用：zip。
# 直接调用：text(argument).replace。
# 直接调用：text。
# 直接调用：t.replace。
# 直接调用：_unchecked_dereferences。
# 直接调用：results.append。
# 集合派生：p 来自 walk(callee.child_by_field_name('declarator'))；过滤 p.type == 'parameter_declaration'。
# 集合派生：t 来自 _NULL_TOKENS。
# [/maint-note:aa1e6dce78a3]
def _null_arguments(function, functions):
    """A literal NULL passed to a same-file function that dereferences that parameter unchecked."""
    results = []
    for node in walk(function):
        if node.type != "call_expression":
            continue
        name, arguments = call_parts(node)
        callee = functions.get(name)
        if callee is None or callee is function:
            continue
        parameters = [declared_name(p) for p in walk(callee.child_by_field_name("declarator")) if p.type == "parameter_declaration"]
        for index, (parameter, argument) in enumerate(zip(parameters, arguments)):
            if text(argument).replace(" ", "") in {t.replace(" ", "") for t in _NULL_TOKENS} and _unchecked_dereferences(callee, parameter):
                results.append((node, [f"argument {index + 1} of {name}() is NULL and {name}() dereferences '{parameter}' without a NULL check"]))
    return results


# [maint-note:263464032a86]
# 维护导航：_null_init_dereferences；下列为本函数源码中的接口与路径索引。
# 输入 function：类型由调用方与函数体约束。
# 分支条件：body is None。
# 分支条件：node.type == 'identifier' and text(node) in nulls。
# 分支条件：node.type == 'field_expression'。
# 分支条件：base is not None and base.type == 'identifier' and (text(base) + '.' in nulls)。
# 分支条件：statement.type == 'declaration'。
# 分支条件：declarator.type != 'init_declarator'。
# 分支条件：target.type == 'pointer_declarator' and is_null(value)。
# 分支条件：value is not None and value.type == 'initializer_list' and value.named_children and all((is_null(item) for item in value.named_children))。
# 分支条件：statement.type == 'expression_statement' and statement.named_children and (statement.named_children[0].type == 'assignment_expression')。
# 分支条件：key。
# 分支条件：is_null(assignment.child_by_field_name('right'))。
# 分支条件：statement.type == 'if_statement'。
# 分支条件：returns。
# 分支条件：any((tracked(n) == key for n in walk(condition)))。
# 分支条件：node.type == 'pointer_expression' and text(node).startswith('*')。
# 遍历边界：statement ← body.named_children。
# 遍历边界：declarator ← statement.children_by_field_name('declarator')。
# 遍历边界：key ← list(nulls)。
# 遍历边界：node ← walk(statement)。
# 返回路径：[]。
# 返回路径：node is not None and text(node).replace(' ', '') in null_tokens。
# 返回路径：text(node)。
# 返回路径：text(base) + '.'。
# 返回路径：None。
# 返回路径：results。
# 写入目标：nulls[name]。
# 写入目标：nulls[name + '.']。
# 写入目标：nulls[key]。
# 直接调用：function.child_by_field_name。
# 直接调用：t.replace。
# 直接调用：text(node).replace。
# 直接调用：text。
# 直接调用：node.child_by_field_name。
# 直接调用：statement.children_by_field_name。
# 直接调用：declarator.child_by_field_name。
# 直接调用：declared_name。
# 直接调用：is_null。
# 直接调用：all。
# 直接调用：tracked。
# 直接调用：assignment.child_by_field_name。
# 直接调用：nulls.pop。
# 直接调用：statement.child_by_field_name。
# 直接调用：any。
# 直接调用：walk。
# 直接调用：list。
# 直接调用：text(node).startswith。
# 直接调用：results.append。
# 直接调用：key.rstrip。
# 集合派生：t 来自 _NULL_TOKENS。
# 集合派生：item 来自 value.named_children。
# 集合派生：n 来自 walk(statement.child_by_field_name('consequence'))。
# 集合派生：n 来自 walk(condition)。
# [/maint-note:263464032a86]
def _null_init_dereferences(function):
    """Pointer (or struct whose members are) initialised to NULL, later dereferenced.

    Path-insensitive on purpose: assignments inside if-statements are ignored, and only
    `if (<pointer mentioned>) ... return` counts as a check, so it reports some safe code.
    Runs only when Contract-3 switch F is off during Contract-3 evaluation (security_exam).
    """
    body = function.child_by_field_name("body")
    if body is None:
        return []
    nulls, results = {}, []
    null_tokens = {t.replace(" ", "") for t in _NULL_TOKENS}

    # [maint-note:a15509519875]
    # 维护导航：is_null；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 返回路径：node is not None and text(node).replace(' ', '') in null_tokens。
    # 直接调用：text(node).replace。
    # 直接调用：text。
    # [/maint-note:a15509519875]
    def is_null(node):
        return node is not None and text(node).replace(" ", "") in null_tokens

    # [maint-note:b62791f74a4c]
    # 维护导航：tracked；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 分支条件：node.type == 'identifier' and text(node) in nulls。
    # 分支条件：node.type == 'field_expression'。
    # 分支条件：base is not None and base.type == 'identifier' and (text(base) + '.' in nulls)。
    # 返回路径：text(node)。
    # 返回路径：text(base) + '.'。
    # 返回路径：None。
    # 直接调用：text。
    # 直接调用：node.child_by_field_name。
    # [/maint-note:b62791f74a4c]
    def tracked(node):
        if node.type == "identifier" and text(node) in nulls:
            return text(node)
        if node.type == "field_expression":
            base = node.child_by_field_name("argument")
            if base is not None and base.type == "identifier" and text(base) + "." in nulls:
                return text(base) + "."
        return None

    for statement in body.named_children:
        if statement.type == "declaration":
            for declarator in statement.children_by_field_name("declarator"):
                if declarator.type != "init_declarator":
                    continue
                target, value = declarator.child_by_field_name("declarator"), declarator.child_by_field_name("value")
                name = declared_name(target)
                if target.type == "pointer_declarator" and is_null(value):
                    nulls[name] = statement
                elif value is not None and value.type == "initializer_list" and \
                        value.named_children and all(is_null(item) for item in value.named_children):
                    nulls[name + "."] = statement
            continue
        if statement.type == "expression_statement" and statement.named_children and \
                statement.named_children[0].type == "assignment_expression":
            assignment = statement.named_children[0]
            key = tracked(assignment.child_by_field_name("left"))
            if key:
                if is_null(assignment.child_by_field_name("right")):
                    nulls[key] = statement
                else:
                    nulls.pop(key, None)
            continue
        if statement.type == "if_statement":
            condition = statement.child_by_field_name("condition")
            returns = any(n.type == "return_statement" for n in walk(statement.child_by_field_name("consequence")))
            if returns:
                for key in list(nulls):
                    if any(tracked(n) == key for n in walk(condition)):
                        nulls.pop(key)
            continue  # assignments inside the branch are ignored (path-insensitive)
        for node in walk(statement):
            if node.type == "pointer_expression" and text(node).startswith("*"):
                key = tracked(node.child_by_field_name("argument"))
                if key:
                    results.append((node, [f"'{key.rstrip('.')}' is set to NULL at line "
                                           f"{nulls[key].start_point.row + 1} and dereferenced here"]))
    return results


def _guarded_array_candidates(function):
    """Teaching H: a path-insensitive bounds check ignores early returns.

    Only local literal sizes/indices are modeled. No case IDs, labels or target
    proportions enter the rule. Students must prove guard reachability before
    reporting; the reference does not enable this deliberately flawed checker.
    """
    sizes, values = {}, {}
    nodes = list(walk(function))
    for node in nodes:
        if node.type == "array_declarator":
            size = node.child_by_field_name("size")
            if size is not None:
                sizes[declared_name(node)] = _int_value(size)
        elif node.type == "init_declarator":
            value = node.child_by_field_name("value")
            if value is not None:
                values[declared_name(node)] = _int_value(value)
    guards = [n for n in nodes if n.type == "if_statement"
              and any(c.type == "return_statement" for c in walk(n.child_by_field_name("consequence")))]
    for node in nodes:
        if node.type != "subscript_expression":
            continue
        array = text(node.child_by_field_name("argument"))
        index = text(node.child_by_field_name("index"))
        size, value = sizes.get(array), values.get(index)
        if size is None or value is None or value < size:
            continue
        if any(g.end_byte < node.start_byte and index in identifiers(g.child_by_field_name("condition"))
               for g in guards):
            yield node, [f"Teaching gap H: ignored index guard; {array}[{value}] with capacity {size}"]


# [maint-note:d4ffddf6784a]
# 分析边界：只解析传入的源码文本，不编译、链接或执行目标程序。
# 输出携带规则 ID、文件、行号与证据，供分类器和审阅者使用。
# 容量规则、NULL 实参规则与教学启发式由独立开关控制。
# F 关闭会启用故意忽略条件赋值的教学规则，可能制造空指针误报。
# 普通扫描保留完整规则；教学范围内允许明确的规则缺口。
# 维护导航：scan_c_source；下列为本函数源码中的接口与路径索引。
# 输入 source：str。
# 输入 file：str。
# 输入 context_lines：int。
# 输入 macros：类型由调用方与函数体约束；缺省表达式 None。
# 分支条件：node.start_point.row not in reported。
# 分支条件：switch_off('F')。
# 遍历边界：function ← functions.values()。
# 遍历边界：(node, evidence) ← _command_flows(function)。
# 遍历边界：(node, evidence) ← _races(root, functions)。
# 遍历边界：(node, evidence) ← [] if switch_off('D') else _string_writes(function, functions)。
# 遍历边界：(node, evidence) ← [] if switch_off('E') else _null_arguments(function, functions)。
# 遍历边界：(node, evidence) ← _null_init_dereferences(function)。
# 返回路径：(findings, root.has_error)。
# 直接调用：parse_source。
# 直接调用：Path(file).suffix.lower。
# 直接调用：Path。
# 直接调用：declared_name。
# 直接调用：n.child_by_field_name。
# 直接调用：walk。
# 直接调用：source.splitlines。
# 直接调用：findings.append。
# 直接调用：str(cwe).split。
# 直接调用：str。
# 直接调用：'\n'.join。
# 直接调用：range。
# 直接调用：max。
# 直接调用：min。
# 直接调用：len。
# 直接调用：functions.values。
# 直接调用：_command_flows。
# 直接调用：add。
# 直接调用：_races。
# 直接调用：set。
# 直接调用：switch_off。
# 直接调用：_string_writes。
# 直接调用：reported.add。
# 直接调用：_null_arguments。
# 直接调用：_null_init_dereferences。
# 集合派生：n 来自 walk(root)；过滤 n.type == 'function_definition' and (not n.has_error)。
# 表达式分流：switch_off('D')；真值与假值分支分别计算，不同时执行。
# 表达式分流：switch_off('E')；真值与假值分支分别计算，不同时执行。
# [/maint-note:d4ffddf6784a]
def scan_c_source(source: str, file: str, context_lines: int, macros=None):
    root = parse_source(source, Path(file).suffix.lower(), macros)
    functions = {declared_name(n.child_by_field_name("declarator")): n for n in walk(root)
                 if n.type == "function_definition" and not n.has_error}
    findings = []
    lines = source.splitlines()

    names = {119: ("Array index guard ignored by teaching rule", "guardedArrayAccess"), 78: ("Untrusted string reaches a shell", "shellStringFlow"), 362: ("Conflicting shared accesses in overlapping threads", "sharedGlobalRace"),
             120: ("String write exceeds the destination capacity", "capacityStringWrite"),
             476: ("NULL argument reaches an unchecked dereference", "nullArgumentDeref"),
             "476-init": ("NULL-initialised pointer is dereferenced", "nullInitDeref")}
    advice = {119: "Account for early-return guards before reporting an array access.", 78: "Use a fixed executable and an argument vector without a shell.",
              362: "Protect every conflicting access with the same mutex or an appropriate atomic operation.",
              120: "Bound the write by the remaining destination space, including the terminating NUL.",
              476: "Pass a valid object or check the parameter for NULL before dereferencing it.",
              "476-init": "Assign a valid object on every path, or check the pointer for NULL before dereferencing it."}

    # [maint-note:0e66311a9cd8]
    # 维护导航：add；下列为本函数源码中的接口与路径索引。
    # 输入 node：类型由调用方与函数体约束。
    # 输入 category：类型由调用方与函数体约束。
    # 输入 cwe：类型由调用方与函数体约束。
    # 输入 evidence：类型由调用方与函数体约束。
    # 直接调用：findings.append。
    # 直接调用：str(cwe).split。
    # 直接调用：str。
    # 直接调用：'\n'.join。
    # 直接调用：range。
    # 直接调用：max。
    # 直接调用：min。
    # 直接调用：len。
    # 集合派生：i 来自 range(max(0, line - 1 - context_lines), min(len(lines), line + context_lines))。
    # [/maint-note:0e66311a9cd8]
    def add(node, category, cwe, evidence):
        line = node.start_point.row + 1
        findings.append({
            "rule_id": f"cwe-{str(cwe).split('-')[0]}", "rule_name": names[cwe][0],
            "analyzer": "c-source", "analyzer_id": names[cwe][1],
            "category": category, "severity": "high", "confidence": "medium",
            "file": file, "line": line, "snippet": lines[line - 1][:300],
            "context": "\n".join(f"{i + 1}: {lines[i]}" for i in range(max(0, line - 1 - context_lines), min(len(lines), line + context_lines))),
            "evidence": evidence, "pattern": VERSION,
            "recommendation": advice[cwe],
            "references": [f"https://cwe.mitre.org/data/definitions/{str(cwe).split('-')[0]}.html"],
        })

    for function in functions.values():
        for node, evidence in _command_flows(function):
            add(node, "command_execution", 78, evidence)
    for node, evidence in _races(root, functions):
        add(node, "data_race", 362, evidence)
    reported = set()
    for function in functions.values():
        for node, evidence in ([] if switch_off("D") else _string_writes(function, functions)):  # switch D
            if node.start_point.row not in reported:
                reported.add(node.start_point.row)
                add(node, "string_overflow", 120, evidence)
    for function in functions.values():
        for node, evidence in ([] if switch_off("E") else _null_arguments(function, functions)):  # switch E
            add(node, "null_pointer", 476, evidence)
    if switch_off("F"):  # Contract-3 switch F (security_exam)
        for function in functions.values():
            for node, evidence in _null_init_dereferences(function):
                add(node, "null_pointer", "476-init", evidence)
    if switch_off("H"):
        for function in functions.values():
            for node, evidence in _guarded_array_candidates(function):
                add(node, "array_oob", 119, evidence)
    from code_agent.security_exam import contract3_active
    if contract3_active():
        from code_agent.security_engineering_rules import scan_engineering_source
        native, _ = scan_engineering_source(source, file, context_lines, macros, root=root)
        findings.extend(native)
    return findings, root.has_error
