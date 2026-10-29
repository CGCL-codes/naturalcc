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

VERSION = "c-source-v2"
LIMITATIONS = (
    "Syntax analysis, not a compiler or proof: local string flow and direct global "
    "accesses in overlapping pthread/std::thread lifetimes. No general alias, macro, "
    "cross-file, condition/path, custom sanitizer or custom synchronization analysis. "
    "Review evidence; use instrumented tests for runtime confirmation."
)


def parse_source(source: str, suffix: str):
    language = tree_sitter_cpp if suffix in {".cc", ".cpp", ".cxx", ".hpp"} else tree_sitter_c
    return Parser(Language(language.language())).parse(source.encode()).root_node


def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


def text(node):
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def declared_name(node):
    while node is not None and node.type not in {"identifier", "field_identifier"}:
        node = node.child_by_field_name("declarator")
    return text(node)


def call_parts(node):
    return text(node.child_by_field_name("function")).removeprefix("std::"), list(
        node.child_by_field_name("arguments").named_children)


def identifiers(node):
    return {text(n) for n in walk(node) if n.type == "identifier"} if node is not None else set()


def _command_flows(function):
    tainted = {}
    # String parameters are caller-controlled inputs, not proven external inputs.
    for node in walk(function.child_by_field_name("declarator")):
        if node.type == "parameter_declaration" and re.search(r"\b(?:char|wchar_t)\b", text(node)):
            name = declared_name(node.child_by_field_name("declarator"))
            if name and ("*" in text(node) or "[" in text(node)):
                tainted[name] = f"string parameter {name} at line {node.start_point.row + 1}"

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


def _race_accesses(function, globals_):
    """Direct accesses and lexical locksets; shadowed and atomic objects excluded."""
    local = {declared_name(n.child_by_field_name("declarator")) for n in walk(function)
             if n.type in {"parameter_declaration", "init_declarator"}}
    for n in walk(function):
        if n.type == "declaration":
            local.update(declared_name(c) for c in n.children_by_field_name("declarator"))
    shared = globals_ - local
    locks = set()

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


def _literal_length(node):
    if node is None or node.type != "string_literal":
        return None
    body = text(node)[1:-1]
    return len(re.sub(r"\\(x[0-9a-fA-F]+|[0-7]{1,3}|.)", "x", body))


def _string_writes(function, functions, facts=None, depth=0):
    """Track char-array capacities, aliases and known string lengths in one function
    (path-insensitive, last write wins) and flag string writes that cannot fit.
    Facts follow buffer arguments one call level into functions of the same file."""
    capacity, length, alias = dict(facts or {}).get("capacity", {}), {}, {}
    capacity, length, alias = dict(capacity), dict((facts or {}).get("length", {})), dict((facts or {}).get("alias", {}))

    def buffer(node):
        name = text(node).strip()
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def strlen_of(node):
        literal = _literal_length(node)
        return literal if literal is not None else length.get(buffer(node))

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
            if room is not None and written is not None and written > room:
                results.append((node, [f"{evidence} into '{destination}' of capacity {room}"]))
            if name in {"strcpy", "strncpy"} and len(arguments) >= 2 and strlen_of(arguments[1]) is not None:
                length[destination] = strlen_of(arguments[1])
    return results


_NULL_TOKENS = {"0", "NULL", "nullptr", "((void *)0)", "(void *)0"}


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


def scan_c_source(source: str, file: str, context_lines: int):
    root = parse_source(source, Path(file).suffix.lower())
    functions = {declared_name(n.child_by_field_name("declarator")): n for n in walk(root)
                 if n.type == "function_definition" and not n.has_error}
    findings = []
    lines = source.splitlines()

    names = {78: ("Untrusted string reaches a shell", "shellStringFlow"), 362: ("Conflicting shared accesses in overlapping threads", "sharedGlobalRace"),
             120: ("String write exceeds the destination capacity", "capacityStringWrite"),
             476: ("NULL argument reaches an unchecked dereference", "nullArgumentDeref")}
    advice = {78: "Use a fixed executable and an argument vector without a shell.",
              362: "Protect every conflicting access with the same mutex or an appropriate atomic operation.",
              120: "Bound the write by the remaining destination space, including the terminating NUL.",
              476: "Pass a valid object or check the parameter for NULL before dereferencing it."}

    def add(node, category, cwe, evidence):
        line = node.start_point.row + 1
        findings.append({
            "rule_id": f"cwe-{cwe}", "rule_name": names[cwe][0],
            "analyzer": "c-source", "analyzer_id": names[cwe][1],
            "category": category, "severity": "high", "confidence": "medium",
            "file": file, "line": line, "snippet": lines[line - 1][:300],
            "context": "\n".join(f"{i + 1}: {lines[i]}" for i in range(max(0, line - 1 - context_lines), min(len(lines), line + context_lines))),
            "evidence": evidence, "pattern": VERSION,
            "recommendation": advice[cwe],
            "references": [f"https://cwe.mitre.org/data/definitions/{cwe}.html"],
        })

    for function in functions.values():
        for node, evidence in _command_flows(function):
            add(node, "command_execution", 78, evidence)
    for node, evidence in _races(root, functions):
        add(node, "data_race", 362, evidence)
    reported = set()
    for function in functions.values():
        for node, evidence in _string_writes(function, functions):
            if node.start_point.row not in reported:
                reported.add(node.start_point.row)
                add(node, "string_overflow", 120, evidence)
    for function in functions.values():
        for node, evidence in _null_arguments(function, functions):
            add(node, "null_pointer", 476, evidence)
    return findings, root.has_error
