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

VERSION = "c-source-v1"
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


def scan_c_source(source: str, file: str, context_lines: int):
    root = parse_source(source, Path(file).suffix.lower())
    functions = {declared_name(n.child_by_field_name("declarator")): n for n in walk(root)
                 if n.type == "function_definition" and not n.has_error}
    findings = []
    lines = source.splitlines()

    def add(node, category, cwe, evidence):
        line = node.start_point.row + 1
        findings.append({
            "rule_id": f"cwe-{cwe}", "rule_name": "Untrusted string reaches a shell" if cwe == 78 else "Conflicting shared accesses in overlapping threads",
            "analyzer": "c-source", "analyzer_id": "shellStringFlow" if cwe == 78 else "sharedGlobalRace",
            "category": category, "severity": "high", "confidence": "medium",
            "file": file, "line": line, "snippet": lines[line - 1][:300],
            "context": "\n".join(f"{i + 1}: {lines[i]}" for i in range(max(0, line - 1 - context_lines), min(len(lines), line + context_lines))),
            "evidence": evidence, "pattern": VERSION,
            "recommendation": "Use a fixed executable and an argument vector without a shell." if cwe == 78 else "Protect every conflicting access with the same mutex or an appropriate atomic operation.",
            "references": [f"https://cwe.mitre.org/data/definitions/{cwe}.html"],
        })

    for function in functions.values():
        for node, evidence in _command_flows(function):
            add(node, "command_execution", 78, evidence)
    for node, evidence in _races(root, functions):
        add(node, "data_race", 362, evidence)
    return findings, root.has_error
