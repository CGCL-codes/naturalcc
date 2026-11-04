"""RTOS-aware source checks for the four high-risk contract categories.

These complement ``security_c_rules`` rather than replace it.  The built-in C
rules only know the libc spelling of each API (``memcpy``, ``pthread_mutex_lock``)
and only track plain global scalars, so on an RTOS kernel -- where the same
operations appear as ``rt_memcpy``/``rt_spin_lock_irqsave`` and the shared state
lives in fields reached through an object pointer -- they see nothing.

Every rule here is a syntactic pattern with explicit evidence and no knowledge of
any particular file, function or test label.  They are heuristics, not proofs:
each finding names the construct it matched so a reviewer can confirm or reject it.

Emitted categories: buffer_overflow (CWE-787), data_race (CWE-362),
memory_leak (CWE-401), command_execution (CWE-78).
"""
from __future__ import annotations

import re

from code_agent.security_c_rules import call_parts, declared_name, parse_source, text, walk

VERSION = "rtos-source-v1"
LIMITATIONS = (
    "Syntactic RTOS-aware heuristics: shell command construction, lock-scope escapes, "
    "allocation ownership and destination-capacity mismatches. No alias, macro expansion, "
    "cross-translation-unit, path-sensitive or runtime analysis. Confirm each finding "
    "against the surrounding code before acting on it."
)

# --- RTOS API vocabulary -------------------------------------------------
# Kernels wrap libc; a rule that only knows the libc name is blind to the wrapper.
SHELL_SINKS = {"system", "popen", "_popen", "msh_exec", "finsh_exec", "exec_cmd"}
FORMATTERS = {"sprintf", "snprintf", "rt_sprintf", "rt_snprintf", "vsprintf", "vsnprintf",
              "rt_vsprintf", "rt_vsnprintf"}
UNBOUNDED_COPIES = {"strcpy", "rt_strcpy", "strcat", "rt_strcat", "sprintf", "rt_sprintf",
                    "vsprintf", "rt_vsprintf", "gets"}
BOUNDED_COPIES = {"memcpy", "rt_memcpy", "memmove", "rt_memmove", "strncpy", "rt_strncpy",
                  "strncat", "rt_strncat", "snprintf", "rt_snprintf", "vsnprintf", "rt_vsnprintf"}
ALLOCATORS = {"malloc", "calloc", "realloc", "strdup", "rt_malloc", "rt_calloc", "rt_realloc",
              "rt_strdup", "RT_KERNEL_MALLOC", "rt_malloc_align", "rt_page_alloc"}
DEALLOCATORS = {"free", "rt_free", "RT_KERNEL_FREE", "rt_free_align", "rt_page_free"}
LOCK_ACQUIRE = {"rt_spin_lock", "rt_spin_lock_irqsave", "rt_hw_interrupt_disable",
                "rt_mutex_take", "rt_mutex_take_interruptible", "rt_mutex_take_killable",
                "rt_enter_critical", "rt_sched_lock", "pthread_mutex_lock", "rt_sem_take"}
LOCK_RELEASE = {"rt_spin_unlock", "rt_spin_unlock_irqrestore", "rt_hw_interrupt_enable",
                "rt_mutex_release", "rt_exit_critical", "rt_sched_unlock",
                "rt_sched_unlock_n_resched", "pthread_mutex_unlock", "rt_sem_release"}
# Initialising a lock is not holding one.
LOCK_NOISE = {"rt_spin_lock_init", "rt_mutex_init", "pthread_mutex_init", "rt_sem_init"}

CWE_META = {
    787: ("buffer_overflow", "Write may exceed the destination capacity", "capacityWriteOverflow",
          "Bound every write by the real capacity of the destination, including the terminating NUL."),
    362: ("data_race", "Shared field accessed outside the lock that protects it", "lockScopeEscape",
          "Keep the shared access inside the critical section, or publish it with an atomic operation."),
    401: ("memory_leak", "Allocation is lost before it is released", "ownershipLeak",
          "Release the allocation on every path, or transfer its ownership explicitly."),
    78: ("command_execution", "Externally derived string reaches a shell dispatcher", "shellCommandBuild",
         "Run a fixed command with an argument vector, or validate the value against a strict allowlist."),
}


def _functions(root):
    """Yield every function definition.

    RTOS sources are macro-heavy (``rt_weak``, ``rt_inline``, ``RTM_EXPORT``), so
    tree-sitter marks many otherwise valid definitions as containing an error --
    in RT-Thread's kservice.c that is 26 of 46 functions. Skipping them hides most
    of the kernel, so errors are tolerated here and each matched node is checked
    individually instead (see ``add``).
    """
    for node in walk(root):
        if node.type == "function_definition":
            name = declared_name(node.child_by_field_name("declarator"))
            body = node.child_by_field_name("body")
            if name and body is not None:
                yield name, node, body


def _calls(node):
    for child in walk(node):
        if child.type == "call_expression":
            name, args = call_parts(child)
            yield child, name, args


def _target_name(node, lhs) -> str:
    """The name an assignment writes to.

    A declaration's declarator carries the pointer syntax (``char *buf``), so the
    raw text would be ``*buf`` and would never match the ``buf`` used everywhere
    else; an assignment's left side is already the expression (``mnt->fullpath``).
    """
    if node.type == "init_declarator":
        return declared_name(lhs)
    return text(lhs)


def _unwrap(node):
    """Strip casts and parentheses: ``(struct foo *)rt_malloc(n)`` is still a call.

    C code almost always casts an allocator's result, so matching the bare
    call_expression misses most real allocations.
    """
    while node is not None and node.type in {"cast_expression", "parenthesized_expression"}:
        inner = node.child_by_field_name("value") or (
            node.named_children[-1] if node.named_children else None)
        if inner is None or inner is node:
            break
        node = inner
    return node


def _is_literal(node) -> bool:
    return node is not None and node.type in {"string_literal", "char_literal", "number_literal",
                                              "concatenated_string"}


def _lock_regions(body) -> list[tuple[int, int]]:
    """Byte ranges covered by a held lock, from each acquire to its matching release."""
    events = []
    for node, name, _ in _calls(body):
        if name in LOCK_NOISE:
            continue
        if name in LOCK_ACQUIRE:
            events.append((node.start_byte, "acquire"))
        elif name in LOCK_RELEASE:
            events.append((node.end_byte, "release"))
    events.sort()
    regions, depth, start = [], 0, None
    for position, kind in events:
        if kind == "acquire":
            if depth == 0:
                start = position
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                regions.append((start, position))
                start = None
    if depth > 0 and start is not None:
        regions.append((start, body.end_byte))
    return regions


def _field_target(node):
    """Return ('obj->field' text, field name) when node is a field access expression."""
    if node is None or node.type != "field_expression":
        return None
    field = text(node.child_by_field_name("field"))
    return (text(node), field) if field else None


def _written_fields(body):
    """Field accesses that are written (assigned or ++/--), with their positions."""
    for node in walk(body):
        target = None
        if node.type == "assignment_expression":
            target = _field_target(node.child_by_field_name("left"))
        elif node.type == "update_expression":
            target = _field_target(node.child_by_field_name("argument"))
        if target:
            yield node, target[0], target[1]


# --- CWE-78: a command string built from a non-literal value ---------------

def _command_builds(body):
    """A buffer formatted with a non-literal value and then handed to a shell dispatcher."""
    built: dict[str, tuple[object, str]] = {}
    for node, name, args in _calls(body):
        if name in FORMATTERS and len(args) >= 2:
            fmt_index = 1 if name in {"sprintf", "rt_sprintf", "vsprintf", "rt_vsprintf"} else 2
            if len(args) <= fmt_index:
                continue
            fmt = text(args[fmt_index])
            if not re.search(r"%[-+ #0-9.*]*s", fmt):
                continue
            values = [a for a in args[fmt_index + 1:] if not _is_literal(a)]
            if values:
                built[text(args[0])] = (node, ", ".join(text(v) for v in values))
        elif name in SHELL_SINKS and args:
            target = text(args[0])
            if target in built:
                origin, values = built[target]
                yield node, (f"{values} formatted into {target} at line {origin.start_point.row + 1} "
                             f"-> {name}() interprets the whole string as a command")


# --- CWE-401: an allocation that is neither released nor handed off --------

NULL_TOKENS = {"RT_NULL", "NULL", "0", "nullptr", "((void *)0)", "(void *)0"}
# Kernels spell "give this object back" many ways; a release by any of these names
# ends this function's responsibility for the memory.
RELEASING = re.compile(r"free|destroy|delete|release|unref|deinit|detach|put\b", re.I)


def _inferred_allocators(root) -> set[str]:
    """Functions whose result this file later frees are allocators in this codebase.

    Kernels wrap their heap in named helpers (``dfs_normalize_path``,
    ``rt_ringbuffer_create``); treating a fixed libc list as the only source of
    owned memory misses most real allocations.
    """
    produced: dict[str, str] = {}
    for node in walk(root):
        if node.type in {"init_declarator", "assignment_expression"}:
            lhs = node.child_by_field_name("declarator" if node.type == "init_declarator" else "left")
            rhs = node.child_by_field_name("value" if node.type == "init_declarator" else "right")
            if rhs is not None and rhs.type == "call_expression":
                produced.setdefault(_target_name(node, lhs), call_parts(rhs)[0])
    freed = {text(a) for _, name, args in _calls(root) if name in DEALLOCATORS for a in args[:1]}
    return {producer for target, producer in produced.items()
            if target in freed and producer not in DEALLOCATORS}


def _owning_fields(root) -> set[str]:
    """Fields this file frees or fills from an allocation, so clearing one loses memory."""
    owned = set()
    for _, name, args in _calls(root):
        if name in DEALLOCATORS and args:
            target = _field_target(args[0])
            if target:
                owned.add(target[1])
    allocators = ALLOCATORS | _inferred_allocators(root)
    for node in walk(root):
        if node.type in {"init_declarator", "assignment_expression"}:
            lhs = node.child_by_field_name("declarator" if node.type == "init_declarator" else "left")
            rhs = _unwrap(node.child_by_field_name("value" if node.type == "init_declarator" else "right"))
            if rhs is not None and rhs.type == "call_expression" and call_parts(rhs)[0] in allocators:
                target = _field_target(lhs)
                if target:
                    owned.add(target[1])
    return owned


def _error_exit(statement) -> bool:
    """True for a failure return: a NULL pointer or a negative error code.

    ``return RT_EOK;`` / ``return obj;`` end the successful path, where the
    allocation has normally been handed to the caller.
    """
    value = statement.named_children[0] if statement.named_children else None
    if value is None:
        return False
    spelling = text(value).strip()
    if spelling in {"RT_NULL", "NULL", "nullptr", "((void *)0)", "(void *)0"}:
        return True
    return spelling.startswith("-") and not spelling.startswith("--")


def _mutually_exclusive(first, second) -> bool:
    """True when the two nodes sit in opposite branches of the same ``if``.

    Allocating the same pointer once in each arm of an if/else is not an
    overwrite; only one arm ever runs.
    """
    def branches(node):
        taken, current = {}, node
        while current is not None:
            parent = current.parent
            if parent is not None and parent.type == "if_statement":
                # Each child_by_field_name call returns a fresh wrapper, so compare ids.
                consequence = parent.child_by_field_name("consequence")
                alternative = parent.child_by_field_name("alternative")
                if consequence is not None and current.id == consequence.id:
                    taken[parent.id] = "then"
                elif alternative is not None and current.id == alternative.id:
                    taken[parent.id] = "else"
            current = parent
        return taken

    other = branches(second)
    return any(key in other and other[key] != arm for key, arm in branches(first).items())


def _null_guarded(node, target: str) -> bool:
    """True when node sits under an ``if`` that tested this exact pointer against NULL.

    The allocation failed, so there is nothing to release. The test must name the
    pointer itself: ``if (queue->work_thread == RT_NULL)`` says nothing about
    whether ``queue`` still has to be freed.
    """
    negated = re.compile(rf"!\s*{re.escape(target)}\b")
    compared = re.compile(rf"{re.escape(target)}\s*(?:==|!=)")
    current = node
    while current is not None:
        if current.type == "if_statement":
            condition = text(current.child_by_field_name("condition"))
            # `if (!seq)` is the same failure test as `if (seq == RT_NULL)`.
            if negated.search(condition):
                return True
            if compared.search(condition) and any(token in condition for token in NULL_TOKENS):
                return True
        current = current.parent
    return False


def _leaks(body, owning_fields, allocators):
    allocations, frees = [], []
    for node in walk(body):
        if node.type in {"init_declarator", "assignment_expression"}:
            lhs = node.child_by_field_name("declarator" if node.type == "init_declarator" else "left")
            rhs = _unwrap(node.child_by_field_name("value" if node.type == "init_declarator" else "right"))
            if rhs is not None and rhs.type == "call_expression" and call_parts(rhs)[0] in allocators:
                allocations.append((node, _target_name(node, lhs), call_parts(rhs)[0]))
    for node, name, args in _calls(body):
        if name in DEALLOCATORS and args:
            frees.append((node.start_byte, text(args[0])))

    def released(target: str, base: str, lower: int, upper: int) -> bool:
        return any(lower < position < upper and (freed == target or freed == base)
                   for position, freed in frees)

    def handed_over(target: str, base: str, lower: int, upper: int) -> bool:
        """Ownership left this function: stored elsewhere, or given to a releasing helper.

        Passing the pointer to a reader or writer (``dfs_file_read(f, buf, n)``) is
        ordinary use and must not be mistaken for a hand-over.
        """
        for node in walk(body):
            if not lower < node.start_byte < upper:
                continue
            if node.type == "assignment_expression" and text(node.child_by_field_name("right")) == target:
                return True
            if node.type == "call_expression":
                name, args = call_parts(node)
                if RELEASING.search(name) and any(text(a).strip("&()") in {target, base} for a in args):
                    return True
        return False

    # (a) a return reached while this function still owns the allocation
    for node, target, allocator in allocations:
        base = target.split("->")[0].split(".")[0]
        for statement in walk(body):
            if statement.type != "return_statement" or statement.start_byte <= node.start_byte:
                continue
            if released(target, base, node.start_byte, statement.start_byte):
                continue
            if handed_over(target, base, node.start_byte, statement.start_byte):
                break
            # Returning the object that holds the allocation is the hand-over itself.
            if base and base != target and re.search(rf"\b{re.escape(base)}\b", text(statement)):
                break
            if target == base and re.search(rf"\b{re.escape(base)}\b", text(statement)):
                break
            if _null_guarded(statement, target):
                continue
            # Only an error exit leaks. A success exit is normally the hand-over
            # itself: the object that now holds the allocation goes to the caller,
            # often through a helper this analysis cannot follow.
            if not _error_exit(statement):
                break
            yield statement, (f"{allocator}() filled {target} at line {node.start_point.row + 1}; "
                              f"this error return leaves the function without releasing it or "
                              f"passing ownership on")
            break

    # (b) an owning field cleared instead of released
    for node, target, field in _written_fields(body):
        if field not in owning_fields:
            continue
        value = node.child_by_field_name("right") if node.type == "assignment_expression" else None
        if value is None or text(value) not in NULL_TOKENS:
            continue
        if any(freed in {target, field} for _, freed in frees):
            continue
        yield node, (f"{target} is cleared without releasing it, while this file allocates or frees "
                     f"the same field elsewhere, so the memory it owned becomes unreachable")

    # (c) a pointer that still owns memory overwritten by another value
    for node, target, allocator in allocations:
        base = target.split("->")[0].split(".")[0]
        for other in walk(body):
            if other.type != "assignment_expression" or other.start_byte <= node.start_byte:
                continue
            if text(other.child_by_field_name("left")) != target:
                continue
            if released(target, base, node.start_byte, other.start_byte):
                break
            if _mutually_exclusive(node, other):
                continue
            value = other.child_by_field_name("right")
            if value is not None and text(value) in NULL_TOKENS:
                break
            yield other, (f"{target} still owns the {allocator}() result from line "
                          f"{node.start_point.row + 1} when it is overwritten here, so the "
                          f"previous allocation can no longer be freed")
            break


# --- CWE-362: a shared field touched outside the lock ---------------------

STATE_MUTATORS = {"rt_list_insert_after", "rt_list_insert_before", "rt_list_remove",
                  "rt_slist_append", "rt_slist_insert", "rt_slist_remove",
                  "rt_susp_list_dequeue", "rt_susp_list_enqueue"}


def _lock_escapes(body):
    """Shared state mutated outside the critical section of a lock-holding function.

    A function that takes a lock has declared which state needs protection; a write
    to an object's field that sits outside every lock region is either a statement
    that escaped the critical section or one that runs before it is entered. Both
    are the classic shape of a lost update.
    """
    regions = _lock_regions(body)
    if not regions:
        return
    inside = {field for node, _, field in _written_fields(body)
              if any(start <= node.start_byte < end for start, end in regions)}

    def outside(node) -> bool:
        return not any(start <= node.start_byte < end for start, end in regions)

    for node, expression, field in _written_fields(body):
        if not outside(node):
            continue
        where = ("the same field is also written under the lock" if field in inside
                 else "the statement sits outside every critical section in this function")
        yield node, (f"{expression} is modified without holding the lock this function otherwise "
                     f"takes ({where}), so a concurrent reader or writer can interleave here")
    # List surgery is a multi-step mutation of shared links, not a single store.
    for node, name, args in _calls(body):
        if name in STATE_MUTATORS and outside(node) and args:
            yield node, (f"{name}() rewires shared links outside the critical section this "
                         f"function otherwise holds, so the list can be observed half-updated")


# --- CWE-787: the write can exceed the destination --------------------------

def _local_values(body) -> dict[str, str]:
    """Single-assignment local sizes, so ``rt_malloc(path_len)`` can be compared with a write."""
    values: dict[str, str] = {}
    duplicated: set[str] = set()
    for node in walk(body):
        if node.type in {"init_declarator", "assignment_expression"}:
            lhs = node.child_by_field_name("declarator" if node.type == "init_declarator" else "left")
            rhs = node.child_by_field_name("value" if node.type == "init_declarator" else "right")
            if lhs is None or rhs is None:
                continue
            name = _target_name(node, lhs)
            if name in values:
                duplicated.add(name)
            values[name] = text(rhs)
    return {k: v for k, v in values.items() if k not in duplicated}


def _capacity_overflows(body):
    sizes = _local_values(body)

    def resolve(expression: str) -> str:
        return re.sub(r"\s", "", sizes.get(expression.strip(), expression))

    allocated: dict[str, tuple[object, str]] = {}
    for node in walk(body):
        if node.type in {"init_declarator", "assignment_expression"}:
            lhs = node.child_by_field_name("declarator" if node.type == "init_declarator" else "left")
            rhs = _unwrap(node.child_by_field_name("value" if node.type == "init_declarator" else "right"))
            if rhs is not None and rhs.type == "call_expression":
                allocator, args = call_parts(rhs)
                if allocator in ALLOCATORS and args:
                    allocated[_target_name(node, lhs)] = (node, text(args[0]))
    for node, name, args in _calls(body):
        if name in BOUNDED_COPIES and len(args) >= 3:
            destination, size = text(args[0]), text(args[-1] if name in FORMATTERS else args[2])
            size = text(args[1]) if name in FORMATTERS else size
            normalized = resolve(size)
            # A bound derived from the destination's own size but made larger is always wrong.
            match = re.fullmatch(rf"sizeof\(\*?{re.escape(destination)}\)\+(\d+)", normalized)
            if match:
                yield node, (f"the bound passed to {name}() is sizeof({destination}) + {match[1]}, "
                             f"which is larger than {destination} itself")
                continue
            # A length taken from the source says nothing about the destination.
            source = text(args[1]) if len(args) > 1 else ""
            if source and re.search(rf"\b(?:rt_)?strlen\(\s*{re.escape(source)}\s*\)", normalized):
                yield node, (f"{name}() copies strlen({source}) bytes into {destination}, "
                             f"a length measured on the source instead of the destination's capacity")
                continue
            if destination in allocated:
                origin, requested = allocated[destination]
                if _exceeds(normalized, resolve(requested)):
                    yield node, (f"{destination} was allocated with {requested} at line "
                                 f"{origin.start_point.row + 1}, but {name}() may write {size} bytes")
        if name in UNBOUNDED_COPIES and len(args) >= 2 and not _is_literal(args[1]):
            if name in {"sprintf", "rt_sprintf", "vsprintf", "rt_vsprintf"}:
                yield node, (f"{name}() writes a formatted string into {text(args[0])} with no "
                             f"length limit, so the result is bounded only by the input")
            else:
                yield node, (f"{name}() copies {text(args[1])} into {text(args[0])} with no length "
                             f"limit; the destination capacity is never checked")


def _wrap_off_by_one(body):
    """``if (i > CAP) i = 0;`` leaves i == CAP valid, so the next subscript runs one past the end.

    Only reported when the same expression is actually used as an array subscript in
    this function, which is what makes the off-by-one reachable.
    """
    subscripts = {text(n.child_by_field_name("index")) for n in walk(body)
                  if n.type == "subscript_expression"}
    if not subscripts:
        return
    for node in walk(body):
        if node.type != "if_statement":
            continue
        raw = text(node.child_by_field_name("condition")).strip("()")
        # Split on the comparison only: the '>' in '->' is part of a field access.
        parts = re.split(r"(?<![-=<>!])>(?!=)", raw)
        if len(parts) != 2:
            continue
        index, bound = parts[0].strip(), parts[1].strip()
        if index not in subscripts:
            continue
        body_text = text(node.child_by_field_name("consequence"))
        if not re.search(rf"{re.escape(index)}\s*=\s*0\s*;", body_text):
            continue
        yield node, (f"{index} is reset only when it is strictly greater than {bound}, so the "
                     f"value {bound} itself survives and the next subscript reads one past the end")


def _exceeds(write: str, allocation: str) -> bool:
    """True when two '<same base>+<k>' size expressions disagree in the unsafe direction."""
    write_match = re.fullmatch(r"(.+?)\+(\d+)", write)
    allocation_match = re.fullmatch(r"(.+?)\+(\d+)", allocation)
    if not write_match or not allocation_match or write_match[1] != allocation_match[1]:
        return False
    return int(write_match[2]) > int(allocation_match[2])


def scan_rtos_source(source: str, file: str, context_lines: int):
    """Return (findings, parse_failed) for the four high-risk contract categories."""
    suffix = "." + file.rsplit(".", 1)[-1] if "." in file else ".c"
    root = parse_source(source, suffix.lower())
    lines = source.splitlines()
    owning_fields = _owning_fields(root)
    allocators = ALLOCATORS | _inferred_allocators(root)
    findings: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add(node, cwe, evidence):
        line = node.start_point.row + 1
        # The enclosing function may contain a macro the grammar cannot model; the
        # construct this finding rests on must still parse cleanly.
        if (cwe, line) in seen or line > len(lines) or node.has_error:
            return
        seen.add((cwe, line))
        category, title, analyzer_id, advice = CWE_META[cwe]
        findings.append({
            "rule_id": f"cwe-{cwe}", "rule_name": title,
            "analyzer": "c-source", "analyzer_id": analyzer_id,
            "category": category, "severity": "high", "confidence": "medium",
            "file": file, "line": line, "snippet": lines[line - 1][:300],
            "context": "\n".join(f"{i + 1}: {lines[i]}" for i in range(
                max(0, line - 1 - context_lines), min(len(lines), line + context_lines))),
            "evidence": evidence, "pattern": VERSION, "recommendation": advice,
            "references": [f"https://cwe.mitre.org/data/definitions/{cwe}.html"],
        })

    for _, _, body in _functions(root):
        for node, evidence in _command_builds(body):
            add(node, 78, evidence)
        for node, evidence in _leaks(body, owning_fields, allocators):
            add(node, 401, evidence)
        for node, evidence in _lock_escapes(body):
            add(node, 362, evidence)
        for node, evidence in _capacity_overflows(body):
            add(node, 787, evidence)
        for node, evidence in _wrap_off_by_one(body):
            add(node, 787, evidence)
    return findings, root.has_error
