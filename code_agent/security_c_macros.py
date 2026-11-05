"""Offset-preserving neutralisation of C macros a preprocessor-less grammar cannot parse.

tree-sitter parses C without expanding macros, so kernel and embedded code that
leans on them turns into ERROR nodes:

* attribute macros before a definition: ``rt_weak int f(void)``;
* top-level registrations with free text: ``MSH_CMD_EXPORT(cmd, list the FILEs.)``;
* ``container_of`` style calls whose argument is a type:
  ``rt_list_entry(node, struct rt_object, list)``, ``va_arg(ap, void *)``;
* ``#if`` groups that split a statement or a brace pair across their arms:
  ``#ifdef X if (a) { #else if (b) { #endif`` or an arm that is ``else if ...``.

In RT-Thread's dfs_v2, finsh and kernel sources 26 of 34 files were affected,
which hid functions from the source rules and made the contract statistics
refuse to report. ``neutralize`` rewrites only those constructs, and only by
replacing characters with spaces (newlines kept), so every byte offset, line and
column of the result is the same place in the original source.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# Decorations that expand to nothing a parser needs. The list is deliberately
# conservative: a name that doubles as an identifier or a goto label (Linux's
# __init/__exit are RT-Thread label names) must not be here. Projects add their
# own through harvest_project_macros().
OBJECT_ATTRIBUTES = frozenset({
    "rt_weak", "rt_inline", "rt_used", "rt_noreturn", "rt_always_inline", "RT_WEAK", "RT_USED",
    "__weak", "__WEAK", "__packed", "__PACKED", "__USED", "__STATIC_INLINE", "__STATIC_FORCEINLINE",
    "__INLINE", "__NO_RETURN", "__root",
    "PRIVILEGED_FUNCTION", "PRIVILEGED_DATA", "portDONT_DISCARD", "FREERTOS_SYSTEM_CALL",
    "portFORCE_INLINE",
    "ALWAYS_INLINE", "FUNC_NORETURN", "__syscall", "__unused", "__maybe_unused", "__used",
    "__deprecated", "__noinit", "__always_inline", "__ramfunc",
})
FUNCTION_ATTRIBUTES = frozenset({
    "rt_section", "rt_align", "RT_SECTION", "ALIGN", "__ALIGNED", "__aligned", "__section",
    "__in_section", "__declspec", "SECTION",
})
# Constructs the grammar models with a type argument; their arguments stay as they are.
TYPE_TAKING = frozenset({
    "sizeof", "offsetof", "__builtin_offsetof", "_Alignof", "alignof", "__alignof__",
    "typeof", "__typeof__", "__typeof", "_Generic", "defined", "__builtin_types_compatible_p",
})

_ATTRIBUTE_BODY = re.compile(
    r"""^(?:
        __attribute__\s*\(\(.*\)\)
      | (?:static\s+)?(?:__)?inline(?:__)?(?:\s+__attribute__\s*\(\(.*\)\))?
      | __weak | __root | __packed | @.* | (?:_Pragma|PRAGMA)\s*\(.*\) | __declspec\s*\(.*\)
      | (?:volatile|const)(?:\s+(?:volatile|const))?
    )$""", re.X)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\([^)]*\))?[ \t]*(.*?)[ \t]*$", re.M)
_TYPE_ARGUMENT = re.compile(
    r"^\s*(?:(?:const|volatile)\s+)*(?:(?:struct|union|enum)\s+[A-Za-z_]\w*"
    r"|(?:unsigned|signed|char|short|int|long|float|double|void|_Bool)"
    r"(?:\s+(?:unsigned|signed|char|short|int|long|double))*)(?:\s*\*)*\s*$")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_ITERATOR = re.compile(r"(?i)^\w*for_?each\w*$")
_REGISTRATION = re.compile(r"^[A-Z][A-Z0-9_]*$")
_BARE_MARKER = re.compile(r"^[A-Z][A-Z0-9_]*_(?:END|BEGIN|START)$")
_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*", re.M)


# --- vocabulary ---------------------------------------------------------------

def harvest_attribute_macros(headers: Iterable[str]) -> tuple[set[str], set[str]]:
    """Project macros whose expansion is only an attribute, ``inline`` or a qualifier.

    RTOS trees define their decorations per compiler (``rt_weak`` is
    ``__attribute__((weak))`` for GCC, ``__weak`` for IAR); reading the project's
    own headers keeps the vocabulary in step with the code being analysed.
    Empty function-like macros (logging, hooks) are left alone: their uses are
    ordinary calls the grammar already parses.
    """
    definitions: dict[str, tuple[bool, str]] = {}
    for text in headers:
        for name, parameters, body in _DEFINE.findall(text):
            body = body.split("/*")[0].split("//")[0].strip()
            definitions.setdefault(name, (bool(parameters), body))
    object_like, function_like = set(), set()
    known = set(OBJECT_ATTRIBUTES) | set(FUNCTION_ATTRIBUTES)
    changed = True
    while changed:  # a macro defined as another decoration is one too
        changed = False
        for name, (has_parameters, body) in definitions.items():
            if name in object_like or name in function_like:
                continue
            decoration = (body != "" and bool(_ATTRIBUTE_BODY.match(body))) or body in known
            # An empty object-like body is a decoration only for names that look like one.
            if not body and not has_parameters and re.fullmatch(r"(?:rt|RT)_[a-z_]+|__[a-z_]+", name):
                decoration = True
            if decoration:
                (function_like if has_parameters else object_like).add(name)
                known.add(name)
                changed = True
    return object_like, function_like


def harvest_project_macros(project_dir: str | Path, limit: int = 4000) -> tuple[set[str], set[str]]:
    """Harvest from the project's headers, bounded so a huge tree stays cheap."""
    root = Path(project_dir)
    headers = []
    for path in sorted(root.rglob("*.h"))[:limit]:
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            headers.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return harvest_attribute_macros(headers)


# --- lexical helpers ------------------------------------------------------------

def _code_mask(source: str) -> list[bool]:
    """True for characters that are C code: not comments, literals or directives."""
    mask = [True] * len(source)
    index, length, line_start = 0, len(source), True
    while index < length:
        char = source[index]
        if line_start and char in " \t":
            index += 1
            continue
        if line_start and char == "#":
            end = index  # a directive runs to the end of the line, with continuations
            while end < length and (source[end] != "\n" or source[end - 1] == "\\"):
                end += 1
            for position in range(index, end):
                mask[position] = False
            index, line_start = end, False
            continue
        line_start = char == "\n"
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = length if end < 0 else end + 2
        elif source.startswith("//", index):
            end = source.find("\n", index)
            end = length if end < 0 else end
        elif char in "\"'":
            end = index + 1
            while end < length and source[end] != char and source[end] != "\n":
                end += 2 if source[end] == "\\" else 1
            end = min(length, end + 1)
        else:
            index += 1
            continue
        for position in range(index, end):
            if source[position] != "\n":
                mask[position] = False
        index = end
    return mask


def _blank(chars: list[str], start: int, end: int) -> None:
    for position in range(max(0, start), min(end, len(chars))):
        if chars[position] != "\n":
            chars[position] = " "


def _next_code(text: str, mask: list[bool], index: int) -> int:
    while index < len(text) and (not mask[index] or text[index].isspace()):
        index += 1
    return index


def _previous_code(text, mask: list[bool], index: int) -> int:
    while index >= 0 and (not mask[index] or text[index].isspace()):
        index -= 1
    return index


def _matching_paren(text: str, mask: list[bool], open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        if mask[index]:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    return index
    return -1


def _split_arguments(text: str, mask: list[bool], open_index: int, close_index: int) -> list[tuple[int, int]]:
    spans, depth, start = [], 0, open_index + 1
    for index in range(open_index + 1, close_index):
        if not mask[index]:
            continue
        if text[index] in "([{":
            depth += 1
        elif text[index] in ")]}":
            depth -= 1
        elif text[index] == "," and depth == 0:
            spans.append((start, index))
            start = index + 1
    spans.append((start, close_index))
    return spans


def _conditional_groups(text: str):
    """Every #if group as ([(directive_start, arm_start, arm_end), ...], endif_start, endif_end)."""
    groups, stack = [], []
    for found in _DIRECTIVE.finditer(text):
        kind, line_end = found.group(1), found.end()
        if kind in {"if", "ifdef", "ifndef"}:
            stack.append([(found.start(), line_end + 1, None)])
        elif kind in {"elif", "else"} and stack:
            arms = stack[-1]
            arms[-1] = (arms[-1][0], arms[-1][1], found.start())
            arms.append((found.start(), line_end + 1, None))
        elif kind == "endif" and stack:
            arms = stack.pop()
            arms[-1] = (arms[-1][0], arms[-1][1], found.start())
            groups.append((arms, found.start(), line_end))
    return groups


def _first_arm_depths(text: str, mask: list[bool]) -> list[int]:
    """Brace depth at every offset, counting only the first arm of each #if group.

    Counting every arm would add both ``{`` of ``#ifdef X if (a) { #else if (b) {``
    and leave the rest of the file one level too deep.
    """
    skipped = [False] * (len(text) + 1)
    for arms, _, _ in _conditional_groups(text):
        for _, start, end in arms[1:]:
            for position in range(start, min(end, len(text))):
                skipped[position] = True
    depths, depth = [0] * (len(text) + 1), 0
    for index, char in enumerate(text):
        depths[index] = depth
        if mask[index] and not skipped[index]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)
    depths[len(text)] = depth
    return depths


# --- the passes -------------------------------------------------------------------

def _blank_attributes(text: str, mask: list[bool], chars: list[str], objects, functions) -> None:
    for found in _IDENTIFIER.finditer(text):
        start, end, name = found.start(), found.end(), found.group(0)
        if name not in objects and name not in functions:
            continue
        if not mask[start] or (start and (text[start - 1].isalnum() or text[start - 1] == "_")):
            continue
        following, preceding = _next_code(text, mask, end), _previous_code(text, mask, start - 1)
        # Never touch a label or a goto target that happens to share the name.
        if (following < len(text) and text[following] == ":" and text[following:following + 2] != "::") \
                or text[max(0, preceding - 3):preceding + 1] == "goto":
            continue
        if name in objects:
            _blank(chars, start, end)
            continue
        if following >= len(text) or text[following] != "(":
            continue
        close = _matching_paren(text, mask, following)
        after = _next_code(text, mask, close + 1) if close > 0 else len(text)
        # A same-named expression macro (Linux's ALIGN(x, a)) stays: blank only in
        # declaration position -- at the start of a declaration, or directly before a
        # declarator or a function body.
        if close > 0 and (preceding < 0 or text[preceding] in ";{}" or
                          (after < len(text) and (text[after].isalpha() or text[after] in "_{"))):
            _blank(chars, start, close + 1)


def _rewrite_iterators(text: str, mask: list[bool], chars: list[str]) -> None:
    """``rt_list_for_each(node, head) { ... }`` expands to a ``for`` header. Rewrite
    the name as ``while`` padded with spaces: ``while (node, head)`` is a valid loop
    over a comma expression, so the body parses and every argument is kept."""
    for found in _IDENTIFIER.finditer(text):
        start, end, name = found.start(), found.end(), found.group(0)
        if not _ITERATOR.match(name) or len(name) < 5 or not mask[start]:
            continue
        if start and (text[start - 1].isalnum() or text[start - 1] == "_"):
            continue
        after = _next_code(text, mask, end)
        if after >= len(text) or text[after] != "(":
            continue
        close = _matching_paren(text, mask, after)
        following = _next_code(text, mask, close + 1) if close > 0 else len(text)
        # Followed by a body, not by ';' (a plain call that happens to share the pattern).
        if close > 0 and following < len(text) and (text[following] == "{" or text[following].isalpha()
                                                    or text[following] == "_"):
            chars[start:end] = list("while" + " " * (len(name) - 5))


def _blank_type_arguments(text: str, mask: list[bool], chars: list[str]) -> None:
    for found in _IDENTIFIER.finditer(text):
        if found.group(0) in TYPE_TAKING or not mask[found.start()]:
            continue
        after = found.end()
        while after < len(text) and text[after] in " \t":
            after += 1
        if after >= len(text) or text[after] != "(":
            continue
        close = _matching_paren(text, mask, after)
        if close < 0:
            continue
        spans = _split_arguments(text, mask, after, close)
        if len(spans) < 2:
            continue
        for number, (start, stop) in enumerate(spans):
            argument = text[start:stop]
            if _TYPE_ARGUMENT.match(argument) or argument.strip() == "()":
                # Take a comma with it so the call keeps a valid argument list.
                _blank(chars, start - 1 if number else start, stop if number else spans[1][0])


def _arm_is_open(text: str, mask: list[bool], depths: list[int], start: int, end: int) -> bool:
    """An arm that splits a statement: it starts by continuing the code before the
    ``#if`` (``else if``, ``| FLAG``) or leaves something for after the ``#endif``
    (a dangling ``else``, an ``if (...)``, an operator, a declarator without body)."""
    end = min(end, len(text))
    first, last = _next_code(text, mask, start), _previous_code(text, mask, end - 1)
    if first >= end or last < start:
        return False
    if re.match(r"else\b", text[first:first + 5]) or text[first] in "|&^,?=<>%":
        return True
    if text[max(start, last - 3):last + 1] == "else":
        return True
    if text[last] in ";{}:":
        return False
    return depths[start] > 0 or text[last] == ")"


def _brace_profile(text: str, mask: list[bool], start: int, end: int) -> tuple[int, int]:
    depth = lowest = 0
    for index in range(start, min(end, len(text))):
        if mask[index]:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                lowest = min(lowest, depth)
    return depth, lowest


def _structural(text: str, mask: list[bool]):
    depths = _first_arm_depths(text, mask)
    found = []
    for arms, endif_start, endif_end in _conditional_groups(text):
        unbalanced = any(_brace_profile(text, mask, start, stop) != (0, 0) for _, start, stop in arms)
        open_arm = any(_arm_is_open(text, mask, depths, start, stop) for _, start, stop in arms)
        after = _next_code(text, mask, endif_end)
        continued = re.match(r"else\b", text[after:after + 5]) is not None
        if unbalanced or open_arm or continued:
            found.append((arms, endif_start, endif_end))
    # Nested groups are kept: one inside a kept arm still needs resolving, and one
    # inside a dropped arm is simply blanked twice, which is harmless.
    return sorted(found, key=lambda g: g[0][0][0])


def _blank_registrations(text: str, mask: list[bool], chars: list[str]) -> None:
    """Top-level ``NAME(free text)`` statements and ``NAME_END`` markers."""
    depths = _first_arm_depths(text, mask)
    index = 0
    while index < len(text):
        if not mask[index] or depths[index] or not (text[index].isalpha() or text[index] == "_") \
                or (index and (text[index - 1].isalnum() or text[index - 1] == "_")):
            index += 1
            continue
        found = _IDENTIFIER.match(text, index)
        name, end = found.group(0), found.end()
        # Judge the statement boundary on the rewritten text: the previous
        # registration has already been blanked, so a run of them is recognised.
        before = _previous_code(chars, mask, index - 1)
        at_statement_start = before < 0 or chars[before] in ";}"
        after = _next_code(text, mask, end)
        if _REGISTRATION.match(name) and at_statement_start and after < len(text) and text[after] == "(":
            close = _matching_paren(text, mask, after)
            if close > 0:
                following = _next_code(text, mask, close + 1)
                if following >= len(text) or text[following] in ";#}" or text[following].isalpha() \
                        or text[following] == "_":
                    stop = following + 1 if following < len(text) and text[following] == ";" else close + 1
                    _blank(chars, index, stop)
                    index = stop
                    continue
        elif _BARE_MARKER.match(name) and at_statement_start:
            line_end = text.find("\n", end)
            if not text[end:line_end if line_end >= 0 else len(text)].strip():
                _blank(chars, index, end)
        index = end


def _prepare(source: str, object_like: Iterable[str], function_like: Iterable[str]) -> list[str]:
    """Passes that do not depend on which #if arm is kept."""
    mask = _code_mask(source)
    chars = list(source)
    _blank_attributes(source, mask, chars, OBJECT_ATTRIBUTES | set(object_like),
                      FUNCTION_ATTRIBUTES | set(function_like))
    _rewrite_iterators("".join(chars), mask, chars)
    _blank_type_arguments("".join(chars), mask, chars)
    return chars


def structural_groups(source: str, object_like: Iterable[str] = (), function_like: Iterable[str] = ()):
    """The #if groups ``neutralize`` resolves, numbered as its ``choose`` expects."""
    text = "".join(_prepare(source, object_like, function_like))
    return _structural(text, _code_mask(text))


def neutralize(source: str, object_like: Iterable[str] = (), function_like: Iterable[str] = (),
               choose: dict[int, int] | None = None) -> str:
    """Return ``source`` with unparsable macro uses blanked; same length, same lines.

    ``choose`` maps a structural #if group (numbered as ``structural_groups``
    returns them) to the arm to keep; by default the first arm is kept.
    """
    chars = _prepare(source, object_like, function_like)
    text = "".join(chars)
    mask = _code_mask(text)
    for number, (arms, endif_start, endif_end) in enumerate(_structural(text, mask)):
        keep = min((choose or {}).get(number, 0), len(arms) - 1)
        for index, (directive_start, arm_start, arm_end) in enumerate(arms):
            _blank(chars, directive_start, arm_start)       # the #if / #elif / #else line
            if index != keep:
                _blank(chars, arm_start, arm_end)           # an arm not taken
        _blank(chars, endif_start, endif_end)               # the matching #endif
    text = "".join(chars)
    _blank_registrations(text, _code_mask(text), chars)
    result = "".join(chars)
    assert len(result) == len(source)
    return result
