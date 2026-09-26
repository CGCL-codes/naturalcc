"""C audit contracts and literal-format checks, independent of evaluation datasets."""
from __future__ import annotations

import re

REVIEW_VERSION = "c-security-review-v1"
REVIEW_GUIDANCE = {
    "evidence": (
        "Scanner findings are candidates, not verdicts; zero findings is not a safety proof. "
        "Read the complete function and callers before classifying. For each candidate, verify "
        "reachability, exact API semantics, guards and the target CWE. Try to refute the candidate "
        "using bounds, validation, ownership or synchronization already present. Report a concrete "
        "path and trigger, not just a dangerous function name. Keep out-of-scope issues separate."
    ),
    "bounds": (
        "For a destination of capacity C, calculate the maximum written bytes including the final "
        "NUL. A successful fgets(dst,n,...) stores at most n-1 bytes plus NUL, not n bytes plus a "
        "newline plus NUL. strcspn on that valid string returns at most its length; replacing its "
        "terminating NUL is within bounds. Check failed reads separately, without inventing an "
        "overlong successful read. strcpy/strcat require strlen(prefix)+strlen(suffix)+1 <= C; "
        "use actual literal lengths, not the capacity of their declared arrays. Width W on scanf "
        "%s/%[ needs W+1 destination bytes; %c does not append NUL. Distinguish reads from writes."
    ),
    "commands": (
        "CWE-78 requires attacker-controlled shell syntax reaching an executable command. Trace "
        "source -> transformations -> validation -> shell sink. Fixed-format %d/%u output is an "
        "integer representation, not the original input string: empty input, failed numeric "
        "validation or integer overflow alone does not prove shell injection. Enumerate the "
        "actual allowed characters and follow rejection/return branches; if even the fixed command "
        "prefix is rejected, that sink may be unreachable. Do not call path traversal, argument "
        "injection or arbitrary hypothetical undefined behavior OS command injection without "
        "a demonstrated shell-syntax path. Conversely a blacklist must cover all shell syntax; "
        "checking one metacharacter or bounding the input length is not sufficient."
    ),
    "concurrency": (
        "CWE-362 requires conflicting accesses to shared state or a shared external resource "
        "without adequate synchronization. Identify the accesses and the lock covering each. "
        "Do not infer a race from pthread_create, sleep, volatile or a missing return check alone. "
        "Inspect mutex initialization and lock failure paths: shared accesses after an unsuccessful "
        "lock must not proceed unprotected; state the feasible failure condition explicitly. "
        "A success guard around shared accesses prevents that failure-path race. A signal handler "
        "calling a non-async-signal-safe mutex function can deadlock (CWE-479/833); that alone is "
        "not evidence of concurrent unprotected accesses. Invalid unlock and cancellation cleanup "
        "are separate issues unless an actual conflicting-access path is demonstrated."
    ),
    "ownership": (
        "For CWE-401 track each allocation identity through aliases, reassignments, swaps, helpers, "
        "return paths and loops. A free of a different allocation is not a release of the original. "
        "Show a successful allocation path on which ownership is lost or never released after its "
        "effective lifetime. Returning ownership to a caller or using stack storage is not by "
        "itself a leak. Do not conflate a null dereference, double free or use-after-free with a leak."
    ),
}

# Preserve literal contents and source positions; discard comments, not code lines.
_C_TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'
                      r"'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|[^\s]")
_CONVERSION = re.compile(
    r"%(?:\d+\$)?(?P<suppress>\*)?(?P<width>\d+)?(?:hh|ll|[hljztL])?"
    r"(?P<kind>\[(?:\^?\])?[^\]]*\]|[%scdiuoxXfFeEgGaApn])"
)


def unbounded_scanf_lines(source: str) -> set[int]:
    """Find unbounded string/scanset assignments in literal scanf-family formats.

    This is a lexical candidate check, not destination-size or reachability analysis.
    Macro/variable formats and escaped character codes require semantic review.
    """
    tokens = [(m.group(), m.start()) for m in _C_TOKEN.finditer(source)
              if not m.group().startswith(("//", "/*"))]
    result = set()
    for i, (name, offset) in enumerate(tokens[:-1]):
        if name not in {"scanf", "fscanf", "sscanf"} or tokens[i + 1][0] != "(":
            continue
        arguments, current, depth = [], [], 0
        for token, _ in tokens[i + 2:]:
            if token == ")" and depth == 0:
                arguments.append(current)
                break
            if token == "," and depth == 0:
                arguments.append(current)
                current = []
                continue
            if token in {"(", "[", "{"}:
                depth += 1
            elif token in {")", "]", "}"}:
                depth -= 1
            current.append(token)
        format_index = 0 if name == "scanf" else 1
        if len(arguments) <= format_index:
            continue
        parts = arguments[format_index]
        if not parts or not all(t.startswith('"') and t.endswith('"') for t in parts):
            continue
        fmt = "".join(t[1:-1] for t in parts)
        for conversion in _CONVERSION.finditer(fmt):
            kind = conversion["kind"]
            if (kind == "s" or kind.startswith("[")) and not (
                    conversion["suppress"] or conversion["width"]):
                result.add(source.count("\n", 0, offset) + 1)
    return result
