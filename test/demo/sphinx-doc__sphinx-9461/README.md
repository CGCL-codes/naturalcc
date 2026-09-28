# Multi-file Sphinx autodoc repair demo

This self-contained project is inspired by the real SWE-bench task
`sphinx-doc__sphinx-9461`: documentation generation fails for a descriptor
written as `@classmethod` wrapping `@property`.

It deliberately begins in a broken state. The user-visible symptom appears in
`demo_repro.py`; the underlying behavior spans descriptor inspection,
autodoc member selection, and final domain rendering.

This is a compact teaching workspace, not the full Sphinx repository. It is
appropriate for demonstrating multi-file reasoning and approvals in the
NaturalCC UI, but it is not an official SWE-bench evaluation artifact.

## UI workspace

```text
D:\桌面2\naturalcc-ncc3\test\demo\sphinx-doc__sphinx-9461
```

## Demo sequence

1. Add `ISSUE.md`, `mini_sphinx/autodoc.py`, and `mini_sphinx/inspect.py` as
   initial UI context.
2. Ask the Agent to trace the issue across the three files before proposing a
   patch.
3. Review each requested edit in Run details. The intended repair is small,
   but should touch the relationship between all three layers.
4. Approve the test commands after reviewing them.

Run these checks after the repair:

```powershell
python demo_repro.py
python -m pytest -q
```

If a regular terminal lacks pytest, use NaturalCC's environment:

```powershell
uv run --project D:\桌面2\naturalcc-ncc3\code_agent python -m pytest -q
```

## Suggested UI prompt

```text
Read ISSUE.md and diagnose the class-property autodoc bug across the
mini_sphinx package. Before editing, explain the descriptor flow from
inspect.py through autodoc.py to domain.py. Make the smallest production-code
changes necessary inside mini_sphinx/ only. Do not modify demo_targets.py,
demo_repro.py, or tests. Preserve ordinary property behavior and existing
formatting. Request my approval before writing files or running commands.
After approval, verify with python demo_repro.py and python -m pytest -q.
```

