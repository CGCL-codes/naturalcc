# SWE-bench-style xarray repair demo

This is a small, self-contained UI demo derived from the real SWE-bench task
`pydata__xarray-4629`. It deliberately starts with the same class of bug:
the merge result aliases a source metadata dictionary.

It is **not** the full xarray repository and must not be reported as an
official SWE-bench score. Its purpose is a reliable live demonstration of the
NaturalCC Agent UI.

## Demonstration sequence

1. Set this directory as the Agent workspace.
2. Add `ISSUE.md` and `mini_xarray.py` as context.
3. Ask the Agent to diagnose and minimally repair the aliasing bug.
4. Review the proposed one-line production diff before approving it.
5. Run the following commands after the approved edit:

   ```powershell
   python demo_repro.py
   python -m pytest -q
   ```

   When running manually from a terminal whose Python does not include pytest,
   use NaturalCC's prepared environment instead:

   ```powershell
   uv run --project D:\桌面2\naturalcc-ncc3\code_agent python -m pytest -q
   ```

   In the Web UI, the backend is started through that same `uv` environment,
   so the Agent can use the shorter `python -m pytest -q` command.

Before the repair, both commands fail. After the intended repair, they pass.

## Suggested UI prompt

```text
Read ISSUE.md and repair the bug with the smallest possible change.
First explain the root cause. Only modify mini_xarray.py; do not modify tests
or demo_repro.py, and preserve the existing formatting. Before editing or
running commands, request my approval. Then verify with python demo_repro.py
and python -m pytest -q.
```
