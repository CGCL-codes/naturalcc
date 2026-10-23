# NaturalCC challenge sample pool

This directory contains 40 benchmark task cards: 10 from each benchmark. The selection rules and per-task IDs, language/CWE, repository, commit, and heuristic scores are recorded in each dataset's `selection.json`.

The selected cases are intended for a small, challenging evaluation pool and UI demonstrations. They are not a random sample and must not be presented as a representative estimate of benchmark-wide performance. Official scoring requires each dataset's official evaluator and pinned environment. Gold patches and hidden tests were excluded from the agent-visible folders.

## Pools

- `CodeCompletion/SecRepoBench/`: 10 secure repository-level C/C++ completion tasks; diversity across CWE/project, with target-function workspace.
- `BugFix/SWE-bench-Multilingual/`: 10 issue-resolution tasks; coverage across the nine language families in the dataset, plus one additional high-scoring task.
- `BugFix/Defects4C/`: 6 ordinary C/C++ bug-fix functions and 4 vulnerability-fix functions, spread across projects.
- `Refactoring/SWE-Atlas-Refactoring/`: 10 official hard refactoring task cards, prioritizing language diversity and larger reference change scope.

The task card folders do not include full SWE-bench or SWE-Atlas repository checkouts. Defects4C includes only the target function. SecRepoBench includes only the masked target function. See each dataset README for official validation requirements.
