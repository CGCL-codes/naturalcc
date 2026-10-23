# Official benchmark evaluators

This folder contains upstream evaluator checkouts (the exact local HEAD revisions are recorded below):

- `SecRepoBench/` — upstream repository-level correctness and security evaluator, HEAD `759cb3031565cb34fc3d31473b3e6e4c587feb87`.
- `Defects4C/` — upstream C/C++ bug-fix benchmark and its remote/local verification client, HEAD `aecc2cf5f751d7c0894ae7d95ee0b8ae28e77b39`.
- `check_readiness.py` — read-only preflight for the NaturalCC run listed below.

Upstream repositories are cloned at the revisions recorded in their local `.git` metadata. Check with `git -C <folder> rev-parse HEAD` before citing a result; record that revision with any published score.

## Run being examined

The NaturalCC run is `test/benchmark_runs/easy-batch-20261023-163713-off` (relative to the project root). It contains 10 cases: 4 Defects4C and 6 SecRepoBench. A NaturalCC `completed` status only means the agent run ended; it is not an official pass.

## Official evaluation requirements

### SecRepoBench

The upstream evaluator reports developer-unit-test `pass@1`, vulnerability PoC results, `secure-pass@1`, compilation failures, and related counts. It runs task-specific `n132/arvo:<task-id>-fix` Docker images and replaces the complete repository file under test. This NaturalCC run used function-only workspaces (`target_function.c` / `buggy_function.c`), so its output cannot be handed to the upstream evaluator as a complete repository file without first integrating the function into the exact pinned source file. Do not copy a function-only file over the real repository file and call that an official result.

The six SecRepoBench images total about 6.9 GB of compressed layers according to the registry manifests queried on 2026-10-23; actual Docker storage can be larger. Docker Desktop must be running and have adequate storage. The upstream code uses the run outputs only if they are exported into its expected completion format; invoking `run_inference.py` would call a model again and would not evaluate this existing NaturalCC run.

After preparing full-file completions from this run, use the upstream evaluator from `SecRepoBench/` with only the selected IDs. The official entry point is `run_eval.py`; it needs `report.json` extracted from `report.json.gz`, `sample_metadata.json`, `assets/ids.txt`, completion files in the upstream naming format, the Python dependencies from `uv sync`, and working Docker access. The official output is written under `eval_results/`. See the upstream `README.md` and `run_eval.py`; do not run the default command unchanged because it targets the full benchmark rather than these six selected samples.

### Defects4C

The upstream project provides an online pass@1 verifier and a local Docker evaluator. The remote `https://defects4c.wj2ai.com/list_defects_bugid` endpoint returned HTTP 502 during this evaluation attempt (2026-10-23). The documented local setup downloads roughly 80 GB of repository data, which is not appropriate with about 30.3 GiB free on D: after checking out the evaluator sources. The official verifier therefore could not be reached or deployed in this environment.

## Current evaluation status

No official accuracy/pass@1 score is claimed yet. The Docker daemon is not available in this environment, and the Defects4C online service returned HTTP 502. These are evaluator-environment failures, not benchmark failures. Keep official score fields unscored until the official test harness returns a result. Do not treat unmodified/no-op samples or NaturalCC's `completed` label as official passes.

Run the read-only preflight from the project root:

```powershell
python "test/Metric-1_Code Completion and Bug Fixing/official/check_readiness.py"
```

It checks the recorded sample counts, upstream source checkouts, Docker daemon availability, D: free space, and the Defects4C verifier endpoint. It does not download Docker images, modify benchmark workspaces, call an LLM, or submit code to a service.

## Reproducibility notes

- Preserve the original NaturalCC run directory and its `results.jsonl`, `events.json`, `audit.json`, and workspaces.
- Keep codegraph set to `off` for this run.
- For a published score, include upstream commit hashes, exact selected task IDs, image digests, evaluator output, model/provider, and a note that the agent was run on function-only workspaces.
- Report completion/compilation, developer tests, and security PoC outcomes separately. A security finding count from NaturalCC's own tool is not a substitute for SecRepoBench's official PoC result.
