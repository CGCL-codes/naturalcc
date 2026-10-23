# NaturalCC benchmark runner

This small harness runs prepared benchmark task cards through NaturalCC's Agent Runtime. It records one isolated workspace copy per case, the run/thread IDs, model usage, changed-file hashes, and likely test-file edits. It does **not** claim benchmark correctness from the agent's final answer: attach each dataset's official evaluator result before reporting accuracy.

## Requirements

- Start the NaturalCC Agent Runtime API (default `http://127.0.0.1:7860`).
- Configure the model/provider in NaturalCC, or pass `--provider`, `--model`, and `--base-url`.
- Prepare each task as `CASE/TASK.md` plus `CASE/workspace/`. If there are multiple source files, provide `CASE/target_files.txt` (one workspace-relative path per line) or pass `--target-file`.
- Never place API keys in this file or in command history. Set `NATURALCC_API_KEY` (or `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` when using that provider) in the shell if the API expects the key in the request.

## Validate inputs without inference

Run these commands from the repository root in WSL or Git Bash. Prefer repository-relative paths; Windows drive paths are also accepted when the appropriate path converter is installed.

```bash
bash test/run_naturalcc_benchmark.sh \
  --dataset-dir 'test/Metric-1_Code Completion and Bug Fixing/BugFix/Defects4C' \
  --codegraph off --limit 1 --dry-run
```

Dry-run makes no HTTP/model calls and changes no files. If a dataset is not prepared as runnable `TASK.md + workspace/` cards, the runner stops instead of guessing or cloning repositories.

## Run the agent

```bash
bash test/run_naturalcc_benchmark.sh \
  --dataset-dir 'test/Metric-1_Code Completion and Bug Fixing/BugFix/Defects4C' \
  --codegraph off --limit 1 --auto-approve-all \
  --provider openrouter --model 'YOUR_MODEL_ID' \
  --max-llm-calls 30 --max-tool-calls 60 \
  --max-input-tokens 120000 --max-output-tokens 24000 --max-seconds 1800
```

Repeat with `--codegraph on` for a paired comparison. Keep the case order, prompt, model, budget, and dataset revision unchanged. By default the runner pauses for manual UI approval and needs an interactive terminal. The two modes should be run against separate copies; the harness does this automatically.

For an unattended benchmark run, add `--auto-approve-all`. This applies only to Runs created by this benchmark invocation, not normal UI conversations. It auto-approves all current approval-required risk classes (`write`, `execute`, `network`, `git_write`) and records each approval in the case manifest and events. **This is high risk:** execute/network actions run with the backend process's host privileges; workspace copies are not an OS security sandbox. Use only trusted task cards, preferably in a disposable VM/container. Omit the flag for manual UI approval.

Main options: `--dataset-dir` selects the prepared dataset; `--codegraph on|off` toggles graph capability; `--limit N` caps cases (0 means all); repeat `--target-file path` to specify editable files; `--api` changes the NaturalCC API address; `--output-dir` changes artifact location; `--provider/--model/--base-url` override model routing; `--max-llm-calls`, `--max-tool-calls`, `--max-input-tokens`, `--max-output-tokens`, and `--max-seconds` set per-run budgets. Defaults are documented by `--help`.

Artifacts are written under `test/benchmark_runs/<dataset>-<timestamp>-<mode>/`: `results.jsonl` and `summary.json` aggregate the batch; each case folder contains `manifest.json`, `TASK.md`, an isolated `workspace/`, `run_state.json`, `events.json`, `audit.json`, and `result.json`. Interrupted runs retain per-case artifacts; rerun in a new timestamped output directory rather than overwriting them.

## Official evaluation and metrics

Run the dataset's official evaluator on a clean copy of its pinned base workspace with the agent patch applied. Do not use modified benchmark tests. Save raw evaluator output, then normalize each scored result as JSONL:

```json
{"case_id":"example-1","codegraph":"off","official_pass":true,"official_metric":"pass@1","official_value":1,"official_result_source":"<evaluator command/log>"}
```

For the image requirement (self-check vulnerabilities <= 2 per 100 generated nonblank LOC), export the NaturalCC scanner's findings to a second JSONL:

```json
{"case_id":"example-1","codegraph":"off","generated_nonblank_loc":120,"self_check_vulnerability_findings":1,"security_result_source":"<scanner and version>"}
```

Then summarize (include both `off` and `on` result files for paired analysis):

```bash
python3 test/summarize_naturalcc_benchmark.py \
  test/benchmark_runs/RUN_OFF/results.jsonl test/benchmark_runs/RUN_ON/results.jsonl \
  --official-results official-results.jsonl \
  --security-results security-results.jsonl
```

The report includes runtime completion, official pass rate, paired CodeGraph outcomes, token/cost totals when supplied by Runtime, test-path-change flags, and the scanner finding density. Missing official/scanner rows stay unscored. The test-path audit is a heuristic warning, not a security boundary; the evaluator must still run against clean tests. Benchmark-specific setup (for example full repository checkout, Docker image, pinned revision, and evaluator) is intentionally not replaced by a generic script.

## Local checks

```bash
python3 -m unittest test/test_naturalcc_benchmark.py -v
python3 test/naturalcc_benchmark.py --help
```
