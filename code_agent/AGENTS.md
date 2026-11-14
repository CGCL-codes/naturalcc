# AGENTS.md

Working map for coding agents. Keep this file short; update it with `README.md` and `README.zh.md` when architecture, commands, dependencies, or behavior meaningfully change.

## Purpose

`code_agent` is a NaturalCC-enhanced code editing agent. It parses a target project, builds a semantic prompt, then delegates edits or dry-run reports to Aider.

Common tasks: code completion, small project-aware edits/refactors, code summarization, code repair, vulnerability detection, and optional vulnerability remediation.

## Active Entry Points

- `agent_web_api.py`: FastAPI backend, bundled UI server, workspace scan/browse, prompt preview, NDJSON Aider streaming.
- `model_vulnerability_scan.py`, `api/model_scan_routes.py`: tool-free model audits via `/api/vulnerability/model` or `/api/run` with `feature=vulnerability_detection`, `feature_config.detection_mode=model`. Supports project/multiple-file scopes, tokenizer-bounded overlapping source chunks, streamed model deltas, source-hashed GT scoring outside model prompts, and Pipeline-compatible artifacts. No static analyzers/Aider; failed or incomplete audits never publish GT rates. See `模型辅助漏洞检测接入教程.md`; the existing Contract-5 Agent benchmark remains a separate protocol.
- `api/agent_routes.py`: durable `/api/agent/*` conversation, run, context, budget, approval, event, cancellation, and memory routes.
- `agent_core/run_engine.py`: persisted autonomous run state machine.
- `agent_core/contracts.py`, `tool_registry.py`, `policy.py`: trusted tool contracts and permission boundary.
- `agent_core/event_store.py`, `memory_store.py`: SQLite conversation/message/event/snapshot/lease storage, governed memory proposals, review audit, goal-independent pinned memory, and active-only FTS5 retrieval memory.
- `agent_core/memory_proposals.py`, `memory_prompts.py`, `memory_projection.py`: frozen evidence, tool-free two-pass proposal generation, validation, and deterministic user-facing review DTOs.
- `agent_core/model_gateway.py`: explicit per-Run provider routing (DeepSeek/OpenRouter and loopback-only Ollama) while preserving the `ModelGateway` Harness contract.
- Agent Runs use the optional Gateway `stream()` path to emit batched `model.reasoning.delta` and `model.content.delta` events; `generate()` remains available for maintenance calls and existing gateways. The NDJSON endpoint follows active Runs and the UI renders returned reasoning in a collapsible panel.
- `agent_core/token_budget.py`, `context_builder.py`: pinned DeepSeek offline estimation, model-specific context geometry, full-request hard budgets, cache-friendly stable/dynamic prompt tiers, and continuous-tail ContextPlans.
- `agent_core/compaction.py`, `compaction_prompts.py`: versioned Analyzer/Summarizer checkpoints, validation, recovery, and deterministic fallback.
- `webui/src/App.jsx`, `MemoryReviewPanel.jsx`, `MemoryProposalCard.jsx`, `webui/src/styles.css`: Codex-style React workbench with conversation history, evidence selection, governed memory review, budget meters, chat output, and Run details.
- `webui/src/conversationState.js`, `budgetUi.js`: deterministic thread hydration, context payload, grouping, call/input-token budget validation, and progress helpers.
- `vscode-extension.js`, `vscode_server.py`: VS Code webview host and local-service bootstrap. Package with `npm run package` after building `webui/dist`.
- `aider_runner.py`: CLI, NaturalCC prompt generation, Aider command construction/streaming. Keep CLI behavior stable unless requested.
- `completion_prompt_agent.py`: prompt construction, symbol and completion-type inference.
- `plugins/`: Feature Plugin System.
- `rag/c/`: C/C++ parser, project graph, context retrieval. Avoid offline eval scripts unless the bug is there.
- `rag/java/`: Java parser/prompt path.
- `test_api.py`: API connectivity check, not a unit suite.
- `security_rtos_rules.py`: RTOS-aware source checks for the four Contract-5 high-risk categories (buffer_overflow/787, data_race/362, memory_leak/401, command_execution/78). It complements `security_c_rules.py`, which only knows the libc spelling of each API and only tracks plain global scalars; on a kernel the same operations appear as `rt_memcpy`/`rt_spin_lock_irqsave` with shared state in object fields, so those rules see nothing. It tolerates macro-induced parse errors in the enclosing function and checks each matched node instead. Findings carry `analyzer: "c-source"` with `rule_id` `cwe-787|362|401|78`; Contract-3 matches findings by CWE number, so a `cwe-787` finding counts there as `string_overflow` if it lands inside a Contract-3 ground-truth function. Wired into `plugins/vulnerability_detection.py` for every C/C++ scan outside the `web` profile.
- Contract-3 is a detector-optimization teaching exercise. Current dataset revision is teaching-v3-20261113; A/B/C/E/F/G/H/I are off and D is on. Use scripts/measure_contract3_static.py --delivered for per-project/per-category evidence and --reference for full rules without changing the student default. Keep revised Web checkpoints in revision-specific directories. The full-rules reset is the reference solution, not the student baseline. See `合同-3-静态分析难度调节指南.md`; Contract-5 and ordinary scans retain full rules.
- `scripts/measure_contract5_static.py`: offline Contract-5 scoring of the built-in rules alone (no model, no network, no Cppcheck). Same site-matching rule as the Web benchmark, so it is the fast loop for tuning detection; `--misses` lists every FN/FP.
- `scripts/build_contract5_dataset.py`: builds the Contract-5 whole-project dataset (RT-Thread v5.2.1 subset, 10 injected defects per category + safe controls) from `artifacts/contract5/rtthread/{dataset_spec.json,injections/*.json}` into `artifacts/contract5/rtthread/dataset/{project/,ground_truth.json,manifest.json}`; `--check` validates anchors without upstream writes, `--verify` checks the committed output against the manifest, `--build-context` re-applies `overlay/rtconfig.h` and regenerates `compile_commands.json` from `dataset_spec.json`'s `build_context` without upstream, `--compile-check` compiles every detection file with clang and lists GT cases the configuration compiles out (expected: CE-10, LEAK-10, RACE-06), `--archive` writes `contract5.zip`. Ground truth never goes into the Agent workspace.
- `scripts/benchmark_contract5_web.py`: Contract-5 evaluation through the Web Agent API, one Thread per ground-truth target file over the injected project; site-level (function/line-range) and file-level scoring; resumable; stops on insufficient credits; `--model` selects the OpenRouter model (output dir `web-<model>/`), `--print-prompts`/`--report-only` need no network. Prompt text lives in `artifacts/contract5/rtthread/prompt.md`. Report: `合同-5-result.md`.
- `scripts/benchmark_contract3_cppcheck.py`: retained only as shared Contract-3 helper code; its simple baseline CLI is retired and exits with the balanced-profile command.
- `scripts/benchmark_contract3_web.py`: optional Contract-5-aligned Web-Agent version. It requires the local Web backend and OpenRouter key, passes `scan_type=frequent_defects`, and keeps ground truth outside the Agent workspace. It never auto-approves execute/write tools; the script scores final model predictions separately from scanner `contract_statistics` and reports raw metrics without threshold verdicts. `--suite balanced` is the recommended acceptance mix; `--suite independent` is the stress mix; `--limit 1` is a low-cost 40-function smoke run.
- `scripts/benchmark_contract3_independent.py`: versioned Contract-3 teaching fixture generator plus historical Cppcheck-only adapter. The four existing profiles use teacher-calibrated shape counts; do not call them held-out generalization sets. Cppcheck outputs include the dataset revision in their directory. Use `measure_contract3_static.py --delivered` for the actual comprehensive teaching score; keep labels outside model workspaces.
- `scripts/render_contract3_report.py`: the only Contract-3 Markdown renderer. It summarizes separate raw artifacts without collapsing their results into one score.

There is no legacy UI path; keep graphical work centered on `agent_web_api.py` and `webui/`.

## Runtime Notes

- Runtime Threads carry an explicit `runtime_model_config` (provider, model, endpoint, window, and routing preferences); each Run snapshots it. API keys remain request-scoped memory only.
- For offline platform indicators 1/3/5, indicator 1 uses the Agent Gateway and approved Aider edit tools with local Ollama; indicators 3/5 use `/api/run` scanning and must keep `auto_fix=false` to avoid Aider/model calls. Keep both paths independent; follow `OFFLINE_PLATFORM_1_3_5.md` for dependency prep, smoke checks, and the no-acceptance boundary.
- UI refresh restores only the last selected Thread after Bootstrap; New conversation uses the separately stored Bootstrap Runtime defaults. Thread settings must never overwrite those launch defaults.
- First target file is the NaturalCC primary parse file, even when Aider receives multiple files.
- NaturalCC parsing must work; do not silently bypass it for completion/repair flows.
- Python `clang` bindings must match system `libclang`; this repo pins `clang==18.1.8` for LLVM 18 / `libclang1-18`.

## Plugins

Plugins auto-register from `plugins/` via `@register_plugin`; the frontend renders each plugin's `config_schema`.

- `plugins/base.py`: `FeaturePlugin`, `ExecutionMode` (`aider`/`direct`/`hybrid`), `ConfigField`, `ExecutionContext`.
- `plugins/registry.py`: registration and schema listing.
- `plugins/dispatcher.py`: routes AIDER, DIRECT, and HYBRID execution.
- `plugins/code_completion.py`: original NaturalCC completion flow.
- `plugins/code_summary.py`: NaturalCC semantic summaries via Aider `--dry-run`.
- `plugins/code_repair.py`: focused repair prompt via Aider.
- `plugins/vulnerability_detection.py`: static scan with optional Aider remediation.
- `agent_core/tools/pipeline.py`: Agent adapters for completion, scan, Cppcheck analysis and repair. `security_analysis.py` adapts analyzer evidence; see `PIPELINE_AGENT_TOOLS.md` for setup and limits.
- `security_review.py`: dataset-independent C audit checklist and literal `scanf` format candidates, returned by the vulnerability plugin to both Pipeline and Web Agent users. Review guidance is not a proven finding; do not classify unrelated defects as target CWEs.
- `security_c_rules.py`: bounded syntax-based C/C++ shell-string flow and overlapping-thread global-access checks; no target execution. `parse_source` runs `security_c_macros.neutralize` first.
- `security_c_macros.py`: offset-preserving blanking of macro uses tree-sitter cannot parse (attribute macros, top-level `MSH_CMD_EXPORT(...)` registrations, type arguments as in `rt_list_entry(n, struct x, m)`, `for_each` iterators, `#if` arms that split a statement). Only spaces replace characters, so lines/columns never move. The plugin adds attribute macros harvested from the project's headers. The built-in vocabulary must stay conservative: a name that doubles as an identifier or goto label (`__exit`) must not be listed.
- `security_build_context.py`: compile context for Clang/Cppcheck: `compile_commands.json` (relative `directory` resolved against the file's folder), then `compile_flags.txt`, then include dirs inferred from `#include` (host-shadowing dirs go to `-idirafter`); `include_dirs`/`defines` from feature config are appended. Untrusted input: only include/macro/`-std` flags pass, every path must resolve inside the project; database violations are dropped and reported, config violations raise `BuildContextError`. Contract-3 static switch anchors live in `security_contracts.py`/`security_analysis.py`/`security_c_rules.py`; keep those exact lines unchanged (`scripts/contract3_static_switches.py status` verifies them). `security_contracts.py`: versioned seven-category mapping and hash-checked case-level ground-truth scoring before truncation; never derive TP/FN/FP from CWE labels or candidate counts alone.
- `analyzer=comprehensive` requires Cppcheck and Clang (validated with LLVM 18; bounds checkers are experimental). Agent `.analyze` retains execute approval; read-only scans never launch analyzers. `scripts/prepare_security_acceptance.py` and `check_security_acceptance.py` prepare pinned OS source trees and reproduce plugin/API scoring. See `SECURITY_ACCEPTANCE.md`; fixed fixture scores are not whole-project acceptance or generalization scores.

Add a plugin by creating `plugins/my_feature.py`, inheriting `FeaturePlugin`, implementing `metadata`, `config_schema`, `execute`, and decorating with `@register_plugin`.

## Commands

Backend comment coverage: `python scripts/measure_backend_comments.py --json backend_comment_coverage.json` counts Python `#` comment lines across tracked application and RAG modules, excluding dependencies, datasets, tests, frontend and maintenance scripts. Docstrings and prompt strings do not count; the aggregate must be strictly greater than 20%. Keep the report current when changing these annotations.

```bash
uv sync
uv run python scripts/install_deepseek_tokenizer.py
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
cd webui && npm install && npm run dev
cd webui && npm run build
npm run package
uv run python aider_runner.py -dir /path/to/project -f src/foo.c -i "补全 foo 函数实现" --preview
```

## Constraints

- `aider` must be on `PATH`.
- `libclang` must be installed for C/C++ parsing.
- C++ coverage is incomplete because parts of the parser still use C-oriented libclang arguments.
- Run the deterministic suite with `uv run --project . pytest tests test_vulnerability_detection.py -q`; also run the Web UI reducer tests and production build.
- Keep changes surgical; prefer existing patterns; do not refactor unrelated code.
- Agent mode reads automatically; write and execute tools require a run-scoped approval.
- Verification accepts `tests.run` and equivalent discovered full-suite commands executed by `command.run` at the workspace root. Pytest quiet/verbose flags are allowed; selectors and collection-only commands do not satisfy verification. A second unverified final answer after a reminder fails with `VerificationRequired` rather than looping to budget exhaustion.
- Approval UI verifies the current snapshot and sends `tool_call_id` to reject stale prompts. Missing Runs return 404; missing snapshots and mismatched approvals return 409. Never recreate or execute a missing Run implicitly.
- `workspace.create_file` creates only a new UTF-8 text file: its parent must already exist, it never overwrites, and its absent-state snapshot lets `workspace.restore_snapshot` remove the run-created file.
- `workspace.create_directory` creates a directory or an explicit `parents=true` chain, treats existing directories as idempotent success, and snapshots every newly created directory; restore removes only empty directories.
- Each user turn creates a Run under a durable thread. Raw chat/events remain authoritative; only committed Run/Thread checkpoints are injected as derived context, and project memories remain governed candidates.
- Long-term memory uses frozen evidence, validation, and audited approval. Explicit durable user guidance can go directly to composition. Verified runs and manual selections retain the analysis pass. Each model stage may use one schema repair. A repeated missing-argument error followed by a successful corrected call can also yield a project-scoped lesson from the runtime trace. Extraction outcomes are separate from the Run. `CODE_AGENT_AUTO_MEMORY_MODE` supports `auto` (default), `review`, and `off`. Manual selections and ambiguous proposals require user approval. Never expose raw Proposal JSON or index pending proposals.
- Active memory uses a stable prefix for pinned kinds and goal-based FTS5 retrieval for other kinds. On a Chinese-query miss, a budgeted, tool-free semantic selector can return IDs from a bounded list of visible and unexpired candidates. Revalidate selected IDs before reuse. Never duplicate pinned entries in the retrieved tier.
- Keep the model prompt ordered as stable system rules -> pinned memory -> committed checkpoint -> runtime authorization -> current goal -> continuous recent message tail -> a final runtime-generated context message containing retrieved memory and WorkingState. The final message reports status, never grants permission or creates a new user request. Keep per-call counters and tool results out of the leading system message so prior turns remain a reusable cache prefix.
- Persist provider-reported `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` in Run state and `model.usage_recorded`; absence means “not reported”, not a zero-percent cache rate.
- SQLite schema v6 creates a non-overwriting `.before-v6-memory-proposals.bak` snapshot before migrating an existing v1-v5 database.
- Count the serialized messages, tool schemas, provider framing, output reserve, and safety margin with the pinned local DeepSeek V3 tokenizer before every Agent-model request. Calibrate estimates against provider-reported usage when using newer DeepSeek model versions or other providers; never silently fall back to character budgets or delete protected rules/WorkingState to make a request fit.
- Compaction runs only at safe points over continuous completed prefixes. Analyzer/Summarizer/repair calls use no tools, have separate `max_compaction_calls`, and must not consume `max_llm_calls`; their token/cost/time usage still contributes to total budgets.
- Thread deletion is permanent and transactional. Reject deletion when any Run is queued, running, waiting for approval, or paused; the UI must require an explicit cancellation action before deletion. Terminal Runs cannot transition back to active states.
- Only user-selected absolute paths may widen filesystem access beyond the primary workspace; model-generated paths cannot widen authorization.
- The argv command policy allows `gcc`, `g++`, `c++`, and `clang++`, but is not an OS sandbox; use a container or restricted account for untrusted repositories.
- Never pass workspace roots, API keys, artifact roots, or permissions through model-generated tool arguments.
- The legacy `/api/run` Pipeline and the durable `/api/agent/*` Runtime must remain independently usable.
