# Pipeline Plugins in Agent Runs

The default Agent registry wraps the actual Pipeline plugin classes. Create a Run
or send a thread message through `/api/agent/*`; the model selects registered
tools. No separate `/api/agent/code_completion` endpoint is needed. The existing
`POST /api/run` dispatcher remains available.

| Canonical tool | Model tool name | Approval | Behavior |
| --- | --- | --- | --- |
| `code_completion` | `code_completion` | write | CodeCompletionPlugin -> NaturalCC prompt -> Aider |
| `vulnerability_detection` | `vulnerability_detection` | none | Built-in patterns, C/C++ shell-string/shared-global checks, review checklist and optional TSan log import |
| `vulnerability_detection.analyze` | `vulnerability_detection_analyze` | execute | Required Cppcheck; optional `analyzer=comprehensive` also requires Clang |
| `vulnerability_detection.fix` | `vulnerability_detection_fix` | execute | Scan (Cppcheck if available), then Aider repair |

Completion arguments:

```json
{"target_files":["src/main.c"],"instruction":"Complete parse_input","symbol":"parse_input","completion_type":"function_body"}
```

Scan arguments (both scan tools):

```json
{"scan_scope":"project","incremental":true,"severity_threshold":"medium","max_findings":30,"scan_type":"frequent_defects"}
```

`scan_type` is optional and accepts `frequent_defects` or `high_risk`. Omit it to
preserve the existing scan. The Agent tool schema accepts only those two enum
values; the Pipeline plugin also treats an empty value as unspecified. At this
stage it selects the categories included in `contract_statistics`, while the full
finding list retains other diagnostics. Supply `ground_truth_file` (a workspace-relative,
versioned, source-hashed JSON manifest) to calculate case-level metrics.
Without labels `contract_statistics.status` remains `not_evaluated`.
See [security integration](SECURITY_ACCEPTANCE.md) for the schema and formulas.

The result's `finding_summary` counts candidate records after the severity
threshold and after combining the selected analyzers, before `max_findings` is
applied. It reports `candidate_count`, `returned_count`, `truncated`, and
`max_findings`; the returned `findings` and report body contain at most that many
records. These are scanner candidates, not confirmed defects or TP/FN/FP. CWE
identifiers are analyzer labels. `category`, `category_label`, `metric_eligible`
and `classification_basis` provide the versioned mapping based on analyzer IDs,
diagnostic types and source operations, rather than CWE alone. All metrics use
the complete candidate list before truncation, including with the incremental cache.

Repair arguments:

```json
{"target_files":["src/main.c"],"instruction":"Repair vulnerabilities while preserving the public interface"}
```

The Run supplies workspace, target authorization, model and ephemeral API key;
model-generated arguments cannot override these fields. Plugin targets must be
existing files under the workspace. Selected Run targets restrict mutating calls.
Mutations snapshot selected files and report actual diffs even on plugin failure;
these feed existing verification and CodeGraph dirty-file tracking. Aider retains
its existing process model: this is not an OS sandbox. Child-model usage is not
separately included in the parent token counter, and cancellation of a blocked
Aider stream follows the existing runner behavior.

## Detection Coverage

| Request | Implementation | Limits |
| --- | --- | --- |
| Array out-of-bounds | Cppcheck XML diagnostics including `arrayIndexOutOfBounds` | Installed Cppcheck required; C/C++ |
| Null pointers | Cppcheck `nullPointer` family | Static diagnostics, not proof for every path |
| Memory leaks | Cppcheck `memleak` family | Ownership and cross-file coverage may be incomplete |
| Incremental analysis | SHA-256 keyed bounded in-process rule cache | Local-rule results only; restart clears cache |
| C/C++ command execution | Local string input/parameter flow into `system`/`popen` | Syntax-based candidates, no general cross-file taint or custom sanitizer analysis |
| Static races | Conflicting global scalar accesses in overlapping `pthread` / `std::thread` lifetimes | Direct accesses and recognized mutexes/atomics only; no general alias or path analysis |
| Comprehensive bounds/null/leak checks | Cppcheck plus Clang Static Analyzer | Opt-in; LLVM 18 validated, experimental bounds checkers exposed in coverage |
| Non-reentrant-call hints | Calls in files containing thread creation | Retained as review hints, excluded from metrics |
| Runtime races | Import a ThreadSanitizer log | No automatic instrumented build/run |

### Contract metrics and ownership

NaturalCC supplies analyzer candidates, coverage, the report, category fields and
optional case-level statistics. The platform backend must pass through `scan_type`,
`ground_truth_file`, `analyzer` and the returned `artifacts.contract_statistics` /
`finding_summary`. This repository does not contain that separate backend.
The prepared OS test projects and ground truth are documented in `SECURITY_ACCEPTANCE.md`.
They support reproducible integration checks; final acceptance still depends on an
agreed corpus and build configuration. No pass/fail threshold is returned.

Current evidence and limits:

| Indicator / target | Current NaturalCC capability and evidence | Boundary |
| --- | --- | --- |
| 1: generated-code self-check | The platform can scan generated files and return candidate findings/coverage. | The contract limit is at most 2 vulnerabilities per 100 lines. No verified code-line denominator, counting scope, or method for confirming candidates as vulnerabilities is available, so the current scan cannot establish this density. |
| 3: array bounds, string overflow, null-pointer calls | The Web checkpoints in `feb0a67` record balanced-set recall of 87.22% overall (86.67% array, 100.00% string, 75.00% null), with 0% FPR. The independent Web set records 89.44% overall recall and 2.22% FPR. | These are separate saved model-benchmark runs, not a rerun of the merged scanner. They do not establish `<5 s` response time or complete upstream-project coverage. New comprehensive-scanner fixture results are retained separately under `artifacts/security-acceptance/`; do not average or substitute these datasets. |
| 5: buffer overflow, data races, leaks, command execution | Built-in C/C++ source rules now cover local shell-string flow and direct shared-global conflicts; comprehensive analysis adds Clang bounds/null/leak evidence. | The previous Contract-5 Web report (92.50% accuracy, 92.00% precision, 95.83% recall, 12.50% FPR) is a separate historical experiment. New fixed integration-case results are recorded separately under `artifacts/security-acceptance/`; neither establishes independent generalization or whole-project coverage. |

For target coverage, array bounds and buffer/string overflow overlap at the
analyzer level; use the versioned operation-based mapping and case-level matching
rule in `security_contracts.py`. Static race checks are bounded checks, not general
race detection. TSan import only parses an existing log and does not verify its
revision or authenticity. Neither an empty result nor completed analyzer coverage
proves the absence of defects.

Cppcheck uses an argv list, no shell, and a 120-second timeout; it does not execute
the target program. This version uses default compiler configuration, not the
project compilation database. Analyze the whole project for broader coverage.
Reports list unavailable analyzers, failures and coverage limits.
`analyzer=cppcheck` / `comprehensive` fail explicitly when a required analyzer cannot run;
Pipeline's `analyzer=auto` retains built-in results and reports missing coverage.

Install [Cppcheck](https://cppcheck.sourceforge.io/) on the backend host, expose
`cppcheck` on the service PATH and restart the backend. Completion/repair require
the existing Aider and NaturalCC/libclang setup. Installation on the browser
client alone is insufficient.

For the comprehensive profile, also install Clang (LLVM 18 validated). This profile
uses `core`, `unix`, `cplusplus`, `alpha.security.ArrayBoundV2` and
`alpha.unix.cstring.OutOfBounds`, writes analyzer reports in a temporary directory,
and never runs the target program. The two alpha checkers are experimental and
their findings/coverage explicitly say so. Each file has a 15-second deadline and
the Clang phase a 120-second total deadline; unprocessed files are reported.

Incremental caching is shared across plugin instances in one process, bounded to
256 entries. It keys content, root, path, rule definitions, threshold and context
settings. Edits invalidate file results; new files are scanned; deleted files are
absent from the next report. Unchanged findings remain in reports. Cppcheck always
reanalyzes selected C/C++ files; local cache results do not stand in for
dependency-aware analysis.

For runtime races, build/run suitable tests with ThreadSanitizer on a supported
platform (for example Clang on Linux), save stderr in the workspace, then request:

```json
{"target_files":["src/main.cpp"],"sanitizer_report":"artifacts/tsan.log"}
```

The importer recognizes workspace stack frames in standard
`WARNING: ThreadSanitizer: data race` reports; runtime frames are excluded. Logs
must stay in the workspace and are size-limited. Log revision/authenticity are not
verified; no findings does not establish race freedom. Paths containing whitespace
or emitted in a different checkout may need normalization.

References: [Cppcheck manual](https://cppcheck.sourceforge.io/manual.html),
[ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html).

## Verification

```powershell
uv run --project code_agent python -m pytest code_agent/tests/test_pipeline_tools.py code_agent/tests/test_security_analysis.py code_agent/test_vulnerability_detection.py -q
```

Tests cover API approval -> dispatch -> snapshots, diffs on failures, authority
rejection, analyzer diagnostics/failures, cache invalidation, race-log import and
remediation failure. The real Cppcheck fixture runs when the executable is
installed. Deterministic tests replace Aider/LLM calls; live completion quality
depends on the configured model and parser.
