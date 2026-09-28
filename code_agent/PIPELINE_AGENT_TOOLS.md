# Pipeline Plugins in Agent Runs

The default Agent registry wraps the actual Pipeline plugin classes. Create a Run
or send a thread message through `/api/agent/*`; the model selects registered
tools. No separate `/api/agent/code_completion` endpoint is needed. The existing
`POST /api/run` dispatcher remains available.

| Canonical tool | Model tool name | Approval | Behavior |
| --- | --- | --- | --- |
| `code_completion` | `code_completion` | write | CodeCompletionPlugin -> NaturalCC prompt -> Aider |
| `vulnerability_detection` | `vulnerability_detection` | none | Built-in candidates, C semantic-review checklist and optional TSan log import |
| `vulnerability_detection.analyze` | `vulnerability_detection_analyze` | execute | Built-in rules plus required Cppcheck analysis |
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
stage it is only echoed in the report and artifacts as request context; it does
not select rules, filter findings, or calculate contract metrics. When present,
`contract_statistics.status` is `not_evaluated` with the missing-evidence reason.

The result's `finding_summary` counts candidate records after the severity
threshold and after combining the selected analyzers, before `max_findings` is
applied. It reports `candidate_count`, `returned_count`, `truncated`, and
`max_findings`; the returned `findings` and report body contain at most that many
records. These are scanner candidates, not confirmed defects or TP/FN/FP. CWE
identifiers are analyzer labels and are not automatically mapped to contract
categories.

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
| Race risks | Non-reentrant calls in files containing thread creation | Low-confidence review hints, not general race detection |
| Runtime races | Import a ThreadSanitizer log | No automatic instrumented build/run |

### Contract metrics and ownership

NaturalCC supplies analyzer candidates, coverage, the report, and the optional
scan context above. The platform backend owns task/API forwarding and must
explicitly pass through any new NaturalCC artifacts it wants to expose. In the
backend snapshot checked on 2026-09-28, vulnerability tasks send no `scan_type`
to `/api/run` and return only `findings`, `coverage`, `report`, and `execution`;
they do not expose `finding_summary` or the other new artifacts. The contract or
acceptance owner must provide the approved test projects, ground truth, matching
rules and metric formulas before formal TP/FN/FP or pass/fail values can be
reported. NaturalCC's candidate output alone does not establish those values.

Current evidence and limits:

| Indicator / target | Current NaturalCC capability and evidence | Boundary |
| --- | --- | --- |
| 1: generated-code self-check | The platform can scan generated files and return candidate findings/coverage. | The contract limit is at most 2 vulnerabilities per 100 lines. No verified code-line denominator, counting scope, or method for confirming candidates as vulnerabilities is available, so the current scan cannot establish this density. |
| 3: array bounds, string overflow, null-pointer calls | Built-in C/C++ patterns and optional Cppcheck diagnostics include array-index and null-pointer families; Cppcheck can report memory/bounds diagnostics. The existing Contract-3 Web fixture report records 87.78% overall recall, 88.33% array, 95.00% string, and 80.00% null-pointer recall, with 0% FPR under that report's fixture-specific labels. | The reported null-pointer category is below its `>85%` target. The report does not establish `<5 s` response time or complete upstream-project coverage. Its fixtures and scoring rules are not the contract's final acceptance environment; contract text also needs clarification on the `<=10,000` versus `>=10,000` statement-count requirement. |
| 5: buffer overflow, data races, leaks, command execution | C/C++ built-ins include buffer-related patterns and a low-confidence thread/non-reentrant-call hint; Cppcheck can emit bounds and leak diagnostics. TSan findings are imported from a user-supplied log. | The built-in C/C++ `cwe-78` command-injection rule is not enabled for C/C++; no automatic instrumented build/run is performed. Existing Contract-5 Web report uses 40 file-level samples (6 positive and 4 negative per category), reports 92.50% accuracy, 92.00% precision, 95.83% recall and 12.50% FPR; by its proxy scoring, FPR exceeds `<8%`. It is a tuned, non-independent experiment and not formal acceptance. |

For target coverage, array bounds and buffer/string overflow overlap at the
analyzer level; do not count a CWE label as a contract-category match without an
approved mapping and case-level matching rule. Static race hints are not general
race detection. TSan import only parses an existing log and does not verify its
revision or authenticity. Neither an empty result nor completed analyzer coverage
proves the absence of defects.

Cppcheck uses an argv list, no shell, and a 120-second timeout; it does not execute
the target program. This version uses default compiler configuration, not the
project compilation database. Analyze the whole project for broader coverage.
Reports list unavailable analyzers, failures and coverage limits.
`analyzer=cppcheck` fails explicitly when the required analyzer cannot run;
Pipeline's `analyzer=auto` retains built-in results and reports missing coverage.

Install [Cppcheck](https://cppcheck.sourceforge.io/) on the backend host, expose
`cppcheck` on the service PATH and restart the backend. Completion/repair require
the existing Aider and NaturalCC/libclang setup. Installation on the browser
client alone is insufficient.

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
