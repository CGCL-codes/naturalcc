# Pipeline Plugins in Agent Runs

The default Agent registry wraps the actual Pipeline plugin classes. Create a Run
or send a thread message through `/api/agent/*`; the model selects registered
tools. No separate `/api/agent/code_completion` endpoint is needed. The existing
`POST /api/run` dispatcher remains available.

| Canonical tool | Model tool name | Approval | Behavior |
| --- | --- | --- | --- |
| `code_completion` | `code_completion` | write | CodeCompletionPlugin -> NaturalCC prompt -> Aider |
| `vulnerability_detection` | `vulnerability_detection` | none | Built-in rules and optional TSan log import |
| `vulnerability_detection.analyze` | `vulnerability_detection_analyze` | execute | Built-in rules plus required Cppcheck analysis |
| `vulnerability_detection.fix` | `vulnerability_detection_fix` | execute | Scan (Cppcheck if available), then Aider repair |

Completion arguments:

```json
{"target_files":["src/main.c"],"instruction":"Complete parse_input","symbol":"parse_input","completion_type":"function_body"}
```

Scan arguments (both scan tools):

```json
{"scan_scope":"project","incremental":true,"severity_threshold":"medium","max_findings":30}
```

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
