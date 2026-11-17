"""Run sanitizer witnesses for v7 mechanism implementations, without RTOS boot.

These small witnesses validate the mutation mechanism and its control under
explicit input preconditions. They do not prove every native RTOS path reachable.
The released native functions are checked separately by the static analyzers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts import build_contract3_engineering_v7 as build


def source():
    names = {name: name for name in (
        "offset", "limit", "cursor", "other", "remaining", "position",
        "rendered", "segment", "written")}
    functions, records = [], []
    variants = [(category, forms, "copy") for category, forms in build.FORMS.items()]
    variants.append(("string_overflow", build.STRING_FORMS, "character_write"))
    variants += [('array_oob', ('index_assignment',), 'call_bound'),
                 ('string_overflow', ('source_size_copy',), 'call_bound'),
                 ('string_overflow', ('source_size_copy',), 'call_bound_character_write')]
    for category, forms, variant in variants:
        for mechanism in forms:
            for defect in (False, True):
                site = dict(names=names, destination="data",
                            capacity="(sizeof(data) / sizeof(data[0]))", call_derived_bound=variant.startswith('call_bound'))
                if category == "array_oob":
                    setup = "int data[8] = {0};"
                    prep, expr = build.array_mutation(site, mechanism, defect)
                    operation = f"return {expr};"
                elif category == "null_pointer":
                    setup = "struct Object object = {7}; struct Object *data = &object;"
                    site["pointer_type"] = "struct Object *"
                    prep, expr = build.null_mutation(site, mechanism, defect)
                    operation = f"return ({expr})->value;"
                elif variant.endswith("character_write"):
                    setup = "char data[8] = {0};"
                    site.update(kind="char", capacity="sizeof(data)", old="data[0]")
                    prep, expr = build.string_mutation(site, mechanism, defect)
                    operation = f"{expr} = 'x'; return data[0];"
                else:
                    setup = 'char output[24] = {0}; const char *input = "abcdefghijklmnop";'
                    site.update(kind="copy", callee="snprintf", destination="output",
                                arguments=["output", "sizeof(output)", '"%s"', "input"])
                    prep, expr = build.string_mutation(site, mechanism, defect)
                    operation = f"(void)({expr}); return (unsigned char)output[0];"
                index = len(records)
                # The caller ignores the data value; success is normal process
                # termination, not an accidental match to a chosen return value.
                functions.append(f"static int witness_{index}(void) {{\n{setup}\n" +
                                 "\n".join(prep) + "\n" + operation + "\n}")
                records.append(dict(index=index, category=category, mechanism=mechanism, variant=variant,
                                    expected="defect" if defect else "control"))
    dispatch = "\n".join(f"case {r['index']}: (void)witness_{r['index']}(); return 0;"
                         for r in records)
    text = ("#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
            "struct Object { int value; };\n" + "\n".join(functions) +
            "\nint main(int argc, char **argv) {\nif(argc != 2) return 2;\n"
            "switch(atoi(argv[1])) {\n" + dispatch + "\ndefault: return 2;\n}}\n")
    return text, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=build.DATA / "mechanism-witnesses.json")
    args = parser.parse_args()
    # Some conda Clang installations contain the static analyzer but omit the
    # sanitizer runtime. The system GCC supplies an independent runtime witness.
    compiler = shutil.which("gcc")
    if not compiler:
        raise SystemExit("gcc with sanitizer runtimes is required for the witnesses")
    generated, records = source()
    with tempfile.TemporaryDirectory(prefix="contract3-v7-witness-") as temporary:
        root = Path(temporary)
        src, binary = root / "mechanisms.c", root / "contract3-witness"
        src.write_text(generated)
        command = [compiler, "-std=gnu11", "-O0", "-g", "-fsanitize=address,undefined",
                   "-fno-sanitize-recover=all", str(src), "-o", str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if compiled.returncode:
            raise SystemExit(compiled.stderr)
        environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=0")
        for row in records:
            result = subprocess.run([str(binary), str(row["index"])], env=environment,
                                    capture_output=True, text=True, timeout=10)
            diagnostic = "runtime error:" in result.stderr or "ERROR: AddressSanitizer:" in result.stderr
            row.update(exit_code=result.returncode, sanitizer_diagnostic=diagnostic,
                       diagnostic=result.stderr[:3000],
                       valid=(result.returncode == 0 if row["expected"] == "control"
                              else result.returncode != 0 and diagnostic))
            print(row["category"], row["mechanism"], row["expected"],
                  "PASS" if row["valid"] else "FAIL", flush=True)
    build.write_json(args.output, dict(
        dataset_revision=build.REVISION, created_at=datetime.now(timezone.utc).isoformat(),
        scope="Mechanism witnesses with explicit valid inputs; not a complete RTOS path/boot proof.",
        generator_sha256=build.digest(Path(build.__file__).read_bytes()),
        witness_script_sha256=build.digest(Path(__file__).read_bytes()),
        witness_source_sha256=build.digest(generated.encode()),
        compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        passed=sum(r["valid"] for r in records), total=len(records), records=records))
    if not all(r["valid"] for r in records):
        raise SystemExit("Sanitizer witness failed; candidate must not be promoted")


if __name__ == "__main__":
    main()
