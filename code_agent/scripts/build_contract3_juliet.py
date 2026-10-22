"""Freeze a Contract-3 sample set from NIST Juliet C/C++ 1.3 (third-party labels).

Unlike the generated RTOS fixtures, labels come from the upstream Juliet
OMITBAD/OMITGOOD split.  Only flow variants that depend on macros, globals or
cross-function/cross-file data flow are eligible; the selection order is a
fixed hash of the test-case name, computed before any model run.

Each selected test case yields one vulnerable sample (bad path only) and one
safe sample (good paths only).  Comments are stripped and label-bearing
identifiers (CWE*, *bad*, *good*, *G2B*, *B2G*) are renamed to neutral names so
the Agent cannot read the answer from the source.  Every sample is checked with
``gcc -fsyntax-only`` together with the Juliet support files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "contract3" / "juliet"
JULIET_ZIP = "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"
JULIET_ZIP_SHA256 = "ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb"
SELECTION_SALT = "contract3-juliet-v1"
PER_CATEGORY = 10
# 09-14: global const/var/function-controlled flow; 21-22: static/global flags
# across functions or files; 41-45: data passed/returned through functions,
# function pointers or static globals; 51-54, 61-68: cross-file data flow.
FLOWS = {"09", "10", "11", "12", "13", "14", "21", "22", "41", "42", "44", "45",
         "51", "52", "53", "54", "61", "63", "64", "65", "66", "67", "68"}
FAMILIES = {
    "array_oob": ("CWE121_Stack_Based_Buffer_Overflow", [
        "CWE129_fgets", "CWE129_fscanf", "CWE129_large", "CWE129_rand",
        "CWE805_int_declare_loop", "CWE805_int_alloca_loop"]),
    "string_overflow": ("CWE121_Stack_Based_Buffer_Overflow", [
        "dest_char_declare_cpy", "dest_char_declare_cat", "dest_char_alloca_cpy", "dest_char_alloca_cat",
        "src_char_declare_cpy", "src_char_declare_cat", "CWE193_char_declare_cpy", "CWE193_char_declare_ncpy",
        "CWE805_char_declare_ncpy", "CWE805_char_declare_ncat", "CWE805_char_declare_snprintf",
        "CWE806_char_declare_ncpy", "CWE806_char_declare_ncat", "CWE806_char_declare_snprintf"]),
    "null_pointer": ("CWE476_NULL_Pointer_Dereference", [
        "char", "int", "int64_t", "long", "struct", "binary_if", "deref_after_check", "null_check_after_deref"]),
}
# Subtler defect families for difficulty calibration (probe, then a disjoint
# hard set).  Over/under-reads count as array out-of-bounds; a missing NUL
# terminator counts as a string overflow; unchecked allocation results as NULL.
HARD_FAMILIES = {
    "array_oob": [("CWE124_Buffer_Underwrite", "CWE839_fgets"), ("CWE124_Buffer_Underwrite", "CWE839_rand"),
                  ("CWE127_Buffer_Underread", "CWE839_fscanf"), ("CWE126_Buffer_Overread", "CWE129_large"),
                  ("CWE122_Heap_Based_Buffer_Overflow", "sizeof_int64_t"), ("CWE121_Stack_Based_Buffer_Overflow", "CWE131_loop")],
    "string_overflow": [("CWE121_Stack_Based_Buffer_Overflow", "CWE193_char_declare_loop"),
                        ("CWE126_Buffer_Overread", "CWE170_char_strncpy"), ("CWE126_Buffer_Overread", "char_declare_memcpy"),
                        ("CWE122_Heap_Based_Buffer_Overflow", "c_CWE193_char_memcpy"), ("CWE121_Stack_Based_Buffer_Overflow", "CWE135")],
    "null_pointer": [("CWE690_NULL_Deref_From_Return", "int_malloc"), ("CWE690_NULL_Deref_From_Return", "struct_realloc"),
                     ("CWE690_NULL_Deref_From_Return", "fopen")],
}
PROBE_PER_FAMILY = 2
SUPPORT = ("std_testcase.h", "std_testcase_io.h", "io.c")
LABEL_WORD = re.compile(r"CWE\d|bad|good|Bad|Good|BAD|GOOD|G2B|B2G")
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
CONDITION = re.compile(r"^\s*#\s*(ifdef|ifndef)\s+(OMITBAD|OMITGOOD|INCLUDEMAIN)\b")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            out.append(" ")
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    lines = [line.rstrip() for line in "".join(out).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def resolve(text: str, defined: set[str]) -> str:
    """Resolve only OMITBAD/OMITGOOD/INCLUDEMAIN blocks; keep other directives."""
    out, stack = [], []  # frame: (handled, parent_active, branch_active)
    for line in text.splitlines():
        active = all(frame[2] for frame in stack)
        match = CONDITION.match(line)
        directive = re.match(r"^\s*#\s*(if|ifdef|ifndef|else|elif|endif)\b", line)
        if match:
            taken = (match.group(2) in defined) == (match.group(1) == "ifdef")
            stack.append((True, active, taken))
            continue
        if directive and directive.group(1) in {"if", "ifdef", "ifndef"}:
            stack.append((False, active, True))
        elif directive and directive.group(1) in {"else", "elif"} and stack[-1][0]:
            handled, parent, taken = stack.pop()
            stack.append((handled, parent, not taken))
            continue
        elif directive and directive.group(1) == "endif":
            handled, parent, _ = stack.pop()
            if handled:
                continue
            if parent:
                out.append(line)
            continue
        if all(frame[2] for frame in stack) or (directive and active):
            out.append(line)
    if stack:
        raise ValueError("Unbalanced preprocessor block")
    return "\n".join(out) + "\n"


def support_files(juliet: Path) -> dict[str, str]:
    files = {}
    for name in SUPPORT:
        text = strip_comments((juliet / "testcasesupport" / name).read_text(encoding="latin-1"))
        text = "\n".join(line for line in text.splitlines() if not re.match(r"\s*void (good|bad)\d\(\)", line)) + "\n"
        if LABEL_WORD.search(text):
            raise ValueError(f"Label word left in support file {name}")
        files[name] = text
    return files


def candidates(juliet: Path, families_by_category: dict[str, list[tuple[str, str]]] | None = None) -> dict[str, list[dict]]:
    if families_by_category is None:
        families_by_category = {category: [(directory, family) for family in families]
                                for category, (directory, families) in FAMILIES.items()}
    found: dict[str, list[dict]] = {}
    for category, pairs in families_by_category.items():
        groups: dict[str, list[Path]] = defaultdict(list)
        for directory in sorted({directory for directory, _ in pairs}):
            families = {family for d, family in pairs if d == directory}
            for path in sorted((juliet / "testcases" / directory).rglob("*.c")):
                match = re.match(rf"{directory}__(.+)_(\d\d)([a-e]?)\.c$", path.name)
                if match and match.group(1) in families and match.group(2) in FLOWS:
                    groups[f"{directory}__{match.group(1)}_{match.group(2)}"].append(path)
        rows = [{"testcase": name, "family": name.split("__", 1)[1].rsplit("_", 1)[0], "flow": name[-2:],
                 "files": sorted(paths), "key": digest(f"{SELECTION_SALT}:{name}".encode())}
                for name, paths in groups.items()]
        found[category] = sorted(rows, key=lambda row: row["key"])
    return found


def build_sample(case: dict, vulnerable: bool) -> dict[str, str]:
    defined = {"OMITGOOD"} if vulnerable else {"OMITBAD"}
    parts = {}
    for index, path in enumerate(case["files"]):
        text = resolve(path.read_text(encoding="latin-1"), defined)
        parts[f"src_{'abcde'[index]}.c"] = strip_comments(text)
    names: dict[str, str] = {}
    for text in parts.values():
        for ident in IDENT.findall(text):
            if LABEL_WORD.search(ident) and ident not in names:
                names[ident] = f"sym_{len(names) + 1:02d}"
    renamed = {name: IDENT.sub(lambda m: names.get(m.group(0), m.group(0)), text) for name, text in parts.items()}
    for name, text in renamed.items():
        if LABEL_WORD.search(re.sub(r'"(?:\\.|[^"\\])*"', '""', text)):
            raise ValueError(f"Label word left in {case['testcase']} {name}")
    return renamed


def syntax_check(files: dict[str, str], support: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, text in {**support, **files}.items():
            (root / name).write_text(text, encoding="utf-8")
        sources = [str(root / name) for name in files] + [str(root / "io.c")]
        result = subprocess.run(["gcc", "-fsyntax-only", "-w", "-I", str(root), *sources], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr[:800])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--juliet", type=Path, default=os.environ.get("JULIET_C_ROOT"), required="JULIET_C_ROOT" not in os.environ,
                        help=f"Extracted Juliet C directory (contains testcases/ and testcasesupport/); download {JULIET_ZIP}")
    parser.add_argument("--holdout", action="store_true",
                        help="Freeze the next PER_CATEGORY cases of the same hash order into a separate, untuned set")
    parser.add_argument("--probe", action="store_true",
                        help="Vulnerable-only difficulty probe: first PROBE_PER_FAMILY cases of each hard family")
    args = parser.parse_args()
    if args.probe:
        return build_probe(args.juliet)
    out = OUT.with_name("juliet-holdout") if args.holdout else OUT
    offset = PER_CATEGORY if args.holdout else 0
    if shutil.which("gcc") is None:
        raise SystemExit("gcc is required for the syntax check")
    support = support_files(args.juliet)
    samples, selection = [], {}
    for category, rows in candidates(args.juliet).items():
        chosen = rows[offset:offset + PER_CATEGORY]
        selection[category] = {"eligible": len(rows), "chosen": [row["testcase"] for row in chosen]}
        for number, case in enumerate(chosen, 1):
            for vulnerable in (True, False):
                files = build_sample(case, vulnerable)
                syntax_check(files, support)
                samples.append({"id": f"{category}-{number:02d}-{'v' if vulnerable else 's'}", "category": category,
                                "vulnerable": vulnerable, "testcase": case["testcase"], "family": case["family"],
                                "flow": case["flow"], "origin": [str(path.relative_to(args.juliet)) for path in case["files"]],
                                "origin_sha256": {path.name: digest(path.read_bytes()) for path in case["files"]},
                                "files": files})
    dataset = {"source": "NIST SARD Juliet Test Suite for C/C++ v1.3", "url": JULIET_ZIP, "zip_sha256": JULIET_ZIP_SHA256,
               "support": support, "samples": samples}
    raw = (json.dumps(dataset, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.json").write_bytes(raw)
    manifest = {"dataset_sha256": digest(raw), "selection_salt": SELECTION_SALT, "per_category": PER_CATEGORY, "offset": offset,
                "flows": sorted(FLOWS), "families": {key: value[1] for key, value in FAMILIES.items()},
                "selection": selection,
                "rule": "Eligible C test cases sorted by sha256(salt:testcase); cases [offset, offset+N) per category; "
                        "no outcome-based filtering. Labels from Juliet OMITBAD/OMITGOOD split."}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(samples)} samples -> {out}")
    for category, value in selection.items():
        print(category, value["eligible"], "eligible;", ", ".join(name.split("__", 1)[1] for name in value["chosen"]))



def write_set(out: Path, samples: list[dict], support: dict[str, str], manifest: dict) -> None:
    dataset = {"source": "NIST SARD Juliet Test Suite for C/C++ v1.3", "url": JULIET_ZIP, "zip_sha256": JULIET_ZIP_SHA256,
               "support": support, "samples": samples}
    raw = (json.dumps(dataset, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.json").write_bytes(raw)
    manifest = {"dataset_sha256": digest(raw), "selection_salt": SELECTION_SALT, **manifest, "order": [s["id"] for s in samples]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(samples)} samples -> {out}")


def sample_record(identifier: str, category: str, case: dict, vulnerable: bool, juliet: Path, support: dict[str, str]) -> dict:
    files = build_sample(case, vulnerable)
    syntax_check(files, support)
    return {"id": identifier, "category": category, "vulnerable": vulnerable, "testcase": case["testcase"],
            "family": case["family"], "flow": case["flow"], "origin": [str(path.relative_to(juliet)) for path in case["files"]],
            "origin_sha256": {path.name: digest(path.read_bytes()) for path in case["files"]}, "files": files}


def build_probe(juliet: Path) -> None:
    support = support_files(juliet)
    samples, selection = [], {}
    for category, pairs in HARD_FAMILIES.items():
        for directory, family in pairs:
            rows = candidates(juliet, {category: [(directory, family)]})[category]
            chosen = rows[:PROBE_PER_FAMILY]
            selection[f"{directory}__{family}"] = {"eligible": len(rows), "chosen": [row["testcase"] for row in chosen]}
            for case in chosen:
                samples.append(sample_record(f"{category}-{family}-{case['flow']}-v", category, case, True, juliet, support))
    write_set(OUT.with_name("juliet-probe"), samples, support, {
        "per_family": PROBE_PER_FAMILY, "flows": sorted(FLOWS), "families": HARD_FAMILIES, "selection": selection,
        "rule": "Vulnerable-only difficulty probe: first N cases per hard family in sha256(salt:testcase) order. "
                "Used only to estimate per-family miss rates; the calibrated hard set must use later cases."})


if __name__ == "__main__":
    main()
