"""Prepare pinned OS source trees and labeled, test-only security fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from code_agent.security_c_rules import parse_source, walk
from code_agent.security_contracts import SCAN_CATEGORIES
from code_agent.scripts.benchmark_contract3_cppcheck import load_manifest, source_files
from code_agent.scripts.benchmark_contract3_independent import case_lines

DEFAULT_OUTPUT = ROOT / "artifacts/security-acceptance/workspaces"
HEADERS = "#include <stdlib.h>\n#include <stdio.h>\n#include <string.h>\n#include <pthread.h>\n#include <unistd.h>\n"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _frequent(category, variant, bad):
    if category == "string_overflow":
        value = "ABCDEFGH" if bad else "B"
        return ["    char target[4] = \"A\";", [
            f'    strcpy(target,"{value}");',
            f'    sprintf(target,"%s","{value}");',
            f'    strcat(target,"{value}");',
            f'    strcpy(target + 1,"{value}");',
            f'    sprintf(target,"%s%s","A","{value}");',
        ][variant], "    return target[0];"]
    # Preserve the existing independent forms, including harder pointer flows.
    return case_lines(category, variant, bad)


def _high_risk(category, variant, bad, cpp):
    if category == "buffer_overflow":
        size = 12 if bad else 4
        body = ["unsigned char dst[4]={0}; unsigned char src[16]={0};", [
            f"memcpy(dst,src,{size});", f"memmove(dst,src,{size});", f"memset(dst,0,{size});",
            f"int count={size}; memcpy(dst,src,count);", f"memcpy(dst+1,src,{size if bad else 3});",
        ][variant], "return dst[0];"]
    elif category == "memory_leak":
        body = ["char *p=(char *)malloc(32);", "if(!p) return 0;", "p[0]=1;"]
        if variant == 1:
            body = ["char *p=(char *)calloc(8,4);", "if(!p) return 0;", "p[0]=1;"]
        elif variant == 2:
            body += ["char *alias=p;", "alias[1]=2;"]
        elif variant == 3:
            body += ["if(p[0]) {", "return 1;" if bad else "free(p); return 1;", "}"]
        elif variant == 4:
            body += ["char *q=(char *)malloc(8);", "free(q);"]
        if not bad or variant == 3:
            body.append("free(p);")
        body.append("return 0;")
    elif category == "command_execution":
        body = [
            ['char *input=getenv("NATURALCC_INPUT");', "if(!input)return 0;", "system(input);" if bad else 'system("printf fixed");'],
            ["char input[64];", "if(!fgets(input,sizeof(input),stdin))return 0;", 'popen(input,"r");' if bad else 'popen("printf fixed","r");'],
            ['char *input=getenv("NATURALCC_INPUT");', "if(!input)return 0;", "char cmd[128];", 'snprintf(cmd,sizeof(cmd),"echo %s",input);' if bad else 'snprintf(cmd,sizeof(cmd),"echo %d",atoi(input));', "system(cmd);"],
            ['char *input=getenv("NATURALCC_INPUT");', "if(!input)return 0;", "char cmd[128];", "snprintf(cmd,sizeof(cmd),\"%s\",input);", "system(cmd);" if bad else 'execl("/bin/echo","echo",cmd,(char *)0);'],
            ['char *input=getenv("NATURALCC_INPUT");', "if(!input)return 0;", "char *alias=input;", "system(alias);" if bad else 'alias=(char *)"printf fixed"; system(alias);'],
        ][variant] + ["return 0;"]
    else:
        access = ["++shared;", "shared += 2;", "shared = 3;", "shared--;", "shared *= 2;"][variant]
        if cpp:
            lock = "" if bad else "std::lock_guard<std::mutex> guard(mutex);"
            return ("#include <thread>\n#include <mutex>\nstatic int shared;\nstatic std::mutex mutex;\n"
                f"static void worker() {{ {lock}\n {access}\n}}\n"
                "static int exercise(){std::thread a(worker);std::thread b(worker);a.join();b.join();return 0;}\n"
                "#ifdef RUN_CASE\nint main(){return exercise();}\n#endif\n")
        lock, unlock = ("", "") if bad else ("pthread_mutex_lock(&mutex);", "pthread_mutex_unlock(&mutex);")
        return ("static int shared;\nstatic pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;\n"
            f"static void *worker(void *arg){{ {lock}\n {access}\n {unlock}\n return arg;\n}}\n"
            "static int exercise(){pthread_t a,b;\n"
            "if(pthread_create(&a,0,worker,0))return 1;\n"
            "if(pthread_create(&b,0,worker,0)){pthread_join(a,0);return 1;}\n"
            "pthread_join(a,0);pthread_join(b,0);return 0;}\n"
            "#ifdef RUN_CASE\nint main(){return exercise();}\n#endif\n")
    return "static int exercise(void){\n" + "\n".join(body) + "\n}\n"


def generate_cases(project, scan_type):
    directory = project / "naturalcc_cases"
    directory.mkdir(parents=True, exist_ok=True)
    cases, hashes = [], {}
    count = 20 if scan_type == "frequent_defects" else 10
    ordinal = 0
    for category in SCAN_CATEGORIES[scan_type]:
        for bad in (True, False):
            for index in range(count):
                ordinal += 1
                cpp = scan_type == "high_risk" and index >= 5
                file = f"naturalcc_cases/case_{ordinal:04d}{'.cpp' if cpp else '.c'}"
                if scan_type == "frequent_defects":
                    helpers = ("struct contract3_box { int *pointer; };\n"
                        "static int contract3_identity(int x){return x;}\n"
                        "static int contract3_read(int *p){return *p;}\n")
                    source = HEADERS + helpers + "static int exercise(void){\n" + "\n".join(_frequent(category, index % 5, bad)) + "\n}\n"
                else:
                    source = HEADERS + _high_risk(category, index % 5, bad, cpp)
                path = project / file
                if path.exists() and path.read_text(encoding="utf-8") != source:
                    raise ValueError(f"Refusing to overwrite changed fixture: {path}")
                path.write_text(source, encoding="utf-8")
                hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
                cases.append({"case_id": f"{project.name}-{ordinal:04d}", "category": category,
                    "file": file, "function": "worker/exercise" if category == "data_race" else "exercise",
                    "line_start": len(HEADERS.splitlines()) + 1, "line_end": len(source.splitlines()),
                    "expected": "defect" if bad else "control", "variant": index % 5})
    truth = {"schema_version": 1, "scan_type": scan_type, "source_sha256": hashes, "cases": cases}
    write_json(project / "ground_truth.json", truth)
    return truth


def statement_count(project, roots):
    count, errors = 0, 0
    files = [path for path in source_files(project, roots) if "naturalcc_cases" not in path.parts]
    for path in files:
        tree = parse_source(path.read_text(encoding="utf-8", errors="replace"), path.suffix.lower())
        errors += int(tree.has_error)
        count += sum(n.type.endswith("_statement") and n.type not in {"compound_statement", "labeled_statement"}
                     for n in walk(tree))
    return {"statement_nodes": count, "source_files": len(files), "parse_error_files": errors,
        "counting_rule": "Tree-sitter statement nodes excluding compound/labeled wrappers; declarations/comments/blank lines excluded; all preprocessor branches included; not compiled coverage."}


def prepare(output, fixtures_only=False):
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    summaries = []
    for spec in manifest["sources"]:
        project = output / spec["id"]
        roots = {"freertos_kernel": ["."], "zephyr": ["kernel", "lib", "subsys"], "rt_thread": ["src", "components", "libcpu"]}[spec["id"]]
        stats = None
        if not fixtures_only:
            if not project.exists():
                subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", spec["ref"], spec["url"], str(project)], check=True)
            actual = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
            if actual != spec["commit"]:
                raise ValueError(f"Unexpected source revision: {project}")
            if roots == ["."]:
                subprocess.run(["git", "-C", str(project), "sparse-checkout", "disable"], check=True)
            else:
                subprocess.run(["git", "-C", str(project), "sparse-checkout", "set", *roots], check=True)
            stats = statement_count(project, roots)
            if stats["statement_nodes"] < 10_000:
                raise ValueError(f"{spec['id']} has fewer than 10,000 statement nodes: {stats}")
        truth = generate_cases(project, "frequent_defects")
        summaries.append({**spec, "scan_roots": roots, "size": stats, "cases": len(truth["cases"]), "scan_type": "frequent_defects"})
    high = output / "high_risk"
    if not fixtures_only and not high.exists():
        shutil.copytree(output / "freertos_kernel", high, ignore=shutil.ignore_patterns(".git", "naturalcc_cases", "ground_truth.json"))
    truth = generate_cases(high, "high_risk")
    summaries.append({"id": "high_risk", "upstream": manifest["sources"][0], "cases": len(truth["cases"]), "scan_type": "high_risk"})
    write_json(output / "suite.json", {"schema_version": 1, "fixtures_only": fixtures_only, "projects": summaries})
    print(output / "suite.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixtures-only", action="store_true", help="Offline API smoke cases; does not claim OS project size")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.fixtures_only)
