#!/usr/bin/env python3
"""Prepare small, auditable challenge pools from official benchmark records.

Only task inputs and fixed-revision metadata are exported. Reference patches and
hidden tests are intentionally excluded from the UI-facing sample folders.
"""

from __future__ import annotations

import json
import csv
import io
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import ssl
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
UA = "NaturalCC-benchmark-sample-preparer/1.0"
LANGUAGES = ("C", "C++", "Go", "Java", "JavaScript", "PHP", "Ruby", "Rust", "TypeScript")
ATLAS_LANGUAGES = LANGUAGES + ("Python",)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(1 + attempt * 2)
    assert last_error is not None
    raise last_error


def get_json(url: str) -> Any:
    return json.loads(fetch(url).decode("utf-8"))


def get_text(url: str) -> str:
    return fetch(url).decode("utf-8", errors="replace")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fresh(path: Path) -> None:
    if path.exists():
        dataset_root = path if path.name in {"SecRepoBench", "SWE-bench-Multilingual", "Defects4C", "SWE-Atlas-Refactoring"} else path.parent
        # Permit resuming only folders created by this script before their
        # selection.json completion marker was written.
        if (dataset_root / "README.md").exists() and not (dataset_root / "selection.json").exists():
            return
        raise FileExistsError(f"Refusing to overwrite existing or completed sample folder: {path}")
    path.mkdir(parents=True)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def complexity(code: str) -> tuple[int, int, int]:
    loc = sum(bool(line.strip()) for line in code.splitlines())
    branches = len(re.findall(r"\b(if|for|while|switch|case|goto|catch)\b|\?.*:", code))
    pointer_ops = len(re.findall(r"->|\*\s*[A-Za-z_(]|\[[^]]+\]", code))
    return loc, branches, pointer_ops


def sec_repo_samples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = "ai-sec-lab/SecRepoBench"
    for offset in range(0, 400, 100):
        url = (
            "https://datasets-server.huggingface.co/rows?"
            + urllib.parse.urlencode(
                {"dataset": dataset, "config": "default", "split": "code_completion", "offset": offset, "length": 100}
            )
        )
        page = get_json(url)
        rows.extend(item["row"] for item in page.get("rows", []))
        if len(rows) >= 318:
            break
    if len(rows) < 10:
        raise RuntimeError(f"SecRepoBench returned only {len(rows)} rows")

    candidates = []
    for row in rows:
        code = row["masked_func"]
        loc, branches, pointer_ops = complexity(code)
        cwe = int(row["CWE_ID"]) if row.get("CWE_ID") is not None else -1
        # Prefer long/control-flow-rich functions, while enforcing CWE and project variety.
        score = loc + 2.5 * branches + 0.3 * pointer_ops
        candidates.append((score, row))
    selected: list[tuple[float, dict[str, Any]]] = []
    used_cwe: set[int] = set()
    used_projects: Counter[str] = Counter()
    for score, row in sorted(candidates, key=lambda item: item[0], reverse=True):
        cwe = int(row["CWE_ID"]) if row.get("CWE_ID") is not None else -1
        project = row["project_name"]
        if cwe in used_cwe or used_projects[project] >= 2:
            continue
        selected.append((score, row))
        used_cwe.add(cwe)
        used_projects[project] += 1
        if len(selected) == 10:
            break
    if len(selected) < 10:
        for score, row in sorted(candidates, key=lambda item: item[0], reverse=True):
            if row["id"] not in {item[1]["id"] for item in selected}:
                selected.append((score, row))
            if len(selected) == 10:
                break

    out = ROOT / "CodeCompletion" / "SecRepoBench"
    fresh(out)
    write_text(
        out / "README.md",
        """# SecRepoBench 样本池（10 题）

来源：[官方实现与数据集](https://github.com/ai-sec-lab/SecRepoBench)、[官方数据集卡](https://huggingface.co/datasets/ai-sec-lab/SecRepoBench)。

按目标函数的非空行数、控制流和指针/索引操作复杂度排序，优先抽取不同 CWE 与不同项目；此为本地筛选启发式，不是官方难度标签。每题提供官方 masked function、描述提示和固定修复 commit 元数据。官方测评仍需在完整 C/C++ 仓库环境中编译，并运行开发者单测和 OSS-Fuzz PoC；此处的单函数 workspace 适合 UI 演示，不等同官方计分。

`TASK.md` 和 `workspace/` 是给 NaturalCC 的输入。官方修复补丁与隐藏测试未复制进样本目录。
""",
    )
    manifest = []
    for rank, (score, row) in enumerate(selected, 1):
        task_id = str(row["id"])
        case = out / f"SRB-{rank:02d}-{task_id}"
        fresh(case)
        desc_url = f"https://raw.githubusercontent.com/ai-sec-lab/SecRepoBench/main/descriptions/{task_id}/desc_prompt.txt"
        prompt = get_text(desc_url)
        code = row["masked_func"]
        loc, branches, pointer_ops = complexity(code)
        ext = ".cpp" if row["changed_file"].endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh")) else ".c"
        write_text(case / "workspace" / ("target_function" + ext), code)
        meta = {
            "benchmark": "SecRepoBench",
            "instance_id": task_id,
            "project": row["project_name"],
            "repository": row["repo_addr"],
            "fixed_revision": row["fixing_commit"],
            "target_file": row["changed_file"],
            "cwe_id": row["CWE_ID"],
            "selection": {"heuristic_score": round(score, 2), "nonempty_lines": loc, "control_flow_tokens": branches, "pointer_or_index_operations": pointer_ops},
            "source": desc_url,
            "dataset_record": f"https://huggingface.co/datasets/ai-sec-lab/SecRepoBench",
        }
        write_json(case / "case.json", meta)
        write_text(
            case / "TASK.md",
            f"""# SecRepoBench completion task {task_id}

Complete the masked region in `workspace/target_function{ext}` using the surrounding function context. Preserve the function interface and repository conventions. The target is security-sensitive (CWE-{row['CWE_ID']}); produce code that is both functionally correct and resistant to the relevant malformed or boundary inputs.

Official task description:

{prompt}

Repository: `{row['repo_addr']}`  
Target file: `{row['changed_file']}`  
Reference revision (metadata only): `{row['fixing_commit']}`
""",
        )
        manifest.append(meta)
    write_json(out / "selection.json", manifest)
    return manifest


def language_for(row: dict[str, Any]) -> str:
    repo = row["repo"]
    known = {
        "redis/redis": "C", "valkey-io/valkey": "C", "jqlang/jq": "C", "micropython/micropython": "C",
        "fmtlib/fmt": "C++", "nlohmann/json": "C++",
        "caddyserver/caddy": "Go", "gin-gonic/gin": "Go", "gohugoio/hugo": "Go", "hashicorp/terraform": "Go", "prometheus/prometheus": "Go",
        "apache/druid": "Java", "apache/lucene": "Java", "google/gson": "Java", "javaparser/javaparser": "Java", "projectlombok/lombok": "Java", "reactivex/rxjava": "Java",
        "axios/axios": "TypeScript", "babel/babel": "JavaScript", "facebook/docusaurus": "TypeScript", "immutable-js/immutable-js": "JavaScript", "mrdoob/three.js": "JavaScript", "preactjs/preact": "JavaScript", "vuejs/core": "TypeScript",
        "laravel/framework": "PHP", "php-cs-fixer/php-cs-fixer": "PHP", "phpoffice/phpspreadsheet": "PHP", "briannesbitt/carbon": "PHP",
        "faker-ruby/faker": "Ruby", "fastlane/fastlane": "Ruby", "fluent/fluentd": "Ruby", "jekyll/jekyll": "Ruby", "jordansissel/fpm": "Ruby", "rubocop/rubocop": "Ruby",
        "astral-sh/ruff": "Rust", "burntsushi/ripgrep": "Rust", "nushell/nushell": "Rust", "sharkdp/bat": "Rust", "tokio-rs/axum": "Rust", "tokio-rs/tokio": "Rust", "uutils/coreutils": "Rust",
    }
    if repo in known:
        return known[repo]
    extensions = Counter(re.findall(r"(?m)^diff --git a/[^\n]+?\.([A-Za-z0-9]+) b/", row.get("patch", "")))
    by_ext = {"c": "C", "cpp": "C++", "cc": "C++", "cxx": "C++", "go": "Go", "java": "Java", "js": "JavaScript", "jsx": "JavaScript", "php": "PHP", "rb": "Ruby", "rs": "Rust", "ts": "TypeScript", "tsx": "TypeScript"}
    votes = Counter(by_ext[ext] for ext, count in extensions.items() if ext in by_ext for _ in range(count))
    return votes.most_common(1)[0][0] if votes else "unknown"


def multilingual_samples() -> list[dict[str, Any]]:
    dataset = "SWE-bench/SWE-bench_Multilingual"
    rows = []
    for offset in (0, 100, 200):
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": dataset, "config": "default", "split": "test", "offset": offset, "length": 100}
        )
        rows.extend(item["row"] for item in get_json(url).get("rows", []))
    if len(rows) < 10:
        raise RuntimeError(f"SWE-bench Multilingual returned only {len(rows)} rows")
    candidates = []
    for row in rows:
        paths = re.findall(r"(?m)^diff --git a/(.+?) b/", row.get("patch", ""))
        src = [p for p in paths if not re.search(r"(^|/)(test|tests|__tests__|spec|specs)(/|\.)|Test\.java$|_test\.go$|\.(test|spec)\.", p, re.I)]
        changed = len(re.findall(r"(?m)^\+(?!\+\+)|^-(?!--)", row.get("patch", "")))
        lang = language_for(row)
        f2p, p2p = len(row.get("FAIL_TO_PASS", [])), len(row.get("PASS_TO_PASS", []))
        score = len(src) * 6 + min(changed, 500) / 40 + f2p * 2
        candidates.append((score, lang, row, len(src), changed, f2p, p2p))
    selected = []
    for language in LANGUAGES:
        group = [item for item in candidates if item[1] == language]
        if group:
            selected.append(max(group, key=lambda item: item[0]))
    if len(selected) < 9:
        raise RuntimeError(f"Could not find a sample for each language: selected {len(selected)}")
    selected_ids = {item[2]["instance_id"] for item in selected}
    extras = sorted((item for item in candidates if item[2]["instance_id"] not in selected_ids), key=lambda item: item[0], reverse=True)
    selected.extend(extras[: 10 - len(selected)])

    out = ROOT / "BugFix" / "SWE-bench-Multilingual"
    fresh(out)
    write_text(
        out / "README.md",
        """# SWE-bench Multilingual 样本池（10 题）

来源：[官方数据集卡](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual)、[官方评测仓库](https://github.com/SWE-bench/SWE-bench)。

从 300 条官方 test split 中，先保证 C、C++、Go、Java、JavaScript、PHP、Ruby、Rust、TypeScript 各有一题，再按 gold patch 涉及的非测试源文件数、改动行数和 FAIL_TO_PASS 数量选每种语言的高复杂候选；第 10 题取剩余候选中分数最高者。该分数是本地筛选启发式，不是官方难度标签。

每个样本含 issue、仓库与 base commit、测试标识和官方容器镜像元数据。完整源仓库与 SWE-bench harness 需按官方环境准备；`patch` 和 `test_patch` 不导出到 agent 可见目录，避免泄露答案及新增测试。
""",
    )
    manifest = []
    for rank, (score, lang, row, src_count, changed, f2p, p2p) in enumerate(selected, 1):
        case = out / f"SWE-MULTI-{rank:02d}-{slug(row['instance_id'])}"
        fresh(case)
        meta = {key: row.get(key) for key in ("instance_id", "repo", "base_commit", "version", "image", "problem_statement", "hints_text", "created_at", "eval_type", "log_parser")}
        meta.update({"language": lang, "selection": {"heuristic_score": round(score, 2), "non_test_source_files_in_reference_patch": src_count, "reference_patch_changed_lines": changed, "fail_to_pass_test_count": f2p, "pass_to_pass_test_count": p2p}, "source": "https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual"})
        write_json(case / "case.json", meta)
        write_text(
            case / "TASK.md",
            f"""# SWE-bench Multilingual task: {row['instance_id']}

Resolve the GitHub issue below in the repository workspace. Follow existing conventions and add or update tests as appropriate. Do not use the reference patch; the official evaluator validates the patch in the repository's pinned environment.

## Issue

{row['problem_statement']}

## Repository and revision

- Repository: `{row['repo']}`
- Language: {lang}
- Base commit: `{row['base_commit']}`
- SWE-bench instance: `{row['instance_id']}`

The selected record is metadata and task text. It does not contain a checked-out repository; use the official harness or fetch the repository at the exact base commit before asking the agent to edit files.
""",
        )
        manifest.append(meta)
    write_json(out / "selection.json", manifest)
    return manifest


def defects4c_samples() -> list[dict[str, Any]]:
    root = "https://raw.githubusercontent.com/defects4c/defects4c/master/defectsc_tpl/data/"
    csv_rows = csv.DictReader(io.StringIO(get_text(root + "raw_info_step1.csv")))
    commit_to_repo: dict[str, str] = {}
    for csv_row in csv_rows:
        match = re.search(r"/repos/([^/]+/[^/]+)/commits/([0-9a-f]{40})", csv_row.get("api_url", ""))
        if match:
            commit_to_repo[match.group(2)] = match.group(1)
    sources = [("buggy_errmsg/single_function_repair.json", "bug-fix") , ("buggy_errmsg_cve/single_function_repair.json", "vulnerability-fix")]
    groups = []
    for path, label in sources:
        data = get_json(root + path)
        candidates = []
        for key, record in data.items():
            code = record.get("buggy", "")
            loc, branches, pointer_ops = complexity(code)
            commit = key.split("___", 1)[0]
            project = commit_to_repo.get(commit, commit)
            score = loc + 2.5 * branches + 0.3 * pointer_ops
            candidates.append((score, project, key, record, path, label, loc, branches, pointer_ops))
        groups.append(candidates)
    # Balance ordinary defect repair and vulnerability repair. Limit repeated
    # projects and exact source paths, while retaining the largest functions.
    selected = []
    project_counts: Counter[str] = Counter()
    selected_sources: set[tuple[str, str]] = set()
    for candidates, limit in zip(groups, (6, 4)):
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        for item in ranked:
            source_identity = (item[1], item[2].split("___", 1)[-1])
            if project_counts[item[1]] >= 2 or source_identity in selected_sources:
                continue
            selected.append(item)
            project_counts[item[1]] += 1
            selected_sources.add(source_identity)
            if sum(1 for chosen in selected if chosen[5] == item[5]) == limit:
                break
    if len(selected) < 10:
        raise RuntimeError(f"Defects4C data returned only {len(selected)} distinct-project candidates")

    out = ROOT / "BugFix" / "Defects4C"
    fresh(out)
    write_text(
        out / "README.md",
        """# Defects4C 样本池（10 题）

来源：[官方数据与评测代码](https://github.com/defects4c/defects4c)、[官方说明](https://defects4c.github.io/)。

从官方普通缺陷修复和 CVE/漏洞修复函数集合中选 6+4 题，优先选择控制流、指针/索引操作较多的函数，并限制重复项目及重复源文件。复杂度分数为本地筛选启发式，不是官方难度评级。每题包含官方数据中的 buggy function 和项目/文件位置标识。

这里的单函数源码适合 UI 演示；它不包含完整项目、测试夹具或可独立编译的上下文。官方本地 Docker 环境约需 80GB，不适合当前 D 盘余量；要正式计分，优先使用官方验证平台/API，或自行准备只含这些实例的环境。
""",
    )
    manifest = []
    for rank, item in enumerate(selected, 1):
        score, project, key, record, source_path, label, loc, branches, pointer_ops = item
        case_id = slug(key)
        case = out / f"D4C-{rank:02d}-{case_id[:48]}"
        fresh(case)
        ext = Path(key.split("___", 1)[-1]).suffix or ".c"
        write_text(case / "workspace" / ("buggy_function" + ext), record.get("buggy", ""))
        meta = {"benchmark": "Defects4C", "instance_key": key, "task_type": label, "project_key": project, "source_file": key.split("___", 1)[-1], "buggy_range": {"start": record.get("start"), "end": record.get("end")}, "selection": {"heuristic_score": round(score, 2), "nonempty_lines": loc, "control_flow_tokens": branches, "pointer_or_index_operations": pointer_ops}, "source": "https://github.com/defects4c/defects4c/blob/master/" + "defectsc_tpl/data/" + source_path}
        write_json(case / "case.json", meta)
        write_text(
            case / "TASK.md",
            f"""# Defects4C {label}: {key}

Inspect and repair the C/C++ function in `workspace/buggy_function{ext}`. Preserve its interface and surrounding project behavior. Explain the root cause, implement a robust fix, and consider boundary cases and memory safety.

- Dataset record: `{key}`
- Project key: `{project}`
- Source file: `{meta['source_file']}`
- Official function line range: `{record.get('start')}`–`{record.get('end')}`
- Task type: {label}

This folder contains the official buggy function input only. For benchmark scoring, use the Defects4C evaluation service or the full official project/test environment; a local UI edit to this function alone is a demonstration, not a scored result.
""",
        )
        manifest.append(meta)
    write_json(out / "selection.json", manifest)
    return manifest


def atlas_samples() -> list[dict[str, Any]]:
    base = "https://raw.githubusercontent.com/scaleapi/swe-atlas/main/data/rf/"
    dataset_text = get_text(base + "dataset.toml")
    task_names = re.findall(r'name\s*=\s*"scale-ai/(task-[a-f0-9]+)"', dataset_text)
    if len(task_names) < 10:
        raise RuntimeError(f"SWE Atlas manifest returned only {len(task_names)} tasks")
    scored = []
    for task in task_names:
        prefix = base + task + "/"
        toml = get_text(prefix + "task.toml")
        instruction = get_text(prefix + "instruction.md")
        patch = get_text(prefix + "solution/gold.patch")
        files = re.findall(r"(?m)^diff --git a/(.+?) b/", patch)
        changed = len(re.findall(r"(?m)^\+(?!\+\+)|^-(?!--)", patch))
        tags_match = re.search(r"tags\s*=\s*\[(.*?)\]", toml, re.S)
        tags = re.findall(r'"([^"]+)"', tags_match.group(1)) if tags_match else []
        language = next((tag for tag in tags if tag in ATLAS_LANGUAGES), "unknown")
        score = len(files) * 10 + min(changed, 2000) / 50 + len(instruction.split()) / 100
        scored.append({"id": task, "toml": toml, "instruction": instruction, "files": files, "changed": changed, "tags": tags, "language": language, "score": score})
    selected = []
    # First cover languages among the hardest multi-file tasks, then fill by score.
    for language in ATLAS_LANGUAGES:
        options = [item for item in scored if item["language"] == language]
        if options:
            selected.append(max(options, key=lambda item: item["score"]))
    used = {item["id"] for item in selected}
    selected.extend(item for item in sorted(scored, key=lambda item: item["score"], reverse=True) if item["id"] not in used)
    selected = selected[:10]

    out = ROOT / "Refactoring" / "SWE-Atlas-Refactoring"
    fresh(out)
    write_text(
        out / "README.md",
        """# SWE-Atlas Refactoring 样本池（10 题）

来源：[官方 benchmark 仓库](https://github.com/scaleapi/swe-atlas)。官方 Refactoring 集合有 70 个任务，任务元数据标注为 hard。这里从官方任务中优先覆盖不同语言标签，再按官方 gold patch 所涉及文件数、改动行数和 instruction 长度排序选取。分数仅用于本次抽样，不是官方 ranking。

每个样本保留官方 instruction、task id、语言标签、相关文件/测试文件名和运行镜像标识；不复制 gold patch、rubrics 或测试脚本。正式计分应通过 Harbor 和 SWE-Atlas 官方 sandbox/validator 执行。当前文件夹是 agent/UI 的任务卡，不是完整项目 checkout。
""",
    )
    manifest = []
    for rank, item in enumerate(selected, 1):
        task = item["id"]
        case = out / f"ATLAS-RF-{rank:02d}-{task}"
        fresh(case)
        prefix = base + task + "/"
        relevant_files = get_json(prefix + "tests/relevant_file_names.json")
        relevant_tests = get_json(prefix + "tests/relevant_test_names.json")
        toml = item["toml"]
        image_match = re.search(r'docker_image\s*=\s*"([^"]+)"', toml)
        image = image_match.group(1) if image_match else ""
        meta = {"benchmark": "SWE-Atlas Refactoring", "task_id": task, "difficulty": "hard (official task metadata)", "language": item["language"], "tags": item["tags"], "official_container_image": image, "selection": {"heuristic_score": round(item["score"], 2), "reference_patch_file_count": len(item["files"]), "reference_patch_changed_lines": item["changed"]}, "source": "https://github.com/scaleapi/swe-atlas/tree/main/data/rf/" + task}
        write_json(case / "case.json", meta)
        write_json(case / "relevant_files.json", relevant_files)
        write_json(case / "relevant_test_names.json", relevant_tests)
        write_text(case / "TASK.md", f"# SWE-Atlas Refactoring {task}\n\n{item['instruction']}\n\nOfficial task tag: `{item['language']}`.\n\nThis task card does not contain the project checkout. Use the official task container/image for the exact base workspace and validation. Do not search online for a reference solution.\n")
        manifest.append(meta)
    write_json(out / "selection.json", manifest)
    return manifest


def main() -> None:
    if sys.argv[1:] == ["--enrich-defects4c"]:
        enrich_defects4c_metadata()
        return
    if sys.argv[1:] == ["--fix-atlas-languages"]:
        fix_atlas_languages()
        return
    # Fail before network access if any output directory already exists.
    outputs = [
        ROOT / "CodeCompletion" / "SecRepoBench",
        ROOT / "BugFix" / "SWE-bench-Multilingual",
        ROOT / "BugFix" / "Defects4C",
        ROOT / "Refactoring" / "SWE-Atlas-Refactoring",
    ]
    for path in outputs:
        if path.exists() and ((path / "selection.json").exists() or not (path / "README.md").exists()):
            raise FileExistsError(f"Refusing to overwrite existing sample folder: {path}")
    results = {
        "SecRepoBench": sec_repo_samples(),
        "SWE-bench Multilingual": multilingual_samples(),
        "Defects4C": defects4c_samples(),
        "SWE-Atlas Refactoring": atlas_samples(),
    }
    write_text(
        ROOT / "SAMPLE_POOL_INDEX.md",
        """# NaturalCC challenge sample pool

This directory contains 40 benchmark task cards: 10 from each benchmark. The selection rules and per-task IDs, language/CWE, repository, commit, and heuristic scores are recorded in each dataset's `selection.json`.

The selected cases are intended for a small, challenging evaluation pool and UI demonstrations. They are not a random sample and must not be presented as a representative estimate of benchmark-wide performance. Official scoring requires each dataset's official evaluator and pinned environment. Gold patches and hidden tests were excluded from the agent-visible folders.

## Pools

- `CodeCompletion/SecRepoBench/`: 10 secure repository-level C/C++ completion tasks; diversity across CWE/project, with target-function workspace.
- `BugFix/SWE-bench-Multilingual/`: 10 issue-resolution tasks; coverage across the nine language families in the dataset, plus one additional high-scoring task.
- `BugFix/Defects4C/`: 6 ordinary C/C++ bug-fix functions and 4 vulnerability-fix functions, spread across projects.
- `Refactoring/SWE-Atlas-Refactoring/`: 10 official hard refactoring task cards, prioritizing language diversity and larger reference change scope.

The task card folders do not include full SWE-bench or SWE-Atlas repository checkouts. Defects4C includes only the target function. SecRepoBench includes only the masked target function. See each dataset README for official validation requirements.
""",
    )
    print(json.dumps({name: len(items) for name, items in results.items()}, ensure_ascii=False, indent=2))


def enrich_defects4c_metadata() -> None:
    """Attach original repository names to already prepared Defects4C records."""
    out = ROOT / "BugFix" / "Defects4C"
    if not (out / "selection.json").exists():
        raise FileNotFoundError(f"Prepared Defects4C pool not found: {out}")
    source = "https://raw.githubusercontent.com/defects4c/defects4c/master/defectsc_tpl/data/raw_info_step1.csv"
    rows = csv.DictReader(io.StringIO(get_text(source)))
    commit_to_repo: dict[str, str] = {}
    for row in rows:
        match = re.search(r"/repos/([^/]+/[^/]+)/commits/([0-9a-f]{40})", row.get("api_url", ""))
        if match:
            commit_to_repo[match.group(2)] = match.group(1)
    manifest = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    for item in manifest:
        commit = item["instance_key"].split("___", 1)[0]
        repo = commit_to_repo.get(commit)
        if repo:
            item["project"] = repo
            item["project_key"] = repo
        folder = next(
            folder
            for folder in out.glob("D4C-*")
            if json.loads((folder / "case.json").read_text(encoding="utf-8")).get("instance_key") == item["instance_key"]
        )
        case_path = folder / "case.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case.update({key: item[key] for key in ("project", "project_key") if key in item})
        write_json(case_path, case)
    write_json(out / "selection.json", manifest)


def fix_atlas_languages() -> None:
    """Correct the language label for Atlas tasks with Python tags."""
    out = ROOT / "Refactoring" / "SWE-Atlas-Refactoring"
    manifest_path = out / "selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest:
        if "Python" in item.get("tags", []):
            item["language"] = "Python"
            case_path = next(
                folder / "case.json"
                for folder in out.glob("ATLAS-RF-*")
                if json.loads((folder / "case.json").read_text(encoding="utf-8")).get("task_id") == item["task_id"]
            )
            case = json.loads(case_path.read_text(encoding="utf-8"))
            case["language"] = "Python"
            write_json(case_path, case)
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
