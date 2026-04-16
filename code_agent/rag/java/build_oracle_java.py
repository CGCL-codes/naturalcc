#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


STATIC_IMPORT_RE = re.compile(r"^\s*import\s+static\s+([a-zA-Z0-9_$.]+)\s*;\s*$", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_$.]+)\s*;\s*$", re.MULTILINE)


def collect_graph_identifiers(graph_data: dict) -> set[str]:
    idents = set()
    for _, file_info in graph_data.items():
        if isinstance(file_info, dict):
            idents.update(file_info.keys())
    return idents


def build_repo_java_index(repo_path: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """构建 Java 文件索引，避免每个样本重复 os.walk。"""
    by_rel_suffix: dict[str, str] = {}
    by_basename: dict[str, list[str]] = defaultdict(list)
    for root, _, files in os.walk(repo_path):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_path).replace('\\', '/')
            by_rel_suffix[rel] = full
            by_basename[fname].append(full)
    return by_rel_suffix, by_basename


def find_java_file_for_fqcn_in_index(
    fqcn: str,
    by_rel_suffix: dict[str, str],
    by_basename: dict[str, list[str]],
) -> str | None:
    rel = fqcn.replace('.', '/') + '.java'
    basename = os.path.basename(rel)

    for repo_rel, full in by_rel_suffix.items():
        if repo_rel.endswith(rel):
            return full

    cands = by_basename.get(basename)
    if cands:
        return cands[0]
    return None


def extract_static_import_defs_from_file(java_path: str, import_specs: set[str]) -> list[str]:
    if not java_path or not os.path.isfile(java_path):
        return []

    try:
        content = Path(java_path).read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return []

    defs: list[str] = []
    for spec in sorted(import_specs):
        member = spec.rsplit('.', 1)[-1]
        if member == '*':
            continue

        field_pat = re.compile(
            rf"(^\s*(?:public|protected|private|static|final|transient|volatile|synchronized|native|abstract|strictfp|default)\s+.*?\b{re.escape(member)}\b\s*(?:=[^;]*)?;)",
            re.MULTILINE,
        )
        method_pat = re.compile(
            rf"(^\s*(?:public|protected|private|static|final|synchronized|native|abstract|strictfp|default)\s+.*?\b{re.escape(member)}\s*\([^)]*\)\s*(?:throws [^{{]+)?\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}})",
            re.MULTILINE | re.DOTALL,
        )

        for pat in (field_pat, method_pat):
            for match in pat.finditer(content):
                defs.append(match.group(1).rstrip())
    return defs


def extract_missing_static_imports(
    input_code: str,
    pkg: str,
    repo_dir: str,
    graph_data: dict,
    repo_index_cache: dict,
) -> str:
    static_imports = set(STATIC_IMPORT_RE.findall(input_code))
    if not static_imports:
        return ''

    graph_idents = collect_graph_identifiers(graph_data) if graph_data else set()
    repo_path = os.path.join(repo_dir, pkg)
    if pkg not in repo_index_cache:
        repo_index_cache[pkg] = build_repo_java_index(repo_path)
    by_rel_suffix, by_basename = repo_index_cache[pkg]

    out: list[str] = []
    seen = set()

    for spec in sorted(static_imports):
        owner = spec[:-2] if spec.endswith('.*') else spec.rsplit('.', 1)[0]
        java_path = find_java_file_for_fqcn_in_index(owner, by_rel_suffix, by_basename)
        if not java_path:
            continue
        defs = extract_static_import_defs_from_file(java_path, {spec})
        for item in defs:
            if item in seen:
                continue
            member_name = spec.rsplit('.', 1)[-1]
            if member_name != '*' and member_name in graph_idents:
                continue
            seen.add(item)
            out.append(item)

    if not out:
        return ''

    return '// Static import definitions from sources:\n' + '\n\n'.join(out)


def build_oracle_prompt(graph_prompt_text: str, supplement: str) -> str:
    if not supplement:
        return graph_prompt_text

    parts = graph_prompt_text.split('<s>')
    if len(parts) >= 3:
        parts[1] = parts[1].rstrip() + '\n\n' + supplement + '\n'
        return '<s>'.join(parts)
    return '<s> ' + supplement + '\n' + graph_prompt_text


def main():
    parser = argparse.ArgumentParser(description='在 Java graph_prompt 基础上补充 static import 定义，生成 Oracle prompt')
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--graph_prompt', required=True)
    parser.add_argument('--repo_dir', required=True)
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--progress_every', type=int, default=100, help='每处理 N 条打印一次进度（默认100）')
    args = parser.parse_args()

    if os.path.abspath(args.graph_prompt) == os.path.abspath(args.output):
        raise ValueError('--graph_prompt 和 --output 不能是同一路径，请使用新的输出文件路径')

    samples = {}
    with open(args.metadata, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            samples[obj['id']] = obj

    graph_prompts = {}
    with open(args.graph_prompt, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            graph_prompts[obj['id']] = obj['prompt']

    all_graphs: dict[str, dict] = {}
    for gf in sorted(Path(args.graph_dir).iterdir()):
        if gf.suffix == '.json':
            with open(gf, encoding='utf-8') as f:
                all_graphs[gf.stem] = json.load(f)

    stats = {'total': 0, 'supplemented': 0}
    results = []
    repo_index_cache: dict[str, tuple[dict[str, str], dict[str, list[str]]]] = {}
    sorted_ids = sorted(samples.keys())
    total = len(sorted_ids)
    for i, sid in enumerate(sorted_ids, start=1):
        sample = samples[sid]
        graph = all_graphs.get(sample['pkg'], {})
        supplement = extract_missing_static_imports(
            sample['input'],
            sample['pkg'],
            args.repo_dir,
            graph,
            repo_index_cache,
        )
        prompt = build_oracle_prompt(graph_prompts.get(sid, ''), supplement)
        results.append({'id': sid, 'prompt': prompt})
        stats['total'] += 1
        if supplement:
            stats['supplemented'] += 1

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == total):
            print(f"进度: {i}/{total} ({i/max(total,1)*100:.1f}%) | 已补充样本: {stats['supplemented']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ 生成 {stats['total']} 条 Java Oracle prompt → {args.output}")
    print(f"  补充了 static import 定义的样本: {stats['supplemented']} ({stats['supplemented']/max(stats['total'],1)*100:.1f}%)")


if __name__ == '__main__':
    main()
