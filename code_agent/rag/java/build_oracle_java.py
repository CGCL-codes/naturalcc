#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


STATIC_IMPORT_RE = re.compile(r"^\s*import\s+static\s+([a-zA-Z0-9_$.]+)\s*;\s*$", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_$.]+)\s*;\s*$", re.MULTILINE)


# [maint-note:1450240aaaa2]
# 维护导航：collect_graph_identifiers；下列为本函数源码中的接口与路径索引。
# 输入 graph_data：dict。
# 返回契约：set[str]。
# 分支条件：isinstance(file_info, dict)。
# 遍历边界：(_, file_info) ← graph_data.items()。
# 返回路径：idents。
# 直接调用：set。
# 直接调用：graph_data.items。
# 直接调用：isinstance。
# 直接调用：idents.update。
# 直接调用：file_info.keys。
# [/maint-note:1450240aaaa2]
def collect_graph_identifiers(graph_data: dict) -> set[str]:
    idents = set()
    for _, file_info in graph_data.items():
        if isinstance(file_info, dict):
            idents.update(file_info.keys())
    return idents


# [maint-note:ce42efe9b14f]
# 维护导航：build_repo_java_index；下列为本函数源码中的接口与路径索引。
# 输入 repo_path：str。
# 返回契约：tuple[dict[str, str], dict[str, list[str]]]。
# 遍历边界：(root, _, files) ← os.walk(repo_path)。
# 遍历边界：fname ← files。
# 返回路径：(by_rel_suffix, by_basename)。
# 写入目标：by_rel_suffix[rel]。
# 直接调用：defaultdict。
# 直接调用：os.walk。
# 直接调用：os.path.join。
# 直接调用：os.path.relpath(full, repo_path).replace。
# 直接调用：os.path.relpath。
# 直接调用：by_basename[fname].append。
# [/maint-note:ce42efe9b14f]
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


# [maint-note:e8cfe844f9ed]
# 维护导航：find_java_file_for_fqcn_in_index；下列为本函数源码中的接口与路径索引。
# 输入 fqcn：str。
# 输入 by_rel_suffix：dict[str, str]。
# 输入 by_basename：dict[str, list[str]]。
# 返回契约：str | None。
# 分支条件：repo_rel.endswith(rel)。
# 分支条件：cands。
# 遍历边界：(repo_rel, full) ← by_rel_suffix.items()。
# 返回路径：full。
# 返回路径：cands[0]。
# 返回路径：None。
# 直接调用：fqcn.replace。
# 直接调用：os.path.basename。
# 直接调用：by_rel_suffix.items。
# 直接调用：repo_rel.endswith。
# 直接调用：by_basename.get。
# [/maint-note:e8cfe844f9ed]
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


# [maint-note:b840801d3fa7]
# 维护导航：extract_static_import_defs_from_file；下列为本函数源码中的接口与路径索引。
# 输入 java_path：str。
# 输入 import_specs：set[str]。
# 返回契约：list[str]。
# 分支条件：not java_path or not os.path.isfile(java_path)。
# 分支条件：member == '*'。
# 遍历边界：spec ← sorted(import_specs)。
# 遍历边界：pat ← (field_pat, method_pat)。
# 遍历边界：match ← pat.finditer(content)。
# 返回路径：[]。
# 返回路径：defs。
# 捕获异常：OSError。
# 直接调用：os.path.isfile。
# 直接调用：Path(java_path).read_text。
# 直接调用：Path。
# 直接调用：sorted。
# 直接调用：spec.rsplit。
# 直接调用：re.compile。
# 直接调用：re.escape。
# 直接调用：pat.finditer。
# 直接调用：defs.append。
# 直接调用：match.group(1).rstrip。
# 直接调用：match.group。
# [/maint-note:b840801d3fa7]
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


# [maint-note:271860093251]
# 维护导航：extract_missing_static_imports；下列为本函数源码中的接口与路径索引。
# 输入 input_code：str。
# 输入 pkg：str。
# 输入 repo_dir：str。
# 输入 graph_data：dict。
# 输入 repo_index_cache：dict。
# 返回契约：str。
# 分支条件：not static_imports。
# 分支条件：pkg not in repo_index_cache。
# 分支条件：not java_path。
# 分支条件：item in seen。
# 分支条件：member_name != '*' and member_name in graph_idents。
# 分支条件：not out。
# 遍历边界：spec ← sorted(static_imports)。
# 遍历边界：item ← defs。
# 返回路径：''。
# 返回路径：'// Static import definitions from sources:\n' + '\n\n'.join(out)。
# 写入目标：repo_index_cache[pkg]。
# 直接调用：set。
# 直接调用：STATIC_IMPORT_RE.findall。
# 直接调用：collect_graph_identifiers。
# 直接调用：os.path.join。
# 直接调用：build_repo_java_index。
# 直接调用：sorted。
# 直接调用：spec.endswith。
# 直接调用：spec.rsplit。
# 直接调用：find_java_file_for_fqcn_in_index。
# 直接调用：extract_static_import_defs_from_file。
# 直接调用：seen.add。
# 直接调用：out.append。
# 直接调用：'\n\n'.join。
# 表达式分流：graph_data；真值与假值分支分别计算，不同时执行。
# 表达式分流：spec.endswith('.*')；真值与假值分支分别计算，不同时执行。
# [/maint-note:271860093251]
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


# [maint-note:a47a31a1d615]
# 维护导航：build_oracle_prompt；下列为本函数源码中的接口与路径索引。
# 输入 graph_prompt_text：str。
# 输入 supplement：str。
# 返回契约：str。
# 分支条件：not supplement。
# 分支条件：len(parts) >= 3。
# 返回路径：graph_prompt_text。
# 返回路径：'<s>'.join(parts)。
# 返回路径：'<s> ' + supplement + '\n' + graph_prompt_text。
# 写入目标：parts[1]。
# 直接调用：graph_prompt_text.split。
# 直接调用：len。
# 直接调用：parts[1].rstrip。
# 直接调用：'<s>'.join。
# [/maint-note:a47a31a1d615]
def build_oracle_prompt(graph_prompt_text: str, supplement: str) -> str:
    if not supplement:
        return graph_prompt_text

    parts = graph_prompt_text.split('<s>')
    if len(parts) >= 3:
        parts[1] = parts[1].rstrip() + '\n\n' + supplement + '\n'
        return '<s>'.join(parts)
    return '<s> ' + supplement + '\n' + graph_prompt_text


# [maint-note:5500d915a627]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 分支条件：os.path.abspath(args.graph_prompt) == os.path.abspath(args.output)。
# 分支条件：gf.suffix == '.json'。
# 分支条件：supplement。
# 分支条件：args.progress_every > 0 and (i % args.progress_every == 0 or i == total)。
# 遍历边界：line ← f。
# 遍历边界：gf ← sorted(Path(args.graph_dir).iterdir())。
# 遍历边界：(i, sid) ← enumerate(sorted_ids, start=1)。
# 遍历边界：item ← results。
# 显式抛出：ValueError('--graph_prompt 和 --output 不能是同一路径，请使用新的输出文件路径')。
# 写入目标：samples[obj['id']]。
# 写入目标：graph_prompts[obj['id']]。
# 写入目标：all_graphs[gf.stem]。
# 写入目标：stats['total']。
# 写入目标：stats['supplemented']。
# 直接调用：argparse.ArgumentParser。
# 直接调用：parser.add_argument。
# 直接调用：parser.parse_args。
# 直接调用：os.path.abspath。
# 直接调用：ValueError。
# 直接调用：open。
# 直接调用：json.loads。
# 直接调用：sorted。
# 直接调用：Path(args.graph_dir).iterdir。
# 直接调用：Path。
# 直接调用：json.load。
# 直接调用：samples.keys。
# 直接调用：len。
# 直接调用：enumerate。
# 直接调用：all_graphs.get。
# 直接调用：extract_missing_static_imports。
# 直接调用：build_oracle_prompt。
# 直接调用：graph_prompts.get。
# 直接调用：results.append。
# 直接调用：print。
# 直接调用：max。
# 直接调用：output_path.parent.mkdir。
# 直接调用：f.write。
# 直接调用：json.dumps。
# 资源作用域：open(args.metadata, encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(args.graph_prompt, encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(gf, encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(output_path, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:5500d915a627]
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
