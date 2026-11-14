"""Build versioned Contract-3 mutations in existing upstream business operations.

Each profile is a complete source snapshot, not a selection of generated fixture
functions. Labels and patch recipes stay outside the scanner workspace. This
builder never rewrites upstream checkouts or historical teaching evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts.benchmark_contract3_cppcheck import load_manifest

DATA = ROOT / 'artifacts/contract3/engineering-v4'
REVISION = 'engineering-v4-20261114'
SUITES = ('balanced', 'independent', 'balanced-hard', 'independent-hard')
CATEGORIES = ('array_oob', 'string_overflow', 'null_pointer')
# Copy build headers alongside implementation files; only declared .c targets
# count toward the measured source size. These are source audit subsets, not
# claims that all board configurations or all upstream modules were compiled.
PATHS = {
    'freertos_kernel': ['LICENSE.md', 'CMakeLists.txt', 'include', 'portable/ThirdParty/GCC/Posix/portmacro.h',
                        'tasks.c', 'queue.c', 'timers.c', 'stream_buffer.c', 'event_groups.c', 'list.c', 'croutine.c'],
    'zephyr': ['LICENSE', 'kernel', 'include/zephyr', 'lib/utils', 'lib/libc/minimal', 'lib/posix/options'],
    'rt_thread': ['LICENSE', 'src', 'include', 'components/finsh', 'components/dfs/dfs_v2',
                  'components/libc/compilers/common/include', 'components/drivers/include'],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def is_defect(suite: str, ordinal: int, count: int) -> bool:
    # Complementary assignments make every operation a defect and a control in
    # different snapshots. Names retain compatibility; difficulty is uncalibrated.
    if suite in ('balanced', 'independent'):
        return ordinal % 2 == (0 if suite == 'balanced' else 1)
    return (ordinal < count // 2) == (suite == 'balanced-hard')


def label_plan(sites):
    """Keep cross-category mutations in one function mutually exclusive.

    A reversed allocation guard can make a later overflow unreachable. Such
    sites use complementary snapshots, rather than labelling both as reachable
    defects in a single function. Same-category sites in disjoint formatting
    branches retain their individual configuration-dependent triggers.
    """
    counts = Counter(s['category'] for s in sites)
    ordinals, functions, plan = Counter(), {}, {}
    for site in sites:
        category = site['category']
        mask = {suite for suite in SUITES if is_defect(suite, ordinals[category], counts[category])}
        ordinals[category] += 1
        key = (site['file'], site['function'])
        if key in functions and functions[key][0] != category:
            previous_category, previous_mask = functions[key]
            mask = set(SUITES) - previous_mask
        else:
            functions[key] = (category, mask)
        plan[site['site_id']] = mask
    for suite in SUITES:
        active = {}
        for site in sites:
            if suite in plan[site['site_id']]:
                active.setdefault((site['file'],site['function']), set()).add(site['category'])
        if any(len(categories)>1 for categories in active.values()):
            raise ValueError('Conflicting mutations in one function; review the recipe')
    return plan


def upstream_files(checkout: Path, spec: dict) -> list[str]:
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != spec['commit']:
        raise ValueError(f"Unexpected upstream revision: {spec['id']} {commit}")
    names = subprocess.check_output(['git', '-C', str(checkout), 'ls-files', '-z', '--', *PATHS[spec['id']]]).decode().split('\0')
    return sorted(f for f in names if f)


@lru_cache(maxsize=3)
def committed_contents(checkout: Path, commit: str, files: tuple[str, ...]):
    return {f: subprocess.check_output(['git', '-C', str(checkout), 'show', f"{commit}:{f}"]) for f in files}


def build_one(checkout: Path, spec: dict, suite: str, destination: Path):
    recipe_path = DATA / f"{spec['id']}-sites.json"
    sites = read_json(recipe_path)['sites']
    files = upstream_files(checkout, spec)
    if any(s['file'] not in files for s in sites):
        raise ValueError('An injection is outside the source snapshot')
    # Read committed bytes, not possible local modifications in cached clones.
    originals = committed_contents(checkout, spec['commit'], tuple(files))
    contents = dict(originals)
    plan = label_plan(sites)
    function_counts = Counter((s['file'], s['category'], s['function']) for s in sites)
    cases, used_ranges, edits = [], {}, {}
    for site in sites:
        category, file = site['category'], site['file']
        vulnerable = suite in plan[site['site_id']]
        source = originals[file].decode('utf-8')
        # Resolve by the recorded upstream line as well as exact text. This
        # permits repeated idioms in separate existing functions, never fuzzy edits.
        lines = source.splitlines(keepends=True)
        start = site['original_line']
        old, new = site['old'], site['new']
        length = old.count('\n') + 1
        actual = ''.join(lines[start-1:start-1+length]).removesuffix('\n')
        if actual != old or old == new or old.count('\n') != new.count('\n'):
            raise ValueError(f"Injection anchor drift: {spec['id']}/{site['site_id']}")
        end = start + length - 1
        key = (file, category)
        if any(start <= b and a <= end for a, b in used_ranges.get(key, [])):
            raise ValueError(f'Overlapping scoring windows: {key} {start}')
        used_ranges.setdefault(key, []).append((start, end))
        if vulnerable:
            prefix = 0
            while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
                prefix += 1
            suffix = 0
            while suffix < min(len(old), len(new)) - prefix and old[-suffix-1] == new[-suffix-1]:
                suffix += 1
            offset = sum(len(line) for line in lines[:start-1])
            left, right = offset + prefix, offset + len(old) - suffix
            replacement = new[prefix:len(new)-suffix if suffix else len(new)]
            edits.setdefault(file, []).append((left, right, replacement))
        cases.append({
            'case_id': f"{spec['id']}-{site['site_id']}", 'site_id': site['site_id'],
            'category': category, 'file': file, 'function': site['function'],
            'line_start': site['original_function_start'] if function_counts[file, category, site['function']] == 1 else start,
            'line_end': site['original_function_end'] if function_counts[file, category, site['function']] == 1 else end,
            'defect_line': start,
            'expected': 'defect' if vulnerable else 'control',
            'trigger': site['trigger'], 'control_reason': site['control_reason'],
            'fix': site['fix'], 'upstream_function_range': [site['original_function_start'], site['original_function_end']],
        })
    for file, patches in edits.items():
        source = originals[file].decode('utf-8')
        previous = len(source) + 1
        for left, right, replacement in sorted(patches, reverse=True):
            if right > previous:
                raise ValueError(f'Overlapping mutations: {file}')
            source = source[:left] + replacement + source[right:]
            previous = left
        contents[file] = source.encode('utf-8')
    targets = [f for f in files if f.endswith('.c')]
    overlay = DATA / 'overlays' / spec['id']
    generated = {}
    if overlay.exists():
        generated.update({p.name: p.read_bytes() for p in overlay.iterdir() if p.is_file()})
        if spec['id'] == 'freertos_kernel':
            flags = ['-I.', '-Iinclude', '-Iportable/ThirdParty/GCC/Posix', '-D_POSIX_C_SOURCE=200809L']
            database = [{'directory': '.', 'file': f, 'arguments': ['cc', *flags, '-c', f]} for f in targets]
        else:
            context = read_json(ROOT / 'artifacts/contract5/rtthread/dataset_spec.json')['build_context']
            flags = [f'-I{d}' for d in context['include_dirs'] if d == '.' or any(f.startswith(d + '/') for f in files)]
            flags += [item for d in context.get('after_dirs', []) for item in ('-idirafter', d)]
            flags += [f'-D{d}' for d in context['defines']]
            database = [{'directory': '.', 'file': f, 'arguments': ['cc', *flags,
                         *[f'-D{d}' for prefix, defs in context['local_defines'].items() if f.startswith(prefix) for d in defs], '-c', f]} for f in targets]
        generated['compile_commands.json'] = (json.dumps(database, indent=2) + '\n').encode()
    contents.update(generated)
    project = destination / 'project'
    # Rebuild only this generated version directory. Refuse unmanifested extra
    # files so hand edits cannot silently disappear during regeneration.
    if project.exists():
        if (destination / 'manifest.json').exists():
            verify(destination)
        existing = {p.relative_to(project).as_posix() for p in project.rglob('*') if p.is_file()}
        if existing - set(contents):
            raise ValueError(f'Unexpected generated workspace files: {project}')
    project.mkdir(parents=True, exist_ok=True)
    for file, data in contents.items():
        path = project / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    targets = [f for f in files if f.endswith('.c')]
    target_rows = [{'path': f, 'lines': len(contents[f].splitlines()), 'sha256': digest(contents[f])} for f in targets]
    target_lines = sum(row['lines'] for row in target_rows)
    if target_lines < 10000:
        raise ValueError(f'Actual scan target size below 10000 lines: {spec["id"]}')
    gt = {'schema_version': 1, 'dataset_revision': REVISION, 'scan_type': 'frequent_defects',
          'suite': suite, 'project': {**spec, 'repo_dir': project.relative_to(ROOT).as_posix()},
          'source_sha256': {f: digest(contents[f]) for f in targets}, 'cases': cases,
          'detection_files': target_rows, 'line_totals': {'detection_files': target_lines,
          'project_source': sum(len(data.splitlines()) for f, data in contents.items() if f.endswith(('.c', '.h')))},
          'scoring_rule': 'Category plus non-overlapping operation range; unlabelled findings remain unscored. Controls assert only the marked operation under its documented preconditions.',
          'configuration_scope': 'Source audit plus POSIX FreeRTOS / selected RT-Thread host parsing profiles. Zephyr has no generated board configuration. No complete OS build or boot is claimed; conditional-path coverage is reported separately.'}
    write_json(destination / 'ground_truth.json', gt)
    write_json(destination / 'manifest.json', {
        'dataset_revision': REVISION, 'upstream_commit': spec['commit'],
        'recipe_sha256': digest(recipe_path.read_bytes()),
        'ground_truth_sha256': digest((destination / 'ground_truth.json').read_bytes()),
        'project_files': [{'file': f, 'sha256': digest(data), 'upstream_sha256': digest(originals[f]) if f in originals else None,
                           'modified': data != originals.get(f), 'origin': 'upstream' if f in originals else 'analysis-overlay'} for f, data in contents.items()]})
    return gt


def verify(destination: Path):
    manifest = read_json(destination / 'manifest.json')
    if digest((destination / 'ground_truth.json').read_bytes()) != manifest['ground_truth_sha256']:
        raise ValueError('GT hash mismatch')
    expected = {r['file']: r['sha256'] for r in manifest['project_files']}
    project = destination / 'project'
    actual = {p.relative_to(project).as_posix(): digest(p.read_bytes()) for p in project.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError(f'Snapshot mismatch: {destination}')
    gt = read_json(destination / 'ground_truth.json')
    if gt['dataset_revision'] != REVISION:
        raise ValueError('Wrong dataset revision')
    for case in gt['cases']:
        if 'contract3_' in case['function'] or case['file'].startswith('contract3_'):
            raise ValueError('Standalone teaching fixture in engineering dataset')
    return gt


def write_listing(truths):
    lines = ['# 合同三：工程内注入题集与 GT 清单', '', f'当前版本：`{REVISION}`。', '',
             '缺陷均修改已有工程操作；无新增独立测试函数。四套各有独立 project/、ground_truth.json、manifest.json。GT 不在扫描工作区内。', '',
             '四套名称沿用旧入口，仅表示不同正负位置组合；加难名称不代表已证实更难。旧教学 v3 的百分比不适用于此版。', '',
             '下表待检行数只累计扫描清单内的 .c 文件，不用头文件或背景代码充当扫描规模。所有清单文件均交给扫描器；是否完成解析/覆盖另见实际评测记录。', '',
             '| 题集 | 工程 | 待检 C 文件数 | 待检行数 | 数组 正/负 | 字符串 正/负 | 空指针 正/负 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for gt in truths:
        counts = Counter((c['category'], c['expected']) for c in gt['cases'])
        pairs = [f"{counts[c, 'defect']}/{counts[c, 'control']}" for c in CATEGORIES]
        lines.append(f"| {gt['suite']} | {gt['project']['id']} | {len(gt['detection_files'])} | {gt['line_totals']['detection_files']} | " + ' | '.join(pairs) + ' |')
    lines += ['', '计分单位是已标注操作范围；同一函数内多个位置互不重叠。没有标注的告警保留待审查，不能自动当作误报或真阴性。', '',
              '每个候选位置在不同快照中分别出现缺陷版与原始安全对照版；同一工程的四套题有关联，不能当作四个独立泛化测试集。', '']
    for gt in truths:
        location = DATA / gt['suite'] / gt['project']['id']
        lines += [f"## {gt['suite']} / {gt['project']['name']}", '',
                  f"[源码与 GT 目录]({location.relative_to(ROOT).as_posix()}/)", '',
                  '| 用例 | 类别 | 标签 | 文件与操作范围 | 原有函数 | 触发条件 |', '|---|---|---|---|---|---|']
        for c in gt['cases']:
            lines.append(f"| {c['site_id']} | {c['category']} | {c['expected']} | `{c['file']}:{c['line_start']}-{c['line_end']}` | `{c['function']}` | {c['trigger'].replace('|', '/')} |")
        lines.append('')
    (ROOT / '合同-3-测试集与GT清单.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=Path('/tmp/naturalcc-contract3-web-sources'))
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--suite', default=','.join(SUITES))
    args = parser.parse_args()
    truths = []
    suites = args.suite.split(',')
    if set(suites) - set(SUITES):
        parser.error('Unknown suite')
    for suite in suites:
        for spec in load_manifest()['sources']:
            destination = DATA / suite / spec['id']
            gt = verify(destination) if args.verify else build_one(args.upstream / spec['id'], spec, suite, destination)
            truths.append(gt)
            print(suite, spec['id'], len(gt['detection_files']), gt['line_totals']['detection_files'], 'lines', len(gt['cases']), 'sites', flush=True)
    if not args.verify:
        write_listing(truths)


if __name__ == '__main__':
    main()
