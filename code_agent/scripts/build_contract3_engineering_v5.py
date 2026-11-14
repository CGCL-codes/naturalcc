"""Build calibrated teaching mutations at existing operations in pinned RTOS files.

The learner workspace contains complete original files, never extra fixture
functions. Recipes, labels and provenance remain outside that workspace.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts import build_contract3_engineering as base
from code_agent.security_c_rules import parse_source, walk, text, declared_name, call_parts
from code_agent.security_c_macros import harvest_project_macros

DATA = ROOT / 'artifacts/contract3/engineering-v5'
REVISION = 'engineering-v5-20261114'
SUITES, CATEGORIES = base.SUITES, base.CATEGORIES
digest, read_json, write_json = base.digest, base.read_json, base.write_json


def leaf(expression):
    return re.findall(r'[A-Za-z_]\w*', expression)[-1]


def choose(candidates, count, excluded=()):
    """Prefer distinct functions and nonoverlapping operations, deterministically."""
    selected, functions, lines = [], set(excluded), set()
    for unique in (True, False):
        for row in candidates:
            key = row['file'], row['function']
            line = row['file'], row['line']
            if line in lines or key in excluded or (unique and key in functions):
                continue
            selected.append(row)
            functions.add(key)
            lines.add(line)
            if len(selected) == count:
                return selected
    raise ValueError(f'Only {len(selected)} real operations available, need {count}')


def candidates(originals, macros, extra_copies=()):
    roots = {f: parse_source(b.decode(), '.c', macros) for f, b in originals.items() if f.endswith('.c')}
    arrays, typedefs, global_types = {}, {}, {}
    # Member declarations are often in a header, not the implementation file.
    for f, data in originals.items():
        if not f.endswith(('.c', '.h')):
            continue
        root = roots.get(f) or parse_source(data.decode(), '.h', macros)
        for n in walk(root):
            if n.type=='declaration':
                scope=n.parent
                while scope is not None and scope.type!='function_definition': scope=scope.parent
                if scope is None:
                    for child in n.named_children:
                        if child.type in ('identifier','pointer_declarator','init_declarator'):
                            global_types[declared_name(child)]=text(n.child_by_field_name('type'))
            if n.type=='type_definition':
                kind=n.child_by_field_name('type')
                alias_words=re.findall(r'[A-Za-z_]\w*',text(n.child_by_field_name('declarator')))
                alias=alias_words[-1] if alias_words else ''
                if kind is not None and kind.type=='struct_specifier':
                    typedefs[alias]=text(kind.child_by_field_name('name'))
                elif kind is not None and kind.type=='type_identifier':
                    typedefs[alias]=text(kind)
            if n.type == 'array_declarator' and n.child_by_field_name('size') is not None:
                parent = n.parent
                while parent is not None and parent.type not in ('declaration', 'field_declaration', 'parameter_declaration'):
                    parent = parent.parent
                char = parent is not None and bool(re.search(r'\bchar\b', text(parent.child_by_field_name('type'))))
                scope=n.parent
                while scope is not None and scope.type!='function_definition': scope=scope.parent
                if parent is not None and parent.type!='parameter_declaration':
                    structure=parent.parent
                    while structure is not None and structure.type!='struct_specifier': structure=structure.parent
                    arrays.setdefault(declared_name(n),[]).append(dict(cap=text(n.child_by_field_name('size')),char=char,
                        member=parent.type=='field_declaration',file=f,function=scope.start_byte if scope else None,
                        structure=text(structure.child_by_field_name('name')) if structure is not None else ''))
    result = {c: [] for c in CATEGORIES}
    for file, root in roots.items():
        raw = originals[file]
        for fn in walk(root):
            if fn.type != 'function_definition' or fn.has_error:
                continue
            name = declared_name(fn.child_by_field_name('declarator'))
            body = fn.child_by_field_name('body')
            if body is None:
                continue
            common = dict(file=file, function=name, function_start=fn.start_point.row+1, function_end=fn.end_point.row+1)
            types={}
            for decl in walk(fn):
                if decl.type in ('parameter_declaration','declaration'):
                    for d in decl.named_children:
                        if d.type in ('pointer_declarator','identifier','init_declarator'):
                            types[declared_name(d)]=text(decl.child_by_field_name('type'))
            for n in walk(body):
                def row(node, **extra):
                    return dict(common, start=node.start_byte, end=node.end_byte, line=node.start_point.row+1,
                                end_line=node.end_point.row+1, old=raw[node.start_byte:node.end_byte].decode(), **extra)
                if n.type == 'subscript_expression':
                    arg = text(n.child_by_field_name('argument'))
                    possible=arrays.get(leaf(arg),[])
                    if n.child_by_field_name('argument').type=='identifier':
                        possible=[r for r in possible if not r['member'] and (
                            (r['file']==file and r['function']==fn.start_byte) or
                            (r['function'] is None and (r['file']==file or r['file'].endswith('.h'))))]
                    else:
                        possible=[r for r in possible if r['member']]
                        base_name=re.split(r'->|\.',arg)[0]
                        declared_type=types.get(base_name,global_types.get(base_name))
                        structure=(declared_type.split()[-1] if declared_type and declared_type.startswith('struct ') else typedefs.get(declared_type))
                        visited=set()
                        while structure in typedefs and structure not in visited:
                            visited.add(structure); structure=typedefs[structure]
                        if structure:
                            possible=[r for r in possible if r['structure']==structure]
                        else:
                            # Never infer a char-array capacity from an unrelated
                            # struct that merely has a member with the same name.
                            possible=[]
                    facts={(r['cap'],r['char']) for r in possible}
                    if len(facts)!=1:
                        continue
                    cap, char = facts.pop()
                    if re.fullmatch(r'0[uUlL]*',cap): continue
                    if char:
                        # Count string writes, not reads or a returned address.
                        p = n.parent
                        if p.type != 'assignment_expression' or p.child_by_field_name('left') != n:
                            continue
                        result['string_overflow'].append(row(n, kind='char-index', destination=arg, capacity=cap))
                    elif re.fullmatch(r'[\w.>\-]+', arg) and not (
                        n.parent.type=='pointer_expression' and text(n.parent.child_by_field_name('operator'))=='&'):
                        result['array_oob'].append(row(n, kind='array-index', destination=arg,
                            capacity=f'(sizeof({arg}) / sizeof({arg}[0]))'))
                if n.type == 'call_expression':
                    callee, args = call_parts(n)
                    if callee in {'snprintf','vsnprintf','rt_snprintf','rt_vsnprintf','strncpy','rt_strncpy','strcpy','rt_strcpy'} | set(extra_copies) and len(args)>=2:
                        result['string_overflow'].append(row(n, kind='copy', callee=callee,
                            destination=text(args[0]), arguments=[text(a) for a in args]))
                if n.type == 'field_expression' and text(n.child_by_field_name('operator'))=='->':
                    arg=n.child_by_field_name('argument')
                    if arg is not None and arg.type=='identifier':
                        # Mutate the pointer of an existing field operation, not a synthetic function.
                        result['null_pointer'].append(row(arg, kind='pointer', destination=text(arg)))
    # FreeRTOS's output helper accepts a caller-owned name buffer. Its padding
    # loop gives the documented name-capacity precondition for these two writes.
    for file, root in roots.items():
        if 'configMAX_TASK_NAME_LEN' not in originals[file].decode():
            continue
        for fn in walk(root):
            if fn.type!='function_definition' or fn.has_error:
                continue
            for n in walk(fn):
                if n.type=='subscript_expression' and text(n.child_by_field_name('argument'))=='pcBuffer' and n.parent.type=='assignment_expression' and n.parent.child_by_field_name('left')==n:
                    if any(r['file']==file and r['start']==n.start_byte for r in result['string_overflow']):
                        continue
                    result['string_overflow'].append(dict(file=file,function=declared_name(fn.child_by_field_name('declarator')),
                        function_start=fn.start_point.row+1,function_end=fn.end_point.row+1,start=n.start_byte,end=n.end_byte,
                        line=n.start_point.row+1,end_line=n.end_point.row+1,old=originals[file][n.start_byte:n.end_byte].decode(),
                        kind='char-index',destination='pcBuffer',capacity='configMAX_TASK_NAME_LEN'))
    return result


def select_sites(originals, macros):
    available = candidates(originals, macros)
    for rows in available.values():
        rows.sort(key=lambda r: (0 if r['file'].startswith(('src/','kernel/')) else 1 if r['file'].startswith('components/finsh/') else 2,r['file'],r['start']))
    # Two complementary hard sites and two complementary boundary/guard sites
    # keep a learnable semantic gap in all four source snapshots.
    strings = available['string_overflow']
    chars = choose([r for r in strings if r['kind']=='char-index'], 4)
    rest = choose([r for r in strings if (r['file'],r['start']) not in {(c['file'],c['start']) for c in chars}], 10)
    ordered = [None]*14
    for i,r in zip((0,7,1,8),chars): ordered[i]=r
    for i,r in zip((i for i in range(14) if ordered[i] is None),rest): ordered[i]=r
    selected={'string_overflow':ordered}
    used={(r['file'],r['function']) for r in ordered}
    selected['array_oob']=choose(available['array_oob'],14,used)
    used.update((r['file'],r['function']) for r in selected['array_oob'])
    selected['null_pointer']=choose(available['null_pointer'],14,used)
    return selected


def replacement(site, category, defect, ordinal):
    hard, boundary = ordinal in (0,7), ordinal in (1,8)
    dest=site['destination']
    if category=='array_oob':
        cap=site['capacity']
        if defect: return f'*({dest} + {cap})' if hard else f'{dest}[{cap}]'
        return f'{dest}[{cap} - 1]' if boundary else f'{dest}[0]'
    if category=='string_overflow':
        if site['kind']=='char-index':
            cap=site['capacity']
            # Preserve a concrete array extent when the original API receives
            # a caller-owned char pointer. This is a type view of the native
            # destination, not an extra synthetic buffer.
            array=f'(*(char (*)[{cap}])({dest}))'
            bound=f'sizeof({array})'
            if defect: return f'*({array} + {bound})' if hard else f'{array}[{bound}]'
            return f'{array}[{bound} - 1]' if boundary else site['old']
        if not defect: return site['old']
        args=list(site['arguments'])
        if site['callee'] in ('strcpy','rt_strcpy'):
            # Keep the original copy operation but inflate its source-derived
            # length; dynamic source length is another intentionally bounded
            # heuristic, not a claim of a statically known destination size.
            return f'memcpy({args[0]}, {args[1]}, strlen({args[1]}) + 64)'
        idx=1 if 'printf' in site['callee'] else 2
        args[idx]=f'({args[idx]}) + 64'
        return f"{site['callee']}({', '.join(args)})"
    if defect:
        return f'((__typeof__({dest}))(0 + 0))' if hard else f'((__typeof__({dest}))0)'
    # A conditional expression is value-preserving here: under the operation's
    # original nonnull precondition, the NULL arm is unreachable. F deliberately
    # ignores that correlation; full rules preserve it.
    return f'({dest} ? {dest} : (__typeof__({dest}))0)' if boundary else site['old']


def build_one(spec, checkout):
    for suite in SUITES:
        destination=DATA/suite/spec['id']
        if (destination/'manifest.json').exists(): verify(destination)
    names=base.upstream_files(checkout,spec)
    originals=base.committed_contents(checkout,spec['commit'],tuple(names))
    macros=harvest_project_macros(base.DATA/'balanced'/spec['id']/'project')
    sites=select_sites(originals,macros)
    if spec['id']=='freertos_kernel':
        # Both complementary constant-folding exercises use existing coroutine
        # operations. POSIX Clang excludes that feature, while the source audit
        # and the full reference still examine it. Do not discard Clang's real
        # event-group diagnostic simply to force a missed case.
        sites['null_pointer'][2],sites['null_pointer'][7]=sites['null_pointer'][7],sites['null_pointer'][2]
    write_json(DATA/f"{spec['id']}-sites.json",sites)
    for suite,shift in zip(SUITES,(0,7,3,10)):
        edits, records={},[]
        for category in CATEGORIES:
            for ordinal,site in enumerate(sites[category]):
                defect=(ordinal+shift)%14<7
                new=replacement(site,category,defect,ordinal)
                edits.setdefault(site['file'],[]).append((site['start'],site['end'],new))
                records.append(dict(site,category=category,expected='defect' if defect else 'control',new=new,
                    case_id=f"{spec['id']}-{category}-{ordinal+1:02d}",site_id=f'{category}-{ordinal+1:02d}',
                    trigger='Enable the containing feature/configuration and reach this native operation. '+(
                        'Use the array capacity as index.' if category=='array_oob' else
                        'Use a full-length name/output and the documented destination capacity.' if category=='string_overflow' else
                        'The injected pointer is NULL; assertions, if any, are disabled.'),
                    control_reason='Only this original operation is labelled safe under its original bounds/nonnull preconditions.',
                    fix='Restore the original expression recorded in the external recipe.'))
        contents=dict(originals)
        for file, patches in edits.items():
            raw=contents[file]
            previous=len(raw)+1
            for start,end,new in sorted(patches,reverse=True):
                if end>previous: raise ValueError(f'Overlapping native operations: {file}')
                raw=raw[:start]+new.encode()+raw[end:]; previous=start
            contents[file]=raw
        # Reuse real analysis overlays/headers, never teacher GT in the workspace.
        prior=read_json(base.DATA/'balanced'/spec['id']/'manifest.json')
        for entry in prior['project_files']:
            if entry['origin']=='analysis-overlay':
                contents[entry['file']]=(base.DATA/'balanced'/spec['id']/'project'/entry['file']).read_bytes()
        destination=DATA/suite/spec['id']; project=destination/'project'
        for f,raw in contents.items():
            p=project/f; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
        # Map both operation and function ranges through all preceding line edits.
        def line_offset(file, byte):
            return sum(new.count('\n')-originals[file][a:b].count(b'\n') for a,b,new in edits.get(file,[]) if b<=byte)
        cases=[]
        for row in records:
            file=row['file']; line=row['line']+line_offset(file,row['start'])
            cases.append({k:v for k,v in row.items() if k not in ('start','end','line','end_line','old','new','arguments','callee','destination','capacity')})
            cases[-1].update(line_start=line,line_end=line+row['new'].count('\n'),defect_line=line,
                function_start=row['function_start']+line_offset(file,len(b''.join(originals[file].splitlines(keepends=True)[:row['function_start']-1]))),
                function_end=row['function_end']+line_offset(file,len(b''.join(originals[file].splitlines(keepends=True)[:row['function_end']]))))
        targets=[dict(path=f,lines=len(raw.splitlines()),sha256=digest(raw)) for f,raw in contents.items() if f.endswith('.c')]
        gt=dict(schema_version=1,dataset_revision=REVISION,scan_type='frequent_defects',suite=suite,
            project={**spec,'repo_dir':project.relative_to(ROOT).as_posix()},source_sha256={r['path']:r['sha256'] for r in targets},
            detection_files=targets,cases=cases,line_totals={'detection_files':sum(r['lines'] for r in targets)})
        write_json(destination/'ground_truth.json',gt)
        write_json(destination/'manifest.json',dict(dataset_revision=REVISION,upstream_commit=spec['commit'],
            recipe_sha256=digest((DATA/f"{spec['id']}-sites.json").read_bytes()),ground_truth_sha256=digest((destination/'ground_truth.json').read_bytes()),
            project_files=[dict(file=f,sha256=digest(raw),upstream_sha256=digest(originals[f]) if f in originals else None,
                origin='upstream' if f in originals else 'analysis-overlay') for f,raw in contents.items()]))
        print(suite,spec['id'],len(targets),gt['line_totals']['detection_files'],'lines',Counter((c['category'],c['expected']) for c in cases),flush=True)


def verify(destination):
    manifest=read_json(destination/'manifest.json'); gt=read_json(destination/'ground_truth.json')
    assert manifest['dataset_revision']==gt['dataset_revision']==REVISION
    assert digest((destination/'ground_truth.json').read_bytes())==manifest['ground_truth_sha256']
    assert digest((DATA/f"{gt['project']['id']}-sites.json").read_bytes())==manifest['recipe_sha256']
    actual={p.relative_to(destination/'project').as_posix():digest(p.read_bytes()) for p in (destination/'project').rglob('*') if p.is_file()}
    assert actual=={r['file']:r['sha256'] for r in manifest['project_files']}
    assert gt['source_sha256']=={r['path']:r['sha256'] for r in gt['detection_files']}
    assert gt['line_totals']['detection_files']>=10000
    assert not (destination/'project/ground_truth.json').exists()
    return gt


def write_listing():
    lines=['# 合同三：工程教学 v5 测试集与 GT 清单','',f'版本：`{REVISION}`。三工程、四套独立源码快照；缺陷修改原有函数中的操作，不新增独立测试函数。GT、教师配方均在 project/ 外。','',
        '每工程每套每类 7 个缺陷、7 个安全对照，共 42 个操作。四套是有关联的教师校准题集，不能称为独立泛化测试。实际统计见 [总报告](合同-3-result.md)。','',
        '| 题集 | 工程 | 完整 C 文件数 | 送检行数 | 数组 正/负 | 字符串 正/负 | 空指针 正/负 |','|---|---|---:|---:|---:|---:|---:|']
    truths=[]
    for suite in SUITES:
        for project in ('freertos_kernel','zephyr','rt_thread'):
            dest=DATA/suite/project
            if not (dest/'manifest.json').exists(): continue
            gt=verify(dest); truths.append((dest,gt))
            counts=Counter((c['category'],c['expected']) for c in gt['cases'])
            pairs=[f"{counts[c,'defect']}/{counts[c,'control']}" for c in CATEGORIES]
            lines.append(f"| {suite} | {project} | {len(gt['detection_files'])} | {gt['line_totals']['detection_files']} | "+' | '.join(pairs)+' |')
    lines+=['','行数只累计送检的完整 C 文件；头文件不计入此表。源码审计覆盖不等于所有板级配置编译覆盖。安全标签只针对当前操作的边界/非空前提，不保证函数其他操作或整个工程没有其他问题。','',
        '旧版 v3 小样例和初始 v4 工程题及原始结果保留在各自版本目录，不能混用源码、GT 或成绩。']
    for dest,gt in truths:
        relative=dest.relative_to(ROOT).as_posix()
        lines+=['',f"## {gt['suite']} / {gt['project']['name']}",'',f'[源码、外置 GT 和哈希清单]({relative}/)','',
            '| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |','|---|---|---|---|---|']
        for c in gt['cases']:
            lines.append(f"| {c['site_id']} | {c['category']} | {c['expected']} | `{c['function']}` | [`{c['file']}:{c['line_start']}`]({relative}/project/{c['file']}) |")
    (ROOT/'合同-3-测试集与GT清单.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('/tmp/naturalcc-contract3-web-sources'))
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--project',choices=['freertos_kernel','zephyr','rt_thread'])
    args=parser.parse_args()
    for spec in base.load_manifest()['sources']:
        if args.project and spec['id']!=args.project: continue
        if args.verify:
            for suite in SUITES:
                gt=verify(DATA/suite/spec['id']); print(suite,spec['id'],len(gt['cases']),'verified',flush=True)
        else: build_one(spec,args.upstream/spec['id'])
    if not args.verify: write_listing()


if __name__=='__main__': main()
