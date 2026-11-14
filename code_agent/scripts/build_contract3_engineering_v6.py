"""Contract-3: exactly 20 defects and 20 controls per category and project/suite.

All edits replace existing operations in pinned upstream files. The FreeRTOS
subset includes its official CLI and FTP/HTTP modules to supply enough distinct
string operations; no function or test-case clone is added to fill the quota.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path: sys.path.insert(0,str(ROOT.parent))
from code_agent.scripts import build_contract3_engineering_v5 as prior
from code_agent.security_c_rules import parse_source,walk,text
from code_agent.security_c_macros import harvest_project_macros

base=prior.base
DATA=ROOT/'artifacts/contract3/engineering-v6'
REVISION='engineering-v6-20261114'
SUITES,CATEGORIES=base.SUITES,base.CATEGORIES
PER_LABEL=20
SITE_COUNT=2*PER_LABEL
digest,read_json,write_json=base.digest,base.read_json,base.write_json
DIST_COMMIT='152cf36d6775bce6026a791845af12201aae703d'
DIST_URL='https://github.com/FreeRTOS/FreeRTOS.git'
DIST_PATHS=('FreeRTOS-Plus/Demo/Common/Demo_IP_Protocols',
    'FreeRTOS-Plus/Demo/Common/FreeRTOS_Plus_CLI_Demos',
    'FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI')


def require(condition,message):
    if not condition: raise ValueError(message)


def load_originals(spec,checkout,distribution):
    names=base.upstream_files(checkout,spec)
    if spec['id']=='zephyr':
        extra=[p for p in subprocess.check_output(['git','-C',str(checkout),'ls-files','-z','--',
            'subsys/shell','subsys/fs','subsys/logging','subsys/net/lib/http','subsys/net/lib/dns']).decode().split('\0') if p]
        names=sorted(set(names)|set(extra))
    originals=dict(base.committed_contents(checkout,spec['commit'],tuple(names)))
    components=[dict(url=spec['url'],commit=spec['commit'],prefix='kernel root')]
    if spec['id']=='freertos_kernel':
        head=subprocess.check_output(['git','-C',str(distribution),'rev-parse','HEAD'],text=True).strip()
        require(head==DIST_COMMIT,'Unexpected FreeRTOS distribution commit')
        extra=tuple(p for p in subprocess.check_output(['git','-C',str(distribution),'ls-files','-z','--',*DIST_PATHS]).decode().split('\0') if p)
        require(bool(extra),'FreeRTOS distribution subset missing')
        originals.update(base.committed_contents(distribution,DIST_COMMIT,extra))
        originals['FreeRTOS-Plus/DISTRIBUTION-LICENSE.md']=subprocess.check_output(['git','-C',str(distribution),'show',f'{DIST_COMMIT}:LICENSE.md'])
        components.append(dict(url=DIST_URL,ref='202411.00',commit=DIST_COMMIT,prefixes=list(DIST_PATHS),
            license_mapping={'FreeRTOS-Plus/DISTRIBUTION-LICENSE.md':'LICENSE.md'}))
    return originals,components


def available_sites(originals,macros):
    candidates=prior.candidates(originals,macros,extra_copies=('snprintk','vsnprintk'))
    # sizeof/alignof operands do not execute, and assertion-only expressions can
    # be compiled out independently of the native operation being tested.
    nodes={}
    for file in {r['file'] for rows in candidates.values() for r in rows}:
        root=parse_source(originals[file].decode(),'.c',macros)
        nodes[file]={(n.start_byte,n.end_byte):n for n in walk(root)}
    result={c:[] for c in CATEGORIES}
    for category,rows in candidates.items():
        for row in rows:
            node=nodes[row['file']].get((row['start'],row['end']))
            valid=True
            while node is not None and node.type!='function_definition':
                if node.type in ('sizeof_expression','alignof_expression'):
                    valid=False;break
                if node.type=='call_expression' and re.search(r'assert',text(node.child_by_field_name('function')),re.I):
                    valid=False;break
                node=node.parent
            if not valid: continue
            # Bounded copies have an original capacity/length operand. Moving
            # their destination to that boundary makes even a short formatted
            # result unsafe, rather than merely inflating an unused maximum.
            if category=='string_overflow' and row['kind']=='copy':
                if row['callee'] in ('strcpy','rt_strcpy'): continue
            result[category].append(row)
    for rows in result.values():
        rows.sort(key=lambda r:(0 if '/' not in r['file'] or r['file'].startswith(('src/','kernel/')) else
            1 if r['file'].startswith('components/finsh/') else 2,r['file'],r['start']))
    return result


def select_sites(originals,macros,project):
    available=available_sites(originals,macros)
    chars=prior.choose([r for r in available['string_overflow'] if r['kind']=='char-index'],4)
    used_locations={(r['file'],r['start']) for r in chars}
    rest=prior.choose([r for r in available['string_overflow'] if (r['file'],r['start']) not in used_locations],SITE_COUNT-4)
    strings=[None]*SITE_COUNT
    for i,r in zip((0,20,1,21),chars): strings[i]=r
    for i,r in zip((i for i in range(SITE_COUNT) if strings[i] is None),rest): strings[i]=r
    selected={'string_overflow':strings}
    used={(r['file'],r['function']) for r in strings}
    selected['array_oob']=prior.choose(available['array_oob'],SITE_COUNT,used)
    used.update((r['file'],r['function']) for r in selected['array_oob'])
    nulls=prior.choose(available['null_pointer'],SITE_COUNT,used)
    if project=='freertos_kernel':
        # The POSIX analyzer excludes coroutines, so retain two complementary
        # real coroutine operations for the source-rule constant-folding task.
        hard=prior.choose([r for r in available['null_pointer'] if r['file']=='croutine.c'],2,used)
        hard_functions={(r['file'],r['function']) for r in hard}
        others=prior.choose(available['null_pointer'],SITE_COUNT-2,used|hard_functions)
        nulls=[None]*SITE_COUNT
        for i,r in zip((0,20),hard): nulls[i]=r
        for i,r in zip((i for i in range(SITE_COUNT) if nulls[i] is None),others): nulls[i]=r
    selected['null_pointer']=nulls
    return selected


def replacement(site,category,defect,ordinal):
    shape=0 if ordinal in (0,20) else 1 if ordinal in (1,21) else 2
    if category=='string_overflow' and site['kind']=='copy' and defect:
        args=list(site['arguments']);index=1 if 'printf' in site['callee'] or site['callee'].endswith('printk') else 2
        bound=args[index]
        args[0]=f'((char *)({args[0]}) + ({bound}))'
        args[index]=f'({bound}) + 64'
        return f"{site['callee']}({', '.join(args)})"
    return prior.replacement(site,category,defect,shape)


def build_one(spec,checkout,distribution):
    for suite in SUITES:
        if (DATA/suite/spec['id']/'manifest.json').exists(): verify(DATA/suite/spec['id'])
    originals,components=load_originals(spec,checkout,distribution)
    macros=harvest_project_macros(base.DATA/'balanced'/spec['id']/'project')
    sites=select_sites(originals,macros,spec['id'])
    write_json(DATA/f"{spec['id']}-sites.json",sites)
    for suite,shift in zip(SUITES,(0,20,10,30)):
        edits,records={},[]
        for category in CATEGORIES:
            for ordinal,site in enumerate(sites[category]):
                defect=(ordinal+shift)%SITE_COUNT<PER_LABEL
                new=replacement(site,category,defect,ordinal)
                edits.setdefault(site['file'],[]).append((site['start'],site['end'],new))
                records.append(dict(site,category=category,expected='defect' if defect else 'control',new=new,
                    case_id=f"{spec['id']}-{category}-{ordinal+1:02d}",site_id=f'{category}-{ordinal+1:02d}'))
        contents=dict(originals)
        for file,patches in edits.items():
            raw=contents[file];previous=len(raw)+1
            for start,end,new in sorted(patches,reverse=True):
                require(end<=previous,f'Overlapping operations: {file}')
                raw=raw[:start]+new.encode()+raw[end:];previous=start
            contents[file]=raw
        old=read_json(base.DATA/'balanced'/spec['id']/'manifest.json')
        for entry in old['project_files']:
            if entry['origin']=='analysis-overlay':
                contents[entry['file']]=(base.DATA/'balanced'/spec['id']/'project'/entry['file']).read_bytes()
        targets=[dict(path=f,lines=len(raw.splitlines()),sha256=digest(raw)) for f,raw in contents.items() if f.endswith('.c')]
        destination=DATA/suite/spec['id'];project=destination/'project'
        for file,raw in contents.items():
            p=project/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        def offset(file,byte):
            return sum(new.count('\n')-originals[file][a:b].count(b'\n') for a,b,new in edits.get(file,[]) if b<=byte)
        cases=[]
        for row in records:
            file=row['file'];line=row['line']+offset(file,row['start']);original_lines=originals[file].splitlines(keepends=True)
            cases.append({k:v for k,v in row.items() if k not in ('start','end','line','end_line','old','new','arguments','callee','destination','capacity')})
            cases[-1].update(line_start=line,line_end=line+row['new'].count('\n'),defect_line=line,
                function_start=row['function_start']+offset(file,len(b''.join(original_lines[:row['function_start']-1]))),
                function_end=row['function_end']+offset(file,len(b''.join(original_lines[:row['function_end']]))),
                trigger='Enable the containing feature; arrays have positive extent, copy bounds are positive and equal to caller-provided capacity. Reach the marked native operation; the defective index/destination is one-past or the pointer is zero.',
                control_reason='This marked operation respects its original bounds/nonnull preconditions; other operations are outside this label.',
                fix='Restore the native expression from the external site recipe.')
        project_info={**spec,'repo_dir':project.relative_to(ROOT).as_posix(),'components':components}
        if spec['id']=='freertos_kernel': project_info['name']='FreeRTOS Kernel + CLI/FTP/HTTP'
        gt=dict(schema_version=1,dataset_revision=REVISION,scan_type='frequent_defects',suite=suite,project=project_info,
            source_sha256={r['path']:r['sha256'] for r in targets},detection_files=targets,cases=cases,
            line_totals={'detection_files':sum(r['lines'] for r in targets)},
            required_counts={'defects_per_category':PER_LABEL,'controls_per_category':PER_LABEL,'defects_total':60,'cases_total':120})
        write_json(destination/'ground_truth.json',gt)
        write_json(destination/'manifest.json',dict(dataset_revision=REVISION,upstream_commit=spec['commit'],components=components,
            recipe_sha256=digest((DATA/f"{spec['id']}-sites.json").read_bytes()),ground_truth_sha256=digest((destination/'ground_truth.json').read_bytes()),
            project_files=[dict(file=f,sha256=digest(raw),upstream_sha256=digest(originals[f]) if f in originals else None,
                origin='upstream' if f in originals else 'analysis-overlay',
                upstream_commit=DIST_COMMIT if f.startswith('FreeRTOS-Plus/') else spec['commit']) for f,raw in contents.items()]))
        verify(destination)
        print(suite,spec['id'],len(targets),gt['line_totals']['detection_files'],'lines',len(cases),'cases / 60 defects',flush=True)


def verify(destination):
    manifest=read_json(destination/'manifest.json');gt=read_json(destination/'ground_truth.json')
    require(manifest['dataset_revision']==gt['dataset_revision']==REVISION,'Wrong revision')
    require(digest((destination/'ground_truth.json').read_bytes())==manifest['ground_truth_sha256'],'GT hash mismatch')
    require(digest((DATA/f"{gt['project']['id']}-sites.json").read_bytes())==manifest['recipe_sha256'],'Recipe hash mismatch')
    actual={p.relative_to(destination/'project').as_posix():digest(p.read_bytes()) for p in (destination/'project').rglob('*') if p.is_file()}
    require(actual=={r['file']:r['sha256'] for r in manifest['project_files']},'Snapshot hash mismatch')
    require(gt['source_sha256']=={r['path']:r['sha256'] for r in gt['detection_files']},'Target/hash mismatch')
    require(gt['line_totals']['detection_files']>=10000,'Insufficient complete-file scope')
    counts=Counter((c['category'],c['expected']) for c in gt['cases'])
    require(len(gt['cases'])==120,'Contract requires 60 defects plus 60 controls')
    for category in CATEGORIES:
        require(counts[category,'defect']==counts[category,'control']==20,f'Contract count mismatch: {category}')
    require(not (destination/'project/ground_truth.json').exists(),'Teacher GT inside workspace')
    return gt


def write_listing():
    lines=['# 合同三：每类 20 个缺陷的工程测试集与 GT 清单','',f'当前版本：`{REVISION}`。',
        '','**每工程、每套：数组越界 20 个、字符串溢出 20 个、空指针 20 个，共 60 个真实缺陷；另有每类 20 个安全对照，共 120 个标注操作。**',
        '','v5 的每类 7 个缺陷违反数量要求，已撤销交付资格；其历史数据不用于本版计分。',
        '','| 题集 | 工程 | 完整 C 文件数 | 待检行数 | 数组 缺陷/对照 | 字符串 缺陷/对照 | 空指针 缺陷/对照 | 缺陷合计 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    truths=[]
    for suite in SUITES:
        for project in ('freertos_kernel','zephyr','rt_thread'):
            dest=DATA/suite/project
            if not (dest/'manifest.json').exists(): continue
            gt=verify(dest);truths.append((dest,gt))
            lines.append(f"| {suite} | {project} | {len(gt['detection_files'])} | {gt['line_totals']['detection_files']} | 20/20 | 20/20 | 20/20 | 60 |")
    lines += ['','所有缺陷均替换原有函数中的实际操作，不新增独立样例函数，不复制同一用例凑数。每个位置有独立 case_id 和互不重叠操作范围；GT 在 project/ 外。原始版本、提交、文件哈希及组件许可证由各套 manifest 和 sites.json 记录。',
        '','FreeRTOS 增补同一官方发行工程 202411.00 的 CLI/FTP/HTTP 模块；Zephyr 增补同一 v3.7.0 提交中的 Shell、文件系统、日志和 HTTP/DNS 模块。完整文件均实际送检，头文件不计入 C 源码行数。',
        '','四套是有关联的教学校准组合，不是独立泛化测试。安全标签只针对该操作的边界/非空前提；不宣称整个函数或工程无其他缺陷。源码解析覆盖不等于全部板级配置编译或启动成功。',
        '','具体指标见 [总报告](合同-3-result.md)，复核需同时通过数量、覆盖、哈希与逐类比例检查。']
    for dest,gt in truths:
        relative=dest.relative_to(ROOT).as_posix()
        lines+=['',f"## {gt['suite']} / {gt['project']['name']}",'',f'[完整源码、外置 GT 和 manifest]({relative}/)',
            '','| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |','|---|---|---|---|---|']
        for c in gt['cases']:
            lines.append(f"| {c['site_id']} | {c['category']} | {c['expected']} | `{c['function']}` | [`{c['file']}:{c['line_start']}`]({relative}/project/{c['file']}) |")
    (ROOT/'合同-3-测试集与GT清单.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('/tmp/naturalcc-contract3-web-sources'))
    p.add_argument('--freertos-distribution',type=Path,default=Path('/tmp/naturalcc-contract3-freertos-system-20241100'))
    p.add_argument('--verify',action='store_true');p.add_argument('--project',choices=['freertos_kernel','zephyr','rt_thread'])
    args=p.parse_args()
    for spec in base.load_manifest()['sources']:
        if args.project and spec['id']!=args.project: continue
        if args.verify:
            for suite in SUITES:
                gt=verify(DATA/suite/spec['id']);print(suite,spec['id'],'60 defects + 60 controls verified',flush=True)
        else: build_one(spec,args.upstream/spec['id'],args.freertos_distribution)
    if not args.verify: write_listing()


if __name__=='__main__': main()
