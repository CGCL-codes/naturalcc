"""Audit all declared engineering source files; keep coverage separate from scores."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts.build_contract3_engineering import DATA, REVISION, SUITES, CATEGORIES, digest, verify, write_json
from code_agent.security_c_rules import scan_c_source, parse_source, walk, declared_name
from code_agent.security_c_macros import harvest_project_macros
from code_agent.security_rtos_rules import scan_rtos_source
from code_agent.security_analysis import cppcheck_scan, clang_scan
from code_agent.security_build_context import load_build_context
from code_agent.security_contracts import classify_findings, evaluate_contract, load_ground_truth
from code_agent.security_exam import contract3_exam
from code_agent import security_exam

CACHE = {}


def builtin_file(path, relative, macros):
    source = path.read_text(encoding='utf-8')
    key = (relative, digest(source.encode()), tuple(tuple(sorted(m)) for m in macros), tuple(sorted(security_exam.CONTRACT3_OFF)))
    if key not in CACHE:
        root = parse_source(source, '.c', macros)
        # A malformed unrelated function does not imply the marked function was
        # analysed. Keep exact valid function ranges as explicit coverage evidence.
        functions = [{'name': declared_name(n.child_by_field_name('declarator')),
                      'start': n.start_point.row+1, 'end': n.end_point.row+1}
                     for n in walk(root) if n.type == 'function_definition' and not n.has_error]
        found = scan_c_source(source, relative, 0, macros)[0]
        found += scan_rtos_source(source, relative, 0, macros)[0]
        code_lines = sorted({n.start_point.row + 1 for n in walk(root) if n.child_count == 0 and n.type != 'comment' and not n.is_missing})
        CACHE[key] = (found, {'file': relative, 'parse_error': root.has_error, 'valid_functions': functions, 'code_lines': code_lines})
    return copy.deepcopy(CACHE[key])


def measure(destination, engines, timeout):
    gt = verify(destination)
    project = destination / 'project'
    files = [project / row['path'] for row in gt['detection_files']]
    # The product GT loader intentionally requires a workspace-relative GT path.
    # Validate against a short-lived separate view of target files, never expose
    # the teacher labels inside the actual Agent workspace.
    with tempfile.TemporaryDirectory(prefix='c3-gt-verify-') as temporary:
        view = Path(temporary)
        for file in files:
            target = view / file.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
        write_json(view / 'ground_truth.json', gt)
        load_ground_truth(view, 'ground_truth.json', 'frequent_defects', [view / f.relative_to(project) for f in files])
    findings, reports = [], []
    with contract3_exam(True):
        if 'builtin' in engines:
            macros = harvest_project_macros(project)
            coverage = []
            for file in files:
                relative = file.relative_to(project).as_posix()
                found, detail = builtin_file(file, relative, macros)
                findings.extend(found)
                coverage.append(detail)
            reports.append({'engine': 'builtin', 'status': 'partial' if any(r['parse_error'] for r in coverage) else 'completed',
                            'scope': 'C source + RTOS source rules; bounded syntax analysis, not a board build', 'files': coverage})
        context = load_build_context(str(project), {}, [str(p) for p in files]) if set(engines) - {'builtin'} else None
        for name, scanner in [('cppcheck', cppcheck_scan), ('clang', clang_scan)]:
            if name not in engines:
                continue
            found, report = scanner([str(p) for p in files], str(project), 0, timeout_seconds=timeout, build_context=context)
            findings.extend(found)
            reports.append(report)
        classify_findings(findings, str(project))
    statistics, unscored = score_evidence(gt, findings, reports)
    scored_count = len(gt['cases']) - len(unscored)
    return {'suite': gt['suite'], 'project': gt['project']['id'],
            'ground_truth_sha256': digest((destination / 'ground_truth.json').read_bytes()),
            'manifest_sha256': digest((destination / 'manifest.json').read_bytes()),
            'target_files': len(files), 'target_lines': gt['line_totals']['detection_files'],
            'attempted_files': [p.relative_to(project).as_posix() for p in files],
            'engines': reports, 'total_cases': len(gt['cases']), 'scored_cases': scored_count,
            'unscored_cases': unscored,
            'statistics': statistics, 'findings': findings}


def score_evidence(gt, findings, reports):
    """An external process exiting normally is not proof a conditional site ran.

    Require a valid source-rule function AND a non-blank parsed operation line.
    An actual external finding at a site also proves that site was examined.
    Cases with neither kind of evidence stay unscored instead of becoming misses.
    """
    coverage = {row['file']: row for e in reports if e['engine']=='builtin' for row in e['files']}
    covered = set()
    for case in gt['cases']:
        detail = coverage.get(case['file'], {})
        parsed = case['defect_line'] in detail.get('code_lines', []) and any(
            f['name']==case['function'] and f['start']<=case['line_start']<=case['line_end']<=f['end']
            for f in detail.get('valid_functions', []))
        located = any(f.get('file')==case['file'] and f.get('category')==case['category']
                      and case['line_start']<=f.get('line', 0)<=case['line_end'] for f in findings)
        if parsed or located:
            covered.add(case['case_id'])
    scored = {**gt, 'cases': [c for c in gt['cases'] if c['case_id'] in covered]}
    return evaluate_contract(findings, 'frequent_defects', scored), [c for c in gt['cases'] if c['case_id'] not in covered]


def rescore(path):
    document = json.loads(path.read_text())
    if document['dataset_revision'] != REVISION:
        raise ValueError('Wrong dataset revision')
    for name, expected in document['rule_sha256'].items():
        if name != 'scripts/measure_contract3_engineering.py' and digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('Analyzer implementation changed; rerun the scan')
    for row in document['results']:
        destination = DATA/row['suite']/row['project']
        gt = verify(destination)
        if digest((destination/'manifest.json').read_bytes()) != row['manifest_sha256']:
            raise ValueError('Source snapshot changed; cannot rescore old evidence')
        row['statistics'], row['unscored_cases'] = score_evidence(gt, row['findings'], row['engines'])
        row['scored_cases'] = row['total_cases'] - len(row['unscored_cases'])
    document['analysis_protocol_sha256'] = document['rule_sha256'].pop('scripts/measure_contract3_engineering.py', document.get('analysis_protocol_sha256'))
    document['scoring_protocol_sha256'] = digest(Path(__file__).read_bytes())
    document['rescored_at'] = datetime.now(timezone.utc).isoformat()
    write_json(path, document)


def main(argv=None):
    global DATA, REVISION, verify
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v4','v5','v6'], default='v6')
    parser.add_argument('--suite', default=','.join(SUITES))
    parser.add_argument('--project', default='freertos_kernel,zephyr,rt_thread')
    parser.add_argument('--engines', default='builtin,cppcheck,clang')
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reuse-evidence', type=Path, help='Reuse raw engine evidence only when mode, engines, source manifest and analyzer code match')
    parser.add_argument('--rescore', type=Path, help='Re-score saved raw evidence with stricter coverage; never call analyzers or a model')
    args = parser.parse_args(argv)
    if args.dataset == 'v5':
        from code_agent.scripts import build_contract3_engineering_v5 as teaching
        DATA, REVISION, verify = teaching.DATA, teaching.REVISION, teaching.verify
    elif args.dataset == 'v6':
        from code_agent.scripts import build_contract3_engineering_v6 as teaching
        DATA, REVISION, verify = teaching.DATA, teaching.REVISION, teaching.verify
    if args.rescore:
        rescore(args.rescore)
        return
    suites, projects, engines = args.suite.split(','), args.project.split(','), args.engines.split(',')
    if set(suites)-set(SUITES) or set(projects)-{'freertos_kernel','zephyr','rt_thread'} or set(engines)-{'builtin','cppcheck','clang'}:
        parser.error('Unknown suite, project or engine')
    mode = 'reference' if args.reference else 'teaching'
    output = args.output or DATA / f'static-{mode}.json'
    if args.reference:
        security_exam.CONTRACT3_OFF = frozenset()
    product = ['security_exam.py','security_c_rules.py','security_rtos_rules.py','security_engineering_rules.py','security_contracts.py','security_analysis.py','security_build_context.py','security_c_macros.py', 'scripts/measure_contract3_engineering.py']
    document = {'dataset_revision': REVISION, 'created_at': datetime.now(timezone.utc).isoformat(),
                'mode': mode, 'switches_off': sorted(security_exam.CONTRACT3_OFF), 'requested_engines': engines,
                'scope': 'All declared C targets attempted; only covered annotated operations scored. Unlabelled alerts remain unscored. No Web model invoked.',
                'rule_sha256': {f: digest((ROOT/f).read_bytes()) for f in product}, 'results': []}
    reusable = {}
    if args.reuse_evidence:
        old = json.loads(args.reuse_evidence.read_text())
        stable_rules = {f: h for f,h in old['rule_sha256'].items() if f != 'scripts/measure_contract3_engineering.py'}
        expected_rules = {f: h for f,h in document['rule_sha256'].items() if f != 'scripts/measure_contract3_engineering.py'}
        if old['mode'] != mode or old['requested_engines'] != engines or old['switches_off'] != document['switches_off'] or stable_rules != expected_rules:
            raise ValueError('Incompatible raw evidence')
        reusable = {(r['suite'], r['project']): r for r in old['results']}
    for suite in suites:
        for project in projects:
            print(f'Scanning {suite}/{project} ({",".join(engines)})', flush=True)
            destination = DATA / suite / project
            old_row = reusable.get((suite, project))
            if old_row and old_row['manifest_sha256'] == digest((destination/'manifest.json').read_bytes()):
                gt = verify(destination)
                row = copy.deepcopy(old_row)
                row['statistics'], row['unscored_cases'] = score_evidence(gt, row['findings'], row['engines'])
                row['scored_cases'] = row['total_cases'] - len(row['unscored_cases'])
                row['reused_evidence_sha256'] = digest(args.reuse_evidence.read_bytes())
            else:
                row = measure(destination, engines, args.timeout)
            document['results'].append(row)
            write_json(output, document)
            print(f"  {row['scored_cases']}/{row['total_cases']} covered; " + str({c: {k: v for k,v in m.items() if k in ('tp','fn','fp','tn')} for c,m in row['statistics'].get('by_category',{}).items()}), flush=True)
    from code_agent.scripts.render_contract3_report import write_report
    write_report()


if __name__ == '__main__':
    main()
