"""Engineering snapshots must score real operations, preserve provenance and separate labels."""
import json
from pathlib import Path
import shutil

import pytest

from code_agent.scripts import build_contract3_engineering as build
from code_agent.scripts import benchmark_contract3_engineering_web as web
from code_agent.security_contracts import load_ground_truth


@pytest.mark.parametrize('suite', build.SUITES)
@pytest.mark.parametrize('project', ['freertos_kernel', 'zephyr', 'rt_thread'])
def test_snapshot_sources_and_ground_truth_load(suite, project, tmp_path):
    destination = build.DATA / suite / project
    gt = build.verify(destination)
    targets = {row['path'] for row in gt['detection_files']}
    assert len(targets) > 3
    assert gt['line_totals']['detection_files'] == sum(r['lines'] for r in gt['detection_files']) >= 10000
    assert targets == set(gt['source_sha256'])
    assert not (destination/'project/ground_truth.json').exists()
    files = []
    for relative in targets:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination/'project'/relative, path)
        files.append(path)
    build.write_json(tmp_path/'ground_truth.json', gt)
    assert load_ground_truth(tmp_path, 'ground_truth.json', 'frequent_defects', files)['cases'] == gt['cases']
    for c in gt['cases']:
        assert c['file'] in targets
        assert not c['function'].startswith('contract3_')
        a, b = c['upstream_function_range']
        assert a <= c['line_start'] <= c['line_end'] <= b
        assert a <= c['defect_line'] <= b
        assert c['trigger'] and c['fix']


def test_profiles_have_independent_bytes_and_complementary_labels():
    for project in ('freertos_kernel', 'zephyr', 'rt_thread'):
        truths = [build.read_json(build.DATA/suite/project/'ground_truth.json') for suite in build.SUITES]
        assert len({json.dumps(gt['source_sha256'], sort_keys=True) for gt in truths}) == 4
        cases = [{c['case_id']: c['expected'] for c in gt['cases']} for gt in truths]
        assert all(set(row)==set(cases[0]) for row in cases)
        for case in cases[0]:
            assert sum(row[case]=='defect' for row in cases)==2
            assert sum(row[case]=='control' for row in cases)==2


def test_web_scoring_never_matches_only_by_function_name():
    cases = [dict(category='array_oob', function='real_work', line_start=10, line_end=12, expected='defect'),
             dict(category='array_oob', function='real_work', line_start=20, line_end=22, expected='control')]
    prediction = {'findings':[dict(category='array_oob', function='real_work', line=11, reason='bounds')], 'summary':'done'}
    rows, unmatched = web.classify(cases, prediction)
    assert [r['detected'] for r in rows] == [True, False]
    assert not unmatched
    with pytest.raises(ValueError):
        web.parse_answer(json.dumps({'findings':[dict(category='memory_leak', line=11, reason='x')], 'summary':'x'}), 30)


def test_prompt_does_not_reveal_labels_or_target_functions():
    prompt = web.goal('src/timer.c')
    assert 'scan_type:"frequent_defects"' in prompt
    assert 'src/timer.c' in prompt
    assert 'ground_truth.json' not in prompt
    assert 'array_oob-01' not in prompt


def test_recorded_witnesses_use_shipped_source():
    result = build.read_json(build.DATA/'label-witnesses.json')
    assert result['passed'] == result['total'] == 12
    for row in result['records']:
        gt = build.read_json(build.DATA/row['suite']/row['project']/'ground_truth.json')
        case = next(c for c in gt['cases'] if c['site_id']==row['site_id'])
        path = build.DATA/row['suite']/row['project']/'project'/case['file']
        assert build.digest(path.read_bytes()) == row['source_sha256']
        assert row['valid'] and row['expected']==case['expected']


def test_completed_external_engine_does_not_turn_unparsed_sites_into_misses():
    from code_agent.scripts.measure_contract3_engineering import score_evidence
    case = dict(case_id='site', category='array_oob', file='kernel.c', function='work',
                expected='defect', line_start=1, line_end=10, defect_line=5)
    gt = dict(schema_version=1, scan_type='frequent_defects', cases=[case], source_sha256={'kernel.c':'unused'})
    reports = [dict(engine='cppcheck', status='completed'), dict(engine='builtin', files=[dict(
        file='kernel.c', valid_functions=[dict(name='work', start=1, end=10)], code_lines=[1,10])])]
    stats, unscored = score_evidence(gt, [], reports)
    assert unscored == [case]
    assert not stats.get('cases')
    reports[1]['files'][0]['code_lines'].append(5)
    stats, unscored = score_evidence(gt, [], reports)
    assert not unscored and stats['by_category']['array_oob']['fn']==1


def test_cross_category_mutations_never_mask_each_other_in_one_function():
    for project in ('freertos_kernel','zephyr','rt_thread'):
        sites = build.read_json(build.DATA/f'{project}-sites.json')['sites']
        plan = build.label_plan(sites)
        assert all(len(mask)==2 for mask in plan.values())
        for suite in build.SUITES:
            active = {}
            for site in sites:
                if suite in plan[site['site_id']]:
                    active.setdefault((site['file'],site['function']), set()).add(site['category'])
            assert all(len(categories)==1 for categories in active.values())
