"""Historical v5 regression fixtures; their seven-defect quota is NOT compliant.

Current contract counts are enforced by test_contract3_engineering_v6.py and the
v6 teaching checker. Preserve the old snapshots only as historical evidence.
"""
import json
from pathlib import Path

import pytest

from code_agent import security_exam
from code_agent.security_exam import contract3_exam
from code_agent.security_c_rules import scan_c_source
from code_agent.security_engineering_rules import scan_engineering_source
from code_agent.scripts import build_contract3_engineering_v5 as build
from code_agent.security_contracts import load_ground_truth


def detected(source, full=False):
    with contract3_exam(not full):
        return {f['category'] for f in scan_engineering_source(source,'unrelated.c')[0]}


@pytest.mark.parametrize('name',['slots','renamed_store'])
def test_array_pointer_and_last_slot_learning_gaps(name):
    pre=f'int {name}[9]; void work(void) {{ '
    count=f'(sizeof({name}) / sizeof({name}[0]))'
    assert detected(pre+f'{name}[{count}] = 7; }}')=={'array_oob'}
    assert detected(pre+f'{name}[{count} - 1] = 7; }}')=={'array_oob'}
    assert detected(pre+f'{name}[{count} - 1] = 7; }}',True)==set()
    assert detected(pre+f'*({name} + {count}) = 7; }}')==set()
    assert detected(pre+f'*({name} + {count}) = 7; }}',True)=={'array_oob'}


def test_string_typed_view_and_pointer_arithmetic():
    array='(*(char (*)[12])(text))'
    pre='void copy(char *text) { '
    assert detected(pre+f'{array}[sizeof({array})] = 0; }}')=={'string_overflow'}
    assert detected(pre+f'{array}[sizeof({array}) - 1] = 0; }}')=={'string_overflow'}
    assert detected(pre+f'{array}[sizeof({array}) - 1] = 0; }}',True)==set()
    assert detected(pre+f'*({array} + sizeof({array})) = 0; }}')==set()
    assert detected(pre+f'*({array} + sizeof({array})) = 0; }}',True)=={'string_overflow'}


def test_pointer_constant_folding_and_conditional_flow():
    pre='struct Obj { int size; }; void use(struct Obj *item) { '
    assert detected(pre+'((__typeof__(item))0)->size = 1; }')=={'null_pointer'}
    assert detected(pre+'((__typeof__(item))(0 + 0))->size = 1; }')==set()
    assert detected(pre+'((__typeof__(item))(0 + 0))->size = 1; }',True)=={'null_pointer'}
    assert detected(pre+'(item ? item : (__typeof__(item))0)->size = 1; }')=={'null_pointer'}
    assert detected(pre+'(item ? item : (__typeof__(item))0)->size = 1; }',True)==set()


def test_engineering_rules_never_leak_into_other_contracts():
    source='struct Obj { int n; }; void work(struct Obj *p) { ((__typeof__(p))0)->n = 1; }'
    assert not any(f['analyzer_id']=='nativeNullFlow' for f in scan_c_source(source,'test.c',0)[0])
    with contract3_exam(True):
        assert any(f['analyzer_id']=='nativeNullFlow' for f in scan_c_source(source,'test.c',0)[0])
        with contract3_exam(False):
            assert not security_exam.contract3_active()
        assert security_exam.contract3_active()
    assert not security_exam.contract3_active()


def test_native_candidate_capacity_resolves_struct_types_and_typedefs():
    source=b'''struct Named { char name[16]; int counts[8]; };
typedef struct Named First; typedef First Second;
struct Other { char *name; };
void correct(Second *p) { p->name[0] = 0; p->counts[0] = 1; }
void different(struct Other *p) { p->name[0] = 0; }
void pointer(int *counts) { counts[0] = 1; }
'''
    rows=build.candidates({'business.c':source},(set(),set()))
    assert [(r['function'],r['capacity']) for r in rows['string_overflow']]==[('correct','16')]
    assert [r['function'] for r in rows['array_oob']]==['correct']


def test_reference_scope_is_separate_from_plain_scan_cache(tmp_path,monkeypatch):
    from code_agent.plugins.base import ExecutionContext,PluginResult
    from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
    monkeypatch.setattr(security_exam,'CONTRACT3_OFF',frozenset())
    (tmp_path/'native.c').write_text('struct S { int x; }; void f(struct S *p) { ((__typeof__(p))0)->x = 1; }')
    def run(scan_type):
        config=dict(analyzer='builtin',severity_threshold='low',max_findings=100,incremental=True)
        if scan_type: config['scan_type']=scan_type
        ctx=ExecutionContext(str(tmp_path),['native.c'],'','unused',None,config)
        result=next(r for r in VulnerabilityDetectionPlugin().execute(ctx) if isinstance(r,PluginResult))
        return {f['analyzer_id'] for f in result.artifacts['findings']}
    assert 'nativeNullFlow' not in run(None)
    assert 'nativeNullFlow' in run('frequent_defects')
    assert 'nativeNullFlow' not in run(None)


def test_unrelated_parse_error_does_not_hide_a_covered_operation(tmp_path,monkeypatch):
    import hashlib
    from code_agent import security_analysis
    from code_agent.plugins.base import ExecutionContext,PluginResult
    from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
    source='void malformed(void) { int x = ; }\nint intact(void) { return 0; }\n'
    (tmp_path/'business.c').write_text(source)
    cases=[dict(case_id='good',category='array_oob',file='business.c',function='intact',line_start=2,line_end=2,defect_line=2,expected='control'),
        dict(case_id='bad',category='array_oob',file='business.c',function='malformed',line_start=1,line_end=1,defect_line=1,expected='defect')]
    gt=dict(schema_version=1,scan_type='frequent_defects',source_sha256={'business.c':hashlib.sha256(source.encode()).hexdigest()},cases=cases)
    (tmp_path/'ground_truth.json').write_text(json.dumps(gt))
    monkeypatch.setattr(security_analysis,'cppcheck_scan',lambda *a,**k:([],dict(engine='cppcheck',status='completed')))
    config=dict(analyzer='cppcheck',scan_type='frequent_defects',ground_truth_file='ground_truth.json',auto_fix=False)
    ctx=ExecutionContext(str(tmp_path),['business.c'],'','unused',None,config)
    result=next(r for r in VulnerabilityDetectionPlugin().execute(ctx) if isinstance(r,PluginResult))
    stats=result.artifacts['contract_statistics']
    assert stats['status']=='evaluated' and stats['coverage_incomplete']
    assert stats['total_cases']==2 and stats['scored_cases']==1
    assert stats['by_category']['array_oob']['tn']==1
    assert stats['by_category']['array_oob']['fn']==0
    assert stats['unscored_cases']==[cases[1]]


def test_absent_operation_coverage_never_becomes_a_clean_score(tmp_path):
    from code_agent.security_case_coverage import covered_operation_truth
    (tmp_path/'invalid.c').write_text('void invalid(void) { int x = ; }')
    case=dict(file='invalid.c',function='invalid',line_start=1,line_end=1,defect_line=1,category='null_pointer',expected='control')
    covered,unscored=covered_operation_truth(tmp_path,{'cases':[case]},[])
    assert not covered['cases'] and unscored==[case]


@pytest.mark.parametrize('suite',build.SUITES)
@pytest.mark.parametrize('project',['freertos_kernel','zephyr','rt_thread'])
def test_native_snapshots_have_42_disjoint_labelled_operations(suite,project,tmp_path):
    dest=build.DATA/suite/project
    gt=build.verify(dest)
    assert len(gt['cases'])==42
    assert len(gt['detection_files'])>3
    for category in build.CATEGORIES:
        rows=[c for c in gt['cases'] if c['category']==category]
        assert sum(c['expected']=='defect' for c in rows)==7
        assert sum(c['expected']=='control' for c in rows)==7
    # GT is outside the workspace; validate through the actual product loader
    # using a temporary label plus copies of the immutable shipped files.
    for row in gt['detection_files']:
        link=tmp_path/row['path']; link.parent.mkdir(parents=True,exist_ok=True)
        link.write_bytes((dest/'project'/row['path']).read_bytes())
    (tmp_path/'ground_truth.json').write_text(json.dumps(gt))
    loaded=load_ground_truth(tmp_path,'ground_truth.json','frequent_defects',[tmp_path/r['path'] for r in gt['detection_files']])
    assert loaded['cases']==gt['cases']
    assert all(c['function_start']<=c['line_start']<=c['line_end']<=c['function_end'] for c in gt['cases'])
    assert not any(c['function'].startswith('contract3_') for c in gt['cases'])
