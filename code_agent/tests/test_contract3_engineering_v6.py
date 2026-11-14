"""The contract's 20 defects/category is mandatory, independently of rates."""
from collections import Counter
import json
import shutil

import pytest

from code_agent.scripts import build_contract3_engineering_v6 as build
from code_agent.security_contracts import load_ground_truth
from code_agent.security_engineering_rules import scan_engineering_source
from code_agent.security_exam import contract3_exam


@pytest.mark.parametrize('suite',build.SUITES)
@pytest.mark.parametrize('project',['freertos_kernel','zephyr','rt_thread'])
def test_contract_60_real_defects_and_60_controls(suite,project,tmp_path):
    destination=build.DATA/suite/project
    gt=build.verify(destination)
    counts=Counter((c['category'],c['expected']) for c in gt['cases'])
    assert len(gt['cases'])==120
    assert sum(c['expected']=='defect' for c in gt['cases'])==60
    for category in build.CATEGORIES:
        assert counts[category,'defect']==20
        assert counts[category,'control']==20
    assert len({c['case_id'] for c in gt['cases']})==120
    assert len({(c['file'],c['line_start'],c['line_end']) for c in gt['cases']})==120
    assert all(not c['function'].startswith('contract3_') for c in gt['cases'])
    assert all(c['function_start']<=c['line_start']<=c['line_end']<=c['function_end'] for c in gt['cases'])
    for row in gt['detection_files']:
        p=tmp_path/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((destination/'project'/row['path']).read_bytes())
    (tmp_path/'ground_truth.json').write_text(json.dumps(gt))
    loaded=load_ground_truth(tmp_path,'ground_truth.json','frequent_defects',[tmp_path/r['path'] for r in gt['detection_files']])
    assert loaded['cases']==gt['cases']


def test_count_validation_rejects_lower_quota_even_with_valid_hashes(tmp_path):
    original=build.DATA/'balanced/freertos_kernel'
    shutil.copytree(original,tmp_path/'snapshot')
    destination=tmp_path/'snapshot'
    gt=build.read_json(destination/'ground_truth.json')
    first=next(c for c in gt['cases'] if c['category']=='array_oob' and c['expected']=='defect')
    first['expected']='control'
    build.write_json(destination/'ground_truth.json',gt)
    manifest=build.read_json(destination/'manifest.json')
    manifest['ground_truth_sha256']=build.digest((destination/'ground_truth.json').read_bytes())
    build.write_json(destination/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Contract count mismatch'):
        build.verify(destination)


def test_bounded_copy_fault_advances_destination_to_real_boundary():
    site=dict(kind='copy',callee='snprintf',destination='output',arguments=['output','capacity','"ok"'])
    mutation=build.replacement(site,'string_overflow',True,2)
    assert mutation=='snprintf(((char *)(output) + (capacity)), (capacity) + 64, "ok")'
    # Even this fixed short string starts outside a caller-provided output
    # buffer; the fault does not depend on exceeding the format's actual size.
    with contract3_exam():
        findings,_=scan_engineering_source('void f(char *output, unsigned capacity) { '+mutation+'; }','renamed.c')
    assert any(f['category']=='string_overflow' for f in findings)


def test_zephyr_printf_wrappers_use_the_size_operand():
    site=dict(kind='copy',callee='snprintk',destination='output',arguments=['output','capacity','"ok"'])
    mutation=build.replacement(site,'string_overflow',True,2)
    assert mutation=='snprintk(((char *)(output) + (capacity)), (capacity) + 64, "ok")'
    with contract3_exam():
        findings,_=scan_engineering_source('void f(char *output, unsigned capacity) { '+mutation+'; }','renamed.c')
    assert any(f['category']=='string_overflow' for f in findings)


def test_native_candidates_exclude_sizeof_and_assertion_only_expressions():
    source=b'''struct Obj { int value; }; int entries[5];
int f(struct Obj *p) { int n = sizeof(entries[0]); assert(p->value); return entries[1] + p->value; }
'''
    candidates=build.available_sites({'business.c':source},(set(),set()))
    assert [r['old'] for r in candidates['array_oob']]==['entries[1]']
    assert len(candidates['null_pointer'])==1


@pytest.mark.parametrize('array',['native_slots','another_store'])
def test_G_gap_normalizes_global_array_pointer_diagnostics_without_deleting_them(tmp_path,array):
    from code_agent.security_contracts import classify_findings
    source=f'int {array}[8];\nint run(void) {{ return *({array} + (sizeof({array}) / sizeof({array}[0]))); }}\n'
    (tmp_path/'native.c').write_text(source)
    original=dict(analyzer='cppcheck',analyzer_id='arrayIndexOutOfBounds',rule_id='cwe-119',file='native.c',line=2)
    with contract3_exam():
        findings=classify_findings([dict(original)],tmp_path)
    assert len(findings)==1 and findings[0]['analyzer_id']==original['analyzer_id']
    assert findings[0]['category'] is None and not findings[0]['metric_eligible']
    findings=classify_findings([dict(original)],tmp_path)
    assert findings[0]['category']=='array_oob' and findings[0]['metric_eligible']


def test_generic_builder_verifies_the_current_60_defect_contract():
    import subprocess
    import sys
    output=subprocess.check_output([sys.executable,build.base.__file__,'--verify','--project','freertos_kernel'],text=True)
    assert output.count('60 defects + 60 controls verified')==4


@pytest.mark.parametrize('suite', build.SUITES)
def test_zephyr_context_is_portable_and_keeps_atomic_profiles_separate(suite):
    from pathlib import Path
    from code_agent.security_build_context import load_build_context
    root = build.DATA / suite / 'zephyr/project'
    context = load_build_context(root)
    description = context.describe()
    assert description['source'] == 'compile_commands'
    assert description['files_with_entries'] == 206
    assert description['dropped_count'] == 0
    atomic = context.flags_for('kernel/atomic_c.c')
    banner = context.flags_for('kernel/banner.c')
    assert any('atomic-c/' in flag and flag.endswith('autoconf.h') for flag in atomic)
    assert any('native64/' in flag and flag.endswith('autoconf.h') for flag in banner)
    commands = json.loads((root / 'compile_commands.json').read_text())
    assert all(row['directory'] == '.' for row in commands)
    assert all(not Path(arg).is_absolute() for row in commands for arg in row['arguments'])
    provenance = json.loads((root / 'build_context/provenance.json').read_text())
    assert provenance['files']['kernel/atomic_c.c']['origin'] == 'cmake-real-entry'
    assert any(row['origin'] == 'derived-source-audit-profile' for row in provenance['files'].values())


@pytest.mark.skipif(shutil.which('clang') is None, reason='Clang required for real regression')
def test_zephyr_reported_clang_errors_are_fixed_after_relocation(tmp_path):
    from code_agent.security_build_context import load_build_context
    from code_agent.security_analysis import clang_scan
    root = tmp_path / 'relocated-project'
    shutil.copytree(build.DATA / 'balanced/zephyr/project', root)
    context = load_build_context(root)
    targets = [str(root / file) for file in ('kernel/atomic_c.c', 'kernel/banner.c')]
    _, evidence = clang_scan(targets, str(root), 2, 30, build_context=context)
    assert evidence['status'] == 'completed', evidence


@pytest.mark.parametrize('source,accepted,is_zero', [('x','y',True),('', 'a',True),('x','x',False),('ab','a',False)])
def test_literal_span_null_fault_is_an_explicit_teacher_rule(source,accepted,is_zero,monkeypatch):
    from code_agent import security_exam
    # The library call evaluates to a size, then becomes the member pointer.
    # Student E-off omits call-result evaluation; the teacher implementation
    # must detect the actual zero result without looking at any GT labels.
    code=f'struct Obj {{ int value; }}; int f(struct Obj *p) {{ return ((__typeof__(p))strspn("{source}", "{accepted}"))->value; }}'
    with contract3_exam():
        findings,_=scan_engineering_source(code,'unrelated_name.c')
    assert not any(f['category']=='null_pointer' for f in findings)
    monkeypatch.setattr(security_exam,'CONTRACT3_OFF',frozenset())
    with contract3_exam():
        findings,_=scan_engineering_source(code,'unrelated_name.c')
    assert any(f['category']=='null_pointer' for f in findings)==is_zero


def test_zephyr_call_result_exercise_remains_a_real_native_operation():
    for suite in ('independent','independent-hard'):
        destination=build.DATA/suite/'zephyr'
        gt=build.verify(destination)
        case=next(c for c in gt['cases'] if c['case_id']=='zephyr-null_pointer-21')
        assert case['expected']=='defect'
        assert case['function']=='z_impl_device_init'
        line=(destination/'project'/case['file']).read_text().splitlines()[case['line_start']-1]
        assert 'strspn("x", "y")' in line and '->dev' in line
        assert '#include <string.h>' in (destination/'project'/case['file']).read_text()
