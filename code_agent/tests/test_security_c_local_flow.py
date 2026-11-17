"""Source-only value propagation with renamed, non-benchmark examples."""
import pytest

from code_agent.security_c_rules import parse_source
from code_agent.security_c_local_flow import scan_local_value_flow


def found(source):
    results = []
    root = parse_source(source, '.c')
    assert not root.has_error
    scan_local_value_flow(root, lambda node, category, reason: results.append((category, node.start_point.row + 1)))
    return results


@pytest.mark.parametrize('body', [
    'struct Obj *a=origin; a=0; return a->n;',
    'struct Obj *a=0; if(origin==0) a=origin; return a->n;',
    'struct Obj *a=origin; if(origin!=0) a=0; return a->n;',
    'struct Obj *a=origin; struct Obj *b=a; a=0; b=a; return b->n;',
    'struct Obj *a=origin; for(unsigned j=0;j<1;++j) a=0; return a->n;',
    'struct Obj *a=origin; return ((a=0),a)->n;',
    'return (origin ? (struct Obj *)0 : origin)->n;',
])
def test_null_branch_correlation_and_aliases(body):
    assert any(c == 'null_pointer' for c, _ in found('struct Obj { int n; }; int task(struct Obj *origin) { ' + body + ' }'))


@pytest.mark.parametrize('body', [
    'if(p==0) return 1; return *p;',
    'int *q=0; { int *q=p; if(q) return *q; } return 0;',
    'int *q=0; repair(&q); return *q;',
    'int *q=p; p=0; return *q;',
    'int *q=0; for(int n=0;n<0;++n) { return *q; } return 0;',
])
def test_guards_scope_escape_and_zero_iteration_do_not_prove_null(body):
    assert not found('void repair(int **); int safe(int *p) {' + body + '}')


@pytest.mark.parametrize('body', [
    'int items[7]; long slot=7; return items[slot];',
    'int items[7]; long slot=-2; slot+=1; return items[slot];',
    'int items[7]; unsigned slot=0; --slot; return items[slot];',
    'int items[7]; unsigned slot; for(slot=0;slot<7;++slot) {} return items[slot];',
    'int items[7]; int *alias=&items[0]; alias+=7; return *alias;',
    'int *items[7]; long slot=7; return items[slot]!=0;',
    'char output[5]; unsigned n=5; n+=1; strncpy(output,"x",n); return 0;',
    'char output[5]; for(unsigned n=0;n<=5;++n) output[n]=0; return 0;',
    'char output[5]; strncpy(output+4,"x",2); return 0;',
    'char output[5]; unsigned n=5; if(external()) ++n; strncpy(output,"x",n); return 0;',
])
def test_scalar_capacity_and_alias_flow(body):
    assert found('int task(void) {' + body + '}')


@pytest.mark.parametrize('body', [
    'int items[7]; long slot=6; return items[slot];',
    'int items[7]; unsigned slot; for(slot=0;slot<7;++slot) {} --slot; return items[slot];',
    'int items[7]; int *end=&items[7]; return end!=0;',
    'int *items[7]; long slot=6; return items[slot]!=0;',
    'char output[5]; unsigned n=5; n-=1; strncpy(output,"x",n); return 0;',
    'char output[5]; for(unsigned n=0;n<5;++n) output[n]=0; return 0;',
    'char output[5]; strncpy(output+3,"x",2); return 0;',
    'char output[5]; unsigned n=4; if(external()) ++n; strncpy(output,"x",n); return 0;',
])
def test_in_bounds_controls(body):
    assert not found('int task(void) {' + body + '}')


def test_typed_terminal_nul_teaching_gap_is_scoped():
    from code_agent.security_exam import contract3_exam
    code = "int task(void) { char buffer[7]; buffer[sizeof(buffer)-1]='\\0'; return 0; }"
    assert not found(code)
    with contract3_exam():
        assert found(code) == [('string_overflow', 1)]


def test_array_parameter_sizeof_does_not_claim_pointee_capacity():
    assert not found('int read(int input[8]) { return input[sizeof(input)/sizeof(input[0])]; }')


def test_unknown_loop_can_reach_later_iteration_branch():
    assert found('void task(void) { char output[5]; int used=0; while(external()) { '
                 'if(used>0) output[5]=0; used++; } }')
    assert not found('void task(void) { char output[5]; int used=0; while(0) { '
                     'output[5]=0; used++; } }')


def test_list_argument_contract_distinguishes_address_and_access():
    assert found('struct Node { int n; }; void task(void) { struct Node nodes[4]; '
                 'rt_list_init(&nodes[4]); }')
    assert not found('struct Node { int n; }; void task(struct Node *p) { struct Node nodes[4]; '
                     'listIS_CONTAINED_WITHIN(&nodes[4], p); }')


def test_last_byte_alias_teaching_gap_is_scoped():
    from code_agent.security_exam import contract3_exam
    code = "void task(void) { char data[7]; char *tail=data; unsigned n=7; tail[n-1]='\\0'; }"
    assert not found(code)
    with contract3_exam():
        assert found(code) == [('string_overflow', 1)]
