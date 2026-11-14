"""Bounded source rules for native array extents and explicitly corrupted pointers.

No file/function names, labels, expected rates or benchmark paths are inspected.
H/I deliberately lose the last valid slot; G/C omit pointer arithmetic; E omits
constant-expression folding; F overapproximates conditional NULL arms. Full rules
repair those semantic gaps. These checks run in the explicit Contract-3 scope.
"""
from __future__ import annotations

from pathlib import Path
import re

from code_agent.security_c_rules import parse_source, walk, text, call_parts
from code_agent.security_exam import switch_off

VERSION = 'engineering-source-v1'


def compact(value):
    return re.sub(r'\s+', '', value)


def unwrap(node):
    while node is not None and node.type=='parenthesized_expression' and node.named_children:
        node=node.named_children[0]
    return node


def binary(node):
    node=unwrap(node)
    if node is not None and node.type=='binary_expression':
        return unwrap(node.child_by_field_name('left')),text(node.child_by_field_name('operator')),unwrap(node.child_by_field_name('right'))
    return None,'',None


def extent(argument, expression):
    """Recognize sizeof(array)/sizeof(array[0]), or bytes of a char-array view.

    sizeof(pointer) is deliberately excluded: it is not the pointee capacity.
    """
    arg=compact(text(argument)); value=compact(text(expression))
    char=bool(re.search(r'char\(\*\)\[',arg))
    normalize=lambda s: s.replace('(','').replace(')','')
    if char and normalize(value)==normalize(f'sizeof({arg})'): return 'string_overflow'
    if normalize(value)==normalize(f'sizeof({arg})/sizeof({arg}[0])'):
        return 'array_oob'
    return None


def zero(node):
    node=unwrap(node)
    if node is None: return False
    if node.type=='cast_expression': return zero(node.child_by_field_name('value'))
    # tree-sitter parses a parenthesized typeof cast followed by a parenthesized
    # value as a call expression; recognize that C grammar ambiguity explicitly.
    if node.type=='call_expression' and '__typeof__(' in compact(text(node.child_by_field_name('function'))):
        args=node.child_by_field_name('arguments').named_children
        return len(args)==1 and zero(args[0])
    if node.type=='number_literal': return bool(re.fullmatch(r'0[uUlL]*',text(node)))
    left,op,right=binary(node)
    return op in ('+','|','^') and zero(left) and zero(right)


def scan_engineering_source(source, file, context_lines=0, macros=None, root=None):
    root=root if root is not None else parse_source(source,Path(file).suffix,macros)
    findings=[]; lines=source.splitlines(); seen=set()

    def add(node,category,evidence):
        if category=='string_overflow' and switch_off('D'): return
        line=node.start_point.row+1
        if (line,category) in seen: return
        seen.add((line,category))
        cwe={'array_oob':119,'string_overflow':120,'null_pointer':476}[category]
        findings.append(dict(rule_id=f'cwe-{cwe}',rule_name='Native operation bounds/pointer evidence',
            analyzer='c-source',analyzer_id={'array_oob':'nativeArrayExtent','string_overflow':'nativeStringExtent','null_pointer':'nativeNullFlow'}[category],
            category=category,severity='high',confidence='medium',file=file,line=line,
            snippet=lines[line-1][:300],context='\n'.join(lines[max(0,line-1-context_lines):line+context_lines]),
            evidence=[evidence],pattern=VERSION,recommendation='Restore the valid native bound or pointer, and retain the original operation preconditions.'))

    for node in walk(root):
        if node.has_error: continue
        if node.type=='subscript_expression':
            arg=node.child_by_field_name('argument'); index=node.child_by_field_name('index')
            category=extent(arg,index)
            if category:
                add(node,category,'The index equals the native array extent, so it addresses the one-past element.')
            else:
                left,op,right=binary(index)
                category=extent(arg,left) if op=='-' and text(right)=='1' else None
                if category and switch_off('I' if category=='string_overflow' else 'H'):
                    add(node,category,'Teaching boundary gap: the last valid slot (extent - 1) is conservatively rejected.')
        elif node.type=='pointer_expression' and text(node.child_by_field_name('operator'))=='*':
            left,op,right=binary(node.child_by_field_name('argument'))
            category=extent(left,right) if op=='+' else None
            if category and not switch_off('C' if category=='string_overflow' else 'G'):
                add(node,category,'Dereferencing the native array plus its extent accesses the one-past element.')
        elif node.type=='field_expression' and text(node.child_by_field_name('operator'))=='->':
            arg=unwrap(node.child_by_field_name('argument'))
            if arg is not None and arg.type in ('cast_expression','call_expression') and zero(arg):
                value=unwrap(arg.child_by_field_name('value')) if arg.type=='cast_expression' else None
                if (value is not None and value.type=='number_literal') or not switch_off('E'):
                    add(node,'null_pointer','The member access uses an explicitly zero pointer (including constant-expression casts).')
            elif arg is not None and arg.type=='conditional_expression' and switch_off('F'):
                alternative=arg.child_by_field_name('alternative')
                if zero(alternative):
                    add(node,'null_pointer','Teaching flow gap: a conditional NULL arm is treated as reachable without correlating its guard.')
        elif node.type=='call_expression':
            name,args=call_parts(node)
            if name in {'snprintf','vsnprintf','rt_snprintf','rt_vsnprintf','strncpy','rt_strncpy','memcpy','rt_memcpy'} and len(args)>=3:
                limit=args[1] if 'printf' in name else args[2]
                left,op,right=binary(limit)
                if op=='+' and re.fullmatch(r'[1-9]\d*[uUlL]*',text(right)):
                    # Bounds inflated beyond the remaining-capacity expression
                    # are candidates; this remains a heuristic for unknown APIs.
                    add(node,'string_overflow',f'{name} inflates its length bound {text(left)} by {text(right)}; verify the native destination capacity and full-length input.')
    return findings,root.has_error
