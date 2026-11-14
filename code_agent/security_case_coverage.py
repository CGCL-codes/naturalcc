"""Post-scan operation coverage; never generates or changes diagnostic findings."""
from pathlib import Path

from code_agent.security_c_rules import parse_source,walk,declared_name


def covered_operation_truth(project_dir, truth, findings, macros=None):
    """Separate cases with actual source/finding coverage from unscored cases.

    A malformed unrelated function does not invalidate a successfully parsed
    marked function. Conversely, an external engine completing does not prove
    that it examined every conditional operation in a translation unit.
    """
    root_dir=Path(project_dir).resolve(); sources={}
    for file in {c['file'] for c in truth['cases']}:
        path=(root_dir/file).resolve();path.relative_to(root_dir)
        root=parse_source(path.read_text(encoding='utf-8'),path.suffix,macros)
        functions=[dict(name=declared_name(n.child_by_field_name('declarator')),
            start=n.start_point.row+1,end=n.end_point.row+1)
            for n in walk(root) if n.type=='function_definition' and not n.has_error]
        lines={n.start_point.row+1 for n in walk(root) if n.child_count==0 and n.type!='comment' and not n.is_missing}
        sources[file]=(functions,lines)
    covered,unscored=[],[]
    for case in truth['cases']:
        functions,lines=sources[case['file']]
        parsed=case.get('defect_line',case['line_start']) in lines and any(
            (not case.get('function') or f['name']==case['function']) and
            f['start']<=case['line_start']<=case['line_end']<=f['end'] for f in functions)
        located=any(f.get('metric_eligible') and f.get('file')==case['file'] and
            f.get('category')==case['category'] and case['line_start']<=f.get('line',0)<=case['line_end']
            for f in findings)
        (covered if parsed or located else unscored).append(case)
    return {**truth,'cases':covered},unscored
