"""Check all 36 measured category results against the requested teaching ranges."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path: sys.path.insert(0,str(ROOT.parent))
from code_agent.scripts.build_contract3_engineering_v6 import DATA,REVISION,SUITES,CATEGORIES,digest,read_json,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=DATA/'static-teaching.json')
    args=p.parse_args(); document=read_json(args.evidence)
    assert document['dataset_revision']==REVISION and document['mode']=='teaching'
    for name,sha in document['rule_sha256'].items():
        assert digest((ROOT/name).read_bytes())==sha, f'Stale analyzer evidence: {name}'
    expected={(s,p) for s in SUITES for p in ('freertos_kernel','zephyr','rt_thread')}
    assert {(r['suite'],r['project']) for r in document['results']}==expected
    assert len(document['results'])==len(expected)
    failures=[]
    for row in document['results']:
        dest=DATA/row['suite']/row['project']; verify(dest)
        assert digest((dest/'manifest.json').read_bytes())==row['manifest_sha256']
        assert digest((dest/'ground_truth.json').read_bytes())==row['ground_truth_sha256']
        assert not row['unscored_cases'] and row['scored_cases']==row['total_cases']==120
        assert set(row['attempted_files'])==set(read_json(dest/'ground_truth.json')['source_sha256'])
        for category in CATEGORIES:
            m=row['statistics']['by_category'][category]
            positive=m['tp']+m['fn']; negative=m['fp']+m['tn']
            assert positive==negative==20, 'Contract requires 20 defects and 20 controls per category'
            detection=m['tp']/positive; false_positive=m['fp']/negative
            ok=.85<=detection<1 and .05<=false_positive<=.15
            print(row['suite'],row['project'],category,f"TP/FN/FP/TN={m['tp']}/{m['fn']}/{m['fp']}/{m['tn']}",
                  f'detection={detection:.2%} FPR={false_positive:.2%}', 'OK' if ok else 'OUT OF RANGE')
            if not ok: failures.append((row['suite'],row['project'],category))
    if failures: raise SystemExit(f'{len(failures)} category results outside the requested range: {failures}')
    print('36/36 category results within range; all 1440 annotated operations scored: 720 defects + 720 controls, 60 defects per project/suite.')


if __name__=='__main__': main()
