"""Exercise real /api/run handlers with v7, independently of the active server.

The evaluation workspace contains GT for deterministic backend scoring only.
No model is invoked or given that workspace. Evidence preserves actual request
and final backend response; the process uses its own temporary runtime database.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts import build_contract3_engineering_v7 as build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', choices=['freertos_kernel', 'zephyr', 'rt_thread'], required=True)
    parser.add_argument('--suite', choices=build.SUITES, default='balanced')
    parser.add_argument('--evidence', type=Path, default=build.DATA / 'static-teaching.json')
    args = parser.parse_args()
    destination = build.DATA / args.suite / args.project
    truth = build.verify(destination)
    static = build.read_json(args.evidence)
    expected = next(r for r in static['results'] if (r['suite'], r['project']) == (args.suite, args.project))
    build.require(expected['manifest_sha256'] == build.digest((destination / 'manifest.json').read_bytes()),
                  'Static evidence does not match source')
    for name, sha in static['rule_sha256'].items():
        build.require(build.digest((ROOT / name).read_bytes()) == sha, f'Stale static code: {name}')
    out = build.DATA / 'api-verification' / args.suite / args.project
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contract3-v7-api-') as temporary:
        root = Path(temporary)
        project = root / 'project'
        shutil.copytree(destination / 'project', project)
        shutil.copyfile(destination / 'ground_truth.json', project / 'ground_truth.json')
        os.environ['CODE_AGENT_DB'] = str(root / 'runtime.db')
        from fastapi.testclient import TestClient
        from code_agent.agent_web_api import app
        results = []
        with TestClient(app) as client:
            for encoding, display_limit in [('json', 1), ('form', 100)]:
                request = dict(project_dir=str(project), feature='vulnerability_detection',
                    target_files=[r['path'] for r in truth['detection_files']],
                    feature_config=dict(detection_mode='static', analyzer='comprehensive', rule_profile='c_cpp',
                        scan_scope='targets', scan_type='frequent_defects', ground_truth_file='ground_truth.json',
                        auto_fix=False, max_findings=display_limit, severity_threshold='medium',
                        analysis_timeout_seconds='600'))
                build.write_json(out / f'{encoding}-request.json', request)
                if encoding == 'json':
                    response = client.post('/api/run', json=request)
                else:
                    response = client.post('/api/run', data={k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                                             for k, v in request.items()})
                response.raise_for_status()
                (out / f'{encoding}-events.ndjson').write_text(response.text)
                events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
                final = next(e for e in reversed(events) if e.get('type') == 'done')
                build.write_json(out / f'{encoding}-final.json', final)
                build.require(final.get('status') == 'success', f'API scan failed: {final.get("status")}')
                artifacts = final['artifacts']
                statistics = artifacts['contract_statistics']
                build.require(statistics['status'] == 'evaluated' and statistics['scored_cases'] == 120,
                              f'API scoring incomplete: {statistics.get("reason")}')
                for category in build.CATEGORIES:
                    for key in ('tp', 'fn', 'fp', 'tn'):
                        build.require(statistics['by_category'][category][key] == expected['statistics']['by_category'][category][key],
                                      f'API/static mismatch: {category}/{key}')
                results.append(dict(encoding=encoding, display_limit=display_limit,
                                    finding_summary=artifacts['finding_summary'],
                                    overall=statistics['overall'], by_category=statistics['by_category'],
                                    coverage_incomplete=statistics['coverage_incomplete']))
                print(args.project, encoding, statistics['overall'], flush=True)
        build.require(results[0]['overall'] == results[1]['overall'], 'Display cap changed official statistics')
        build.write_json(out / 'verification.json', dict(dataset_revision=build.REVISION,
            manifest_sha256=expected['manifest_sha256'], rule_sha256=static['rule_sha256'],
            api_code_sha256={name: build.digest((ROOT / name).read_bytes()) for name in (
                'agent_web_api.py', 'plugins/vulnerability_detection.py', 'plugins/dispatcher.py',
                'security_case_coverage.py', 'scripts/verify_contract3_v7_api.py')},
            scope='Current local API handlers and real analyzers; not customer server acceptance.',
            passed=True, results=results))


if __name__ == '__main__':
    main()
