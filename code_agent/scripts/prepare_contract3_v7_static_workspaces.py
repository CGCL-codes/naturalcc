"""Prepare hash-checked teacher workspaces and concrete static /api/run requests.

The canonical student projects keep GT outside the source tree. This separate
copy is for static GT evaluation, whose loader requires workspace-relative GT.
No model, active backend or network request is invoked by this exporter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts import build_contract3_engineering_v7 as build
from code_agent.security_contracts import load_ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=[*build.SUITES, 'all'], default='balanced')
    parser.add_argument('--output', type=Path, default=build.DATA / 'web-static')
    args = parser.parse_args()
    suites = build.SUITES if args.suite == 'all' else [args.suite]
    for suite in suites:
        parent = args.output.resolve() / suite
        parent.mkdir(parents=True, exist_ok=True)
        for project in ('freertos_kernel', 'zephyr', 'rt_thread'):
            snapshot = build.DATA / suite / project
            truth = build.verify(snapshot)
            workspace = parent / project
            if not workspace.exists():
                with tempfile.TemporaryDirectory(prefix='static-workspace-', dir=parent) as temporary:
                    staged = Path(temporary) / 'project'
                    shutil.copytree(snapshot / 'project', staged)
                    shutil.copyfile(snapshot / 'ground_truth.json', staged / 'ground_truth.json')
                    staged.rename(workspace)
            # Never overwrite a learner's edits on repeat runs. Existing copies
            # must still exactly match every source/header/configuration hash.
            manifest = build.read_json(snapshot / 'manifest.json')
            for row in manifest['project_files']:
                build.require(build.digest((workspace / row['file']).read_bytes()) == row['sha256'],
                              f'Existing workspace differs: {workspace / row["file"]}')
            build.require((workspace / 'ground_truth.json').read_bytes() == (snapshot / 'ground_truth.json').read_bytes(),
                          f'Existing GT differs: {workspace}')
            targets = [r['path'] for r in truth['detection_files']]
            load_ground_truth(workspace, 'ground_truth.json', 'frequent_defects', [workspace / f for f in targets])
            request = dict(project_dir=str(workspace), feature='vulnerability_detection', target_files=targets,
                feature_config=dict(detection_mode='static', analyzer='comprehensive', rule_profile='c_cpp',
                    scan_scope='targets', scan_type='frequent_defects', ground_truth_file='ground_truth.json',
                    auto_fix=False, max_findings=100, severity_threshold='medium', analysis_timeout_seconds='600'))
            build.write_json(parent / f'{project}-request.json', request)
            print(f'{suite}/{project}: {workspace} (120 GT cases; static evaluation only)', flush=True)


if __name__ == '__main__':
    main()
