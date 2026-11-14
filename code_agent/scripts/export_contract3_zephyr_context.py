"""Export portable v6 analysis headers from real Zephyr CMake builds.

This exports two genuine configurations, not a claim that every source belongs
in one build. Files without a CMake entry are explicitly marked as derived
source-audit profiles. Their parse failures must remain visible to the scanner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts.build_contract3_engineering_v6 import DATA, digest, read_json, require, write_json

COMMIT = '36940db938a8f4a1e919496793ed439850a221c2'
OVERLAY = DATA / 'overlays/zephyr'


def export(source: Path, native: Path, atomic: Path, output: Path):
    source, native, atomic = (p.resolve() for p in (source, native, atomic))
    require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT,
            'Unexpected Zephyr source commit')
    profiles = {'native64': native, 'atomic-c': atomic}
    contents = {}
    # Only headers are added: no new C scan targets or teaching operations.
    prefixes = ('arch/posix', 'soc/native', 'boards/native', 'scripts/native_simulator')
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z', '--', *prefixes]).decode().split('\0')
    for file in sorted(name for name in names if name.endswith('.h')):
        # Read the pinned revision, not an untracked or locally edited header.
        contents[file] = subprocess.check_output(['git', '-C', str(source), 'show', f'{COMMIT}:{file}'])
    databases = {}
    for name, build in profiles.items():
        for path in sorted((build / 'zephyr/include/generated').rglob('*.h')):
            contents[f'build_context/{name}/{path.relative_to(build).as_posix()}'] = path.read_bytes()
        contents[f'build_context/{name}/.config'] = (build / 'zephyr/.config').read_bytes()
        rows = read_json(build / 'compile_commands.json')
        databases[name] = {Path(row['file']).resolve().relative_to(source).as_posix(): row
                          for row in rows if Path(row['file']).resolve().is_relative_to(source)}
    require('kernel/atomic_c.c' in databases['atomic-c'], 'Build does not select the C atomic implementation')

    def relative_path(value, directory, build, profile):
        path = Path(value)
        if not path.is_absolute():
            path = Path(directory) / path
        path = path.resolve()
        if path.is_relative_to(source):
            return path.relative_to(source).as_posix()
        require(path.is_relative_to(build), f'External build dependency: {path}')
        return f'build_context/{profile}/{path.relative_to(build).as_posix()}'

    commands, provenance = [], {}
    targets = read_json(DATA / 'balanced/zephyr/ground_truth.json')['detection_files']
    for target in targets:
        file = target['path']
        profile = 'atomic-c' if file == 'kernel/atomic_c.c' else 'native64'
        database, build = databases[profile], profiles[profile]
        # A source-audit fallback is not evidence of a real board build. Its
        # provenance remains explicit and any parse errors remain observable.
        row = database.get(file, databases['native64']['kernel/banner.c'])
        tokens = row.get('arguments') or shlex.split(row['command'])
        flags, index = [], 1
        # Carry only preprocessing/language flags into the portable database.
        # GCC output paths, machine code flags and warning switches do not
        # define the source context and must not leak build-host paths.
        while index < len(tokens):
            token = tokens[index]
            if token in ('-I', '-isystem', '-iquote', '-include', '-imacros'):
                flags.extend([token, relative_path(tokens[index + 1], row['directory'], build, profile)])
                index += 2
                continue
            if token.startswith('-I') and len(token) > 2:
                flags.extend(['-I', relative_path(token[2:], row['directory'], build, profile)])
            elif token.startswith(('-D', '-U', '-std=')):
                flags.append(token)
            index += 1
        commands.append(dict(directory='.', file=file, arguments=['clang', *flags, '-c', file]))
        provenance[file] = dict(profile=profile, origin='cmake-real-entry' if file in database else 'derived-source-audit-profile')
    contents['compile_commands.json'] = (json.dumps(commands, indent=2) + '\n').encode()
    contents['build_context/provenance.json'] = (json.dumps(dict(board='native_posix/native/64',
        zephyr_commit=COMMIT, files=provenance), indent=2) + '\n').encode()
    for file, raw in contents.items():
        path = output / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write_json(output.parent / 'zephyr-context-manifest.json', dict(revision='zephyr-native64-cmake-v1',
        files={file: digest(raw) for file, raw in sorted(contents.items())}))
    print(f'Exported {len(contents)} files, {len(commands)} explicitly profiled source entries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--native-build', type=Path, required=True)
    parser.add_argument('--atomic-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OVERLAY)
    args = parser.parse_args()
    export(args.source, args.native_build, args.atomic_build, args.output)
