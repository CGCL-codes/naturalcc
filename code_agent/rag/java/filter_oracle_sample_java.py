#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


MARKER = '// Static import definitions from sources:'


def load_jsonl_map(path: Path) -> dict[int, dict]:
    items = {}
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = obj.get('id')
            if sample_id is None:
                raise ValueError(f'{path}:{line_no} 缺少 id 字段')
            items[sample_id] = obj
    return items


def load_id_order(path: Path) -> list[int]:
    ids = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = obj.get('id')
            if sample_id is None:
                raise ValueError(f'{path}:{line_no} 缺少 id 字段')
            ids.append(sample_id)
    return ids


def has_static_import_supplement(prompt: str) -> bool:
    return MARKER in prompt


def main():
    parser = argparse.ArgumentParser(description='按 sample prompt 的 id 过滤 Java Oracle prompt，并统计 static import 补充占比')
    parser.add_argument('--sample_ids', required=True)
    parser.add_argument('--oracle_prompt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sample_ids = load_id_order(Path(args.sample_ids))
    oracle_items = load_jsonl_map(Path(args.oracle_prompt))

    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in oracle_items]
    if missing_ids:
        preview = ', '.join(str(x) for x in missing_ids[:10])
        raise ValueError(f'有 {len(missing_ids)} 个 id 在 Oracle prompt 中不存在，例如: {preview}')

    selected_items = [oracle_items[sample_id] for sample_id in sample_ids]
    supplemented_count = sum(1 for item in selected_items if has_static_import_supplement(item.get('prompt', '')))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for item in selected_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    total_count = len(selected_items)
    ratio = supplemented_count / total_count if total_count else 0.0
    print(f'✓ 输出 sample Java Oracle prompt: {output_path}')
    print(f'  sample 总数: {total_count}')
    print(f'  抽样中补充了 static import 定义的样本数: {supplemented_count}')
    print(f'  抽样中 static import 补充占比: {ratio:.1%}')


if __name__ == '__main__':
    main()
