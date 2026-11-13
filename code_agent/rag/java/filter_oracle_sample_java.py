#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


MARKER = '// Static import definitions from sources:'


# [maint-note:40eb1a123840]
# 维护导航：load_jsonl_map；下列为本函数源码中的接口与路径索引。
# 输入 path：Path。
# 返回契约：dict[int, dict]。
# 分支条件：not line。
# 分支条件：sample_id is None。
# 遍历边界：(line_no, line) ← enumerate(f, start=1)。
# 返回路径：items。
# 显式抛出：ValueError(f'{path}:{line_no} 缺少 id 字段')。
# 写入目标：items[sample_id]。
# 直接调用：path.open。
# 直接调用：enumerate。
# 直接调用：line.strip。
# 直接调用：json.loads。
# 直接调用：obj.get。
# 直接调用：ValueError。
# 资源作用域：path.open('r', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:40eb1a123840]
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


# [maint-note:bfc7f033529d]
# 维护导航：load_id_order；下列为本函数源码中的接口与路径索引。
# 输入 path：Path。
# 返回契约：list[int]。
# 分支条件：not line。
# 分支条件：sample_id is None。
# 遍历边界：(line_no, line) ← enumerate(f, start=1)。
# 返回路径：ids。
# 显式抛出：ValueError(f'{path}:{line_no} 缺少 id 字段')。
# 直接调用：path.open。
# 直接调用：enumerate。
# 直接调用：line.strip。
# 直接调用：json.loads。
# 直接调用：obj.get。
# 直接调用：ValueError。
# 直接调用：ids.append。
# 资源作用域：path.open('r', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:bfc7f033529d]
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


# [maint-note:a6949b146b0a]
# 维护导航：has_static_import_supplement；下列为本函数源码中的接口与路径索引。
# 输入 prompt：str。
# 返回契约：bool。
# 返回路径：MARKER in prompt。
# [/maint-note:a6949b146b0a]
def has_static_import_supplement(prompt: str) -> bool:
    return MARKER in prompt


# [maint-note:ea991d5cc0b8]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 分支条件：missing_ids。
# 遍历边界：item ← selected_items。
# 显式抛出：ValueError(f'有 {len(missing_ids)} 个 id 在 Oracle prompt 中不存在，例如: {preview}')。
# 直接调用：argparse.ArgumentParser。
# 直接调用：parser.add_argument。
# 直接调用：parser.parse_args。
# 直接调用：load_id_order。
# 直接调用：Path。
# 直接调用：load_jsonl_map。
# 直接调用：', '.join。
# 直接调用：str。
# 直接调用：ValueError。
# 直接调用：len。
# 直接调用：sum。
# 直接调用：has_static_import_supplement。
# 直接调用：item.get。
# 直接调用：output_path.parent.mkdir。
# 直接调用：output_path.open。
# 直接调用：f.write。
# 直接调用：json.dumps。
# 直接调用：print。
# 集合派生：sample_id 来自 sample_ids；过滤 sample_id not in oracle_items。
# 集合派生：x 来自 missing_ids[:10]。
# 集合派生：sample_id 来自 sample_ids。
# 集合派生：item 来自 selected_items；过滤 has_static_import_supplement(item.get('prompt', ''))。
# 资源作用域：output_path.open('w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 表达式分流：total_count；真值与假值分支分别计算，不同时执行。
# [/maint-note:ea991d5cc0b8]
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
