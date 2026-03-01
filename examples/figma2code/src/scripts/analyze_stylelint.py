#!/usr/bin/env python3
"""Analyze stylelint JSON output and summarize main error types.

Usage:
  python3 src/scripts/analyze_stylelint.py --input grok4_linter.json --outdir output/stylelint_analysis --topn 20

Produces:
  - output/stylelint_analysis/stylelint_summary.json
  - output/stylelint_analysis/stylelint_rule_counts.csv
  - output/stylelint_analysis/stylelint_file_counts.csv
  - output/stylelint_analysis/stylelint_severity_counts.json
  - output/stylelint_analysis/stylelint_top_warnings.csv

The script is defensive about field names so it should work with typical stylelint JSON.
"""
import argparse
import json
import os
import csv
from collections import Counter, defaultdict


def analyze(input_path, outdir, topn=20):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # stylelint JSON usually is a list of file entries
    if not isinstance(data, list):
        # try to normalize: some tools may wrap results
        try:
            data = list(data)
        except Exception:
            data = [data]

    files_scanned = len(data)
    total_warnings = 0
    rule_counter = Counter()
    severity_counter = Counter()
    file_counter = Counter()
    rule_files = defaultdict(set)
    top_warnings = []

    for entry in data:
        # Accept common keys used by stylelint: 'source' or 'filePath'
        filepath = entry.get('source') or entry.get('filePath') or entry.get('file') or entry.get('path') or '<unknown>'
        warnings = entry.get('warnings') or []
        if not isinstance(warnings, list):
            warnings = [warnings]

        num = len(warnings)
        total_warnings += num
        if num > 0:
            file_counter[filepath] += num

        for w in warnings:
            # tolerate different field names
            rule = w.get('rule') or w.get('ruleId') or w.get('ruleName') or 'unknown'
            severity = w.get('severity') or w.get('level') or 'warning'
            text = w.get('text') or w.get('message') or ''
            line = w.get('line')
            column = w.get('column') or w.get('col')

            rule_counter[rule] += 1
            severity_counter[severity] += 1
            rule_files[rule].add(filepath)
            top_warnings.append((filepath, rule, severity, text, line, column))

    os.makedirs(outdir, exist_ok=True)

    summary = {
        'files_scanned': files_scanned,
        'total_warnings': total_warnings,
        'unique_rules': len(rule_counter),
        'files_with_warnings': len(file_counter),
    }

    with open(os.path.join(outdir, 'stylelint_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # rule counts CSV
    with open(os.path.join(outdir, 'stylelint_rule_counts.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rule', 'count', 'files_affected'])
        for rule, cnt in rule_counter.most_common():
            writer.writerow([rule, cnt, len(rule_files[rule])])

    # file counts CSV
    with open(os.path.join(outdir, 'stylelint_file_counts.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'warnings'])
        for file, cnt in file_counter.most_common():
            writer.writerow([file, cnt])

    # severity counts JSON
    with open(os.path.join(outdir, 'stylelint_severity_counts.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(severity_counter), f, ensure_ascii=False, indent=2)

    # top warnings CSV (first topn by occurrence)
    top_sorted = sorted(top_warnings, key=lambda x: (x[0] or '', x[4] or 0))[:topn]
    with open(os.path.join(outdir, 'stylelint_top_warnings.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'rule', 'severity', 'text', 'line', 'column'])
        for row in top_sorted:
            writer.writerow(row)

    # print concise report
    print('\n=== Stylelint Analysis ===')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nTop rules:')
    for rule, cnt in rule_counter.most_common(20):
        print(f"{rule}: {cnt} (files: {len(rule_files[rule])})")

    print('\nTop files with warnings:')
    for file, cnt in file_counter.most_common(20):
        print(f"{file}: {cnt}")

    print(f"\nOutputs written to: {os.path.abspath(outdir)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze stylelint JSON output and summarize main error types')
    parser.add_argument('--input', '-i', required=True, help='Path to stylelint JSON file')
    parser.add_argument('--outdir', '-o', default='output/stylelint_analysis', help='Output directory for summaries')
    parser.add_argument('--topn', '-n', type=int, default=20, help='Top N items to print/save')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    analyze(args.input, args.outdir, topn=args.topn)


if __name__ == '__main__':
    main()
