"""Validate and merge independent engine runs, then publish the v7 audit only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts import build_contract3_engineering_v7 as build

PROJECTS = ("freertos_kernel", "zephyr", "rt_thread")
LABELS = dict(array_oob="数组越界", string_overflow="字符串溢出", null_pointer="空指针")


def merge(paths):
    documents = [build.read_json(path) for path in paths]
    first = documents[0]
    for doc in documents:
        for key in ("dataset_revision", "mode", "switches_off", "requested_engines", "analysis_timeout_seconds", "rule_sha256"):
            build.require(doc[key] == first[key], f"Incompatible measurement: {key}")
    build.require(first['dataset_revision'] == build.REVISION, 'Wrong revision')
    build.require(first['mode'] == 'teaching', 'Do not publish reference evidence as the teaching baseline')
    build.require(first['requested_engines'] == ['builtin', 'cppcheck', 'clang'], 'Three engines required')
    for name, expected in first['rule_sha256'].items():
        build.require(build.digest((ROOT / name).read_bytes()) == expected, f'Stale analyzer: {name}')
    rows = [row for doc in documents for row in doc['results']]
    expected = {(s, p) for s in build.SUITES for p in PROJECTS}
    build.require(len(rows) == len(expected) and {(r['suite'], r['project']) for r in rows} == expected,
                  'The full 3-project x 4-suite matrix is required')
    rows.sort(key=lambda r: (build.SUITES.index(r['suite']), PROJECTS.index(r['project'])))
    for row in rows:
        destination = build.DATA / row['suite'] / row['project']
        build.verify(destination)
        for filename in ('manifest', 'ground_truth'):
            build.require(build.digest((destination / (filename + '.json')).read_bytes()) == row[filename + '_sha256'],
                          f'Stale {filename}: {destination}')
        build.require(row['scored_cases'] == row['total_cases'] == 120 and not row['unscored_cases'],
                      'Uncovered source operation')
        build.require(row['engine_ablations']['source_covered_cases'] == 120, 'Ablation denominator differs')
    witnesses = build.read_json(build.DATA / 'mechanism-witnesses.json')
    build.require(witnesses['dataset_revision'] == build.REVISION and witnesses['passed'] == witnesses['total'] == 86,
                  'All 86 current mechanism witnesses are required')
    build.require(witnesses['generator_sha256'] == build.digest(Path(build.__file__).read_bytes()),
                  'Witnesses use a stale generator')
    build.require(witnesses['witness_script_sha256'] == build.digest((ROOT/'scripts/verify_contract3_v7_mechanisms.py').read_bytes()),
                  'Witness implementation changed')
    return {**first, 'created_at': datetime.now(timezone.utc).isoformat(),
            'input_evidence_sha256': [build.digest(path.read_bytes()) for path in paths], 'results': rows}


def render(document):
    failures = []
    lines = ["# Contract-3 v7：工程内多机制测试与分引擎实测", "",
        f"版本：`{build.REVISION}`。固定源码、外置 GT、manifest 和规则哈希绑定，三工程四套共 1,440 个标注操作。", "",
        "每工程每套每类 20 个缺陷 + 20 个安全对照；每类 10 种机制，各有 2 个缺陷和 2 个对照。"
        "每套的 60 个缺陷分布在 60 个原工程业务函数中，扫描完整 C 文件。四套共享上游与机制，属于关联组合，不是独立泛化集。", "",
        "旧的 `contract3_*_web_*.c` 是历史独立 fixture；v7 的修改位于各自 `project/` 下的原业务 `.c` 文件。"
        "字符串 staging 题把原格式化/拷贝操作的输入和输出接入局部缓冲区，安全性标签仅约束标注操作，不保证改写前后的业务输出完全相同。", "",
        "hard 套的第二个源长度复制机制，以及 RT-Thread hard 套的第二个下标赋值机制，"
        '增加了库调用返回值参与计算的变体。`strspn("", "x")` 和 `strspn("x", "y")` 的真实返回值均为零；'
        "这保留原有边界错误和安全对照，用于教学难度校准。不得把校准后的比例当作独立泛化性能。", "",
        "FreeRTOS 仅比较指针的宏不会解引用首参；对应正例改为越过 one-past 地址边界。"
        "直接字符串写入的对照显式检查容量至少为 2。Zephyr ext2 分文件配置补入固定版本 Kconfig 的真实默认值，来源和哈希见工程内的 `build_context/v7-ext2-provenance.json`。", "",
        "## 计分与覆盖", "",
        "所有 TP/FN/FP/TN 由 NaturalCC 后端 `evaluate_contract` 生成，按文件、类别、标注行范围匹配 Case。"
        "不从 Finding 数量计算率，不按单个引擎结果删减分母。各引擎的 findings 独立运行、独立过滤，同一个 Case 多条告警只记一次。", "",
        "测量沿用当前 Contract-3 教学开关，关闭项为 `" + '/'.join(document['switches_off']) + "`。"
        "这是教学基线，完整参考规则需要另行运行 `--reference`。", "",
        "本次也扩展了内置规则的局部值、别名、分支和循环分析，因此新旧检出率差异同时包含题集与规则变化。"
        "不能据此声称仅靠更换题集就提升了性能，也不强行复现历史约 50%/90% 的差距。"
        "扩展前源码、题集和三引擎结果保存在 [原始基线归档](../baselines/engineering-v7-pre-flow-20261117.tar.gz)，附 SHA256。", "",
        "Cppcheck 遇到不支持的宏参数条件指令时会使用 Clang 的 `-E` 预处理，保留原始 `#line`；"
        "该分项诊断仍全部来自 Cppcheck，Clang 静态分析器的诊断在 Clang 分项单独统计。", "",
        "下表是固定 20 个正例的端到端检出量：未完成分析的正例也在未检出数内，因此不能把这些数字称为"
        "成功编译路径上的纯检测 recall。各引擎失败文件单列；语法树覆盖 120 个操作不等于 Cppcheck/Clang 覆盖所有条件编译路径。", "",
        "| 套题 | 工程 | 类别 | 内置 TP/20 | Cppcheck TP/20 | Clang TP/20 | 合并 TP/20 | 合并 FP/20 | 合同范围 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|"]
    csv_rows = []
    for row in document['results']:
        truth = build.read_json(build.DATA/row['suite']/row['project']/'ground_truth.json')
        case_details = {c['case_id']: c for c in truth['cases']}
        for category in build.CATEGORIES:
            metric = row['statistics']['by_category'][category]
            build.require(metric['tp'] + metric['fn'] == metric['fp'] + metric['tn'] == 20, 'Contract quota')
            ok = .85 <= metric['detection_rate'] < 1 and .05 <= metric['false_positive_rate'] <= .15
            if not ok:
                failures.append(dict(suite=row['suite'], project=row['project'], category=category,
                                     detection_rate=metric['detection_rate'], false_positive_rate=metric['false_positive_rate']))
            tps = [row['engine_ablations']['statistics'][engine]['by_category'][category]['tp']
                   for engine in ('builtin', 'cppcheck', 'clang')]
            lines.append(f"| {row['suite']} | {row['project']} | {LABELS[category]} | "
                         + ' | '.join(str(v) for v in (*tps, metric['tp'], metric['fp']))
                         + f" | {'通过' if ok else '未通过'} |")
        for c in row['engine_ablations']['cases']:
            csv_rows.append(dict(suite=row['suite'], project=row['project'], **{
                k: c[k] for k in ('case_id', 'category', 'mechanism', 'expected', 'file', 'function', 'line_start', 'line_end')},
                **{k: case_details[c['case_id']][k] for k in ('defect_line', 'source_variant', 'call_derived_bound', 'address_comparison')},
                **{e + '_hit': bool(c['hit_lines'][e]) for e in ('builtin', 'cppcheck', 'clang')},
                **{e + '_file_failed': c['file_analysis_failed'][e] for e in ('builtin', 'cppcheck', 'clang')}))
    lines += ["", f"逐项合同门槛（检出率 ≥85% 且 <100%，误报率 5%–15%）：**{36-len(failures)}/36 项通过**。", "",
              "| 套题 | 工程 | C 文件 / 行数 | Cppcheck 失败文件 | Clang 失败文件 | Cppcheck 比内置新增正例 | Clang 比内置新增正例 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in document['results']:
        reports = {e['engine']: e for e in row['engines']}
        extra = row['engine_ablations']['added_defects']
        lines.append(f"| {row['suite']} | {row['project']} | {row['target_files']} / {row['target_lines']} | "
                     f"{len(reports['cppcheck'].get('failed_files', []))} | {len(reports['clang'].get('failed_files', []))} | "
                     f"{len(extra['cppcheck'])} | {len(extra['clang'])} |")
    lines += ["", "新增正例两列都以内置规则为基准，彼此可能重叠。逐 Case 命中及失败标记见 [CSV](case-engine-matrix.csv)，"
              "逐机制统计、告警、编译诊断和工具版本见 [原始 JSON](static-teaching.json)。", "",
              "## 机制与复现", ""]
    for category, mechanisms in build.FORMS.items():
        lines.append(f"- {LABELS[category]}：" + '、'.join(f'`{m}`' for m in mechanisms) + '。')
    lines += ["", "独立 ASan/UBSan 见证覆盖每种机制的正负例、两种字符串注入方式以及调用推导长度变体，共 86 项。"
              "它验证有明确输入条件的机制实现，不证明每个 RTOS 业务路径都能启动或到达。"
              "[见证记录](mechanism-witnesses.json)保留编译器、源码哈希和诊断。", "",
              "```bash", "export PATH=/mnt/onelab0/sub4-wy/luokaic/conda_envs/ncc-static/bin:$PATH",
              "uv run --no-sync python scripts/build_contract3_engineering_v7.py --verify",
              "uv run --no-sync python scripts/verify_contract3_v7_mechanisms.py",
              "uv run --no-sync python scripts/measure_contract3_engineering.py --timeout 600",
              "uv run --no-sync python scripts/render_contract3_report.py",
              "uv run --no-sync python scripts/verify_contract3_v7_api.py --project freertos_kernel", "```", "",
              "合同三测量、验收、GT 导出、报告和 Web 基准入口默认选择 v7；历史工程集须显式传入 `--dataset v6`。"
              "以上是本地静态和 API 验证，不是客户部署验收；更改脚本默认题集不会重启运行中的服务。"]
    if any(row.get('reused_evidence_sha256') for row in document['results']):
        lines += ["", "本轮使用 `--reuse-evidence` 对已保存的三引擎原始结果重新计分；"
                  "源码、GT、检测规则、引擎配置和分析预算保持一致，来源哈希记录在每个结果中。"
                  "这不是一次新的外部引擎扫描；默认入口切换的验证记录见 `validation/summary.json`。"]
    validation = build.DATA / 'validation' / 'summary.json'
    if validation.exists():
        lines += ["", "## 本地回归与 API 复核", "",
                  "[复核记录](validation/summary.json)及同目录日志保存后端测试、前端测试、构建、注释覆盖和验收门禁结果。"
                  "[API 证据](api-verification/)分别保存三工程 balanced 快照的 JSON/表单请求与最终后端统计；"
                  "展示上限 1 和 100 必须得到相同正式统计。其余九套为离线后端评分，不扩称为 API 全矩阵验收。"]
    if failures:
        lines += ["", "**v7 当前为未达合同门槛的多机制候选，不能标为验收通过。** 保留固定分母和全部真实告警；"
                  "后续需要补齐实际编译配置、处理被条件编译排除的路径，并修复已暴露的检测能力缺口。"]
    return '\n'.join(lines) + '\n', csv_rows, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, default=[build.DATA / 'static-teaching.json'])
    args = parser.parse_args(argv)
    document = merge(args.inputs)
    report, rows, failures = render(document)
    document['acceptance'] = dict(passed=not failures, passing_categories=36-len(failures),
                                  total_categories=36, failures=failures)
    build.write_json(build.DATA / 'static-teaching.json', document)
    with (build.DATA / 'case-engine-matrix.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (build.DATA / 'REPORT.md').write_text(report)
    print(f"Validated 1440 cases; category acceptance: {36-len(failures)}/36; {build.DATA / 'REPORT.md'}")


if __name__ == '__main__':
    main()
