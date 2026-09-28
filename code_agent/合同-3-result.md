# 合同指标 3：高频缺陷检测总报告

更新时间（UTC）：2026-09-28T11:18:21.694129+00:00

本报告是唯一的指标 3 Markdown 入口。各测试集的真值、原始结果、模型 Run 和覆盖状态保留在各自工件目录；不得择优挑选或平均不同测试集的分数。

## 统计口径与前后端字段契约

统计单位为单个注入函数或安全对照函数。后端/脚本只产出原始数值，不产出 `passed`、`verdict` 或阈值结论；前端若需展示达标状态，应基于下列字段和其配置阈值自行计算。

- `categories.array_oob`、`categories.string_overflow`、`categories.null_pointer`：每个对象均含 `tp`、`fn`、`fp`、`tn`、`detection_rate`、`false_positive_rate`。
- `overall`：将三类目标样本合并后的同名字段；不得与独立压力集或不同运行批次平均。
- `scored_groups`、`scored_functions`：有效评分组数和函数数；未完成、模型/API 失败、超时或无效 JSON 的组不进入 TP/FN/FP/TN。

接口现状：上述 Web 基准统计由 `benchmark_contract3_web.py` 写入 checkpoint/本报告，尚未提供历史 Web 基准的聚合统计 HTTP 端点。单次漏洞扫描已可通过 `/api/run` 的 `artifacts.contract_statistics` 或 Agent 扫描工具的 `data.contract_statistics` 返回统计：需传入 `scan_type` 和 `ground_truth_file`，分类字段为 `by_category`，总体为 `overall`。这是独立的单次扫描接口，不能用其结果覆盖本报告的模型基准成绩；字段和用法见 [安全检测联调说明](SECURITY_ACCEPTANCE.md)。

定义：TP=正例被至少一个目标类别 Finding 命中；FN=正例没有目标类别 Finding；FP=安全对照被目标类别 Finding 命中；TN=安全对照没有目标类别 Finding。检出率=`TP/(TP+FN)`；误报率=`FP/(FP+TN)`。本报告指标 3 表只展示检出率和误报率。单次扫描的统一结构还可返回 `warning_accuracy=TP/(TP+FP)`，供指标 5 的预警准确率展示使用，不能把它称为检出率。

通俗示例：已注入数组越界且报告数组越界为 TP；已注入空指针调用但未报告为 FN；安全数组访问却报告数组越界为 FP；安全字符串复制且未报告字符串溢出为 TN。这里的“正/负”只针对当前测试类别和 Ground Truth，不表示文件中是否还有其他类型的问题。

## Finding 过滤规则

指标 3 只评分数组越界、字符串溢出、空指针调用三类目标 Finding。CWE-398、CWE-563 等代码质量诊断，以及目标函数范围以外的所有 Finding，可作为额外诊断保留，但绝不参与 TP/FN/FP/TN、检出率或误报率。

例如：数组越界正例只报告 CWE-563 未使用变量，不算 TP；数组安全对照只报告 CWE-398 风格问题，不算 FP。只有类别匹配且命中函数行范围的 Finding 才能改变当前指标统计。指标 5 应采用同样原则，只把缓冲区溢出、多线程竞争、内存泄漏和命令执行漏洞纳入其统计。

- Web：每个任务调用 `vulnerability_detection` 时固定传入 `scan_type:"frequent_defects"`；不传 ground truth，真值始终留在评分脚本侧。工具的 `contract_statistics.status=not_evaluated` 不影响脚本按模型最终返回的 `prediction.findings` 评分。每个任务只接受与该任务类别相同的 `prediction.findings[].category`。Finding 的 `line` 必须落在对应真值函数的 `[line_start, line_end]` 内。类别不符的项写入 `out_of_scope_findings`，不评分；每个函数最终的匹配证据写入 `rows[].matched_findings`。
- 本地 Cppcheck：此历史基准按 `manifest.json` 中该类别的 `cppcheck_ids` 或 `cwes` 匹配，再限定文件与函数行范围；原始扫描产生的其他诊断不进入 `matched_findings`。新增单次扫描使用 `security_contracts.py` 的规则及源操作分类，两套映射不互换。

## 当前结果总览

| 测试集 / 运行方式 | 完成状态 | 检出率 / 误报率 | 定位 |
|---|---|---|---|
| 平衡验收集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 87.22% / 0.00% | Web API 验收链路 |
| 平衡验收集本地 | 未保留 | — | 推荐本地验收口径 |
| 独立压力集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 89.44% / 2.22% | 能力边界 Web 链路 |
| 独立压力集本地 | 未保留 | — | 能力边界对照 |

## 冻结测试来源

- FreeRTOS Kernel：`https://github.com/FreeRTOS/FreeRTOS-Kernel.git`，固定提交 `dbf70559b27d39c1fdb68dfb9a32140b6a6777a0`，扫描根目录 `.`。
- Zephyr RTOS：`https://github.com/zephyrproject-rtos/zephyr.git`，固定提交 `36940db938a8f4a1e919496793ed439850a221c2`，扫描根目录 `kernel`。
- RT-Thread：`https://github.com/RT-Thread/rt-thread.git`，固定提交 `97893c004c65760c638fd7eb571a08fc987a55e5`，扫描根目录 `src`。

测试 fixture 只写入临时上游 checkout，不修改或提交上游源码；真值 JSON 不进入 Web Agent 工作区。每套工程在运行时会验证扫描范围的 C/C++ 源码不少于 10,000 行。

## 平衡验收集（推荐验收口径）

该集混合计算下标、循环、格式化/复制和分支指针形态；每类每工程仍固定 20 正例 + 20 对照。它是推荐的本地检测指标证据，但不是独立泛化分数。

### 平衡验收集 Web

Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。
- 状态：样本队列已处理完毕（未评分项保留在 checkpoint）。
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：157/23/0/180。
- 工件：`artifacts/contract3/web-sonnet45-balanced/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 18/2/0/20 | `8d4cbc06-b55d-4b0b-a48c-42fa9c20570f` |
| freertos_kernel-null_pointer | scored | 40 | 15/5/0/20 | `91be26a3-0c1e-4f87-a0ba-bf99b044ad17` |
| freertos_kernel-string_overflow | scored | 40 | 20/0/0/20 | `457f8b5d-c472-4bc3-bcdd-e1c8ea78057d` |
| rt_thread-array_oob | scored | 40 | 18/2/0/20 | `e8253402-5f41-4fb9-9c99-aca8e2f5a166` |
| rt_thread-null_pointer | scored | 40 | 15/5/0/20 | `43d71b7c-b029-4dd6-9153-9d5bc51f9378` |
| rt_thread-string_overflow | scored | 40 | 20/0/0/20 | `3a3b1ae1-518c-43bf-bb18-bf81748f5540` |
| zephyr-array_oob | scored | 40 | 16/4/0/20 | `67b113db-dad2-44d5-b492-47debe797cb3` |
| zephyr-null_pointer | scored | 40 | 15/5/0/20 | `c3a2f049-7ec2-4180-af39-fa546d5d453a` |
| zephyr-string_overflow | scored | 40 | 20/0/0/20 | `5dc35dfd-ca60-4b09-8abc-422fd900b5ef` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 52 | 8 | 0 | 60 | 86.67% | 0.00% |
| 字符串溢出 | 60 | 0 | 0 | 60 | 100.00% | 0.00% |
| 空指针调用 | 45 | 15 | 0 | 60 | 75.00% | 0.00% |
| 总体 | 157 | 23 | 0 | 180 | 87.22% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 53 | 7 | 0 | 60 | 88.33% | 0.00% |
| Zephyr | 51 | 9 | 0 | 60 | 85.00% | 0.00% |
| RT-Thread | 53 | 7 | 0 | 60 | 88.33% | 0.00% |

上述表为本轮原始统计值；阈值比较由前端或验收方按其配置执行。

## 独立复杂压力集（能力边界）

该集用于暴露跨辅助函数、结构体指针、指针算术和复杂字符串流的边界；不能替换、平均或择优合并平衡验收集。

### 独立压力集 Web

Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。
- 状态：样本队列已处理完毕（未评分项保留在 checkpoint）。
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：161/19/4/176。
- 工件：`artifacts/contract3/web-sonnet45-independent/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 18/2/0/20 | `11dc4b9c-1e49-4047-b9e5-f0645d472dc4` |
| freertos_kernel-null_pointer | scored | 40 | 18/2/0/20 | `a77037ce-85d5-4c8d-8b64-67902d8c349f` |
| freertos_kernel-string_overflow | scored | 40 | 19/1/0/20 | `20a1b5de-b243-4d1c-b7d2-f94b090c2ae6` |
| rt_thread-array_oob | scored | 40 | 17/3/4/16 | `a55729d6-0ede-4a0e-b042-8bfae0aa09d6` |
| rt_thread-null_pointer | scored | 40 | 16/4/0/20 | `bdd78ab8-5197-46e7-8e08-b1dfe509fc2b` |
| rt_thread-string_overflow | scored | 40 | 20/0/0/20 | `af8189f8-c223-4724-9dd0-a411b7748c50` |
| zephyr-array_oob | scored | 40 | 16/4/0/20 | `e53a2fb9-dfe5-48d2-a7ba-ef6d5cce0405` |
| zephyr-null_pointer | scored | 40 | 17/3/0/20 | `d86b1582-996a-4499-840e-aeef6aeaf687` |
| zephyr-string_overflow | scored | 40 | 20/0/0/20 | `635ccc78-e755-432f-88a8-9537bab4b061` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 51 | 9 | 4 | 56 | 85.00% | 6.67% |
| 字符串溢出 | 59 | 1 | 0 | 60 | 98.33% | 0.00% |
| 空指针调用 | 51 | 9 | 0 | 60 | 85.00% | 0.00% |
| 总体 | 161 | 19 | 4 | 176 | 89.44% | 2.22% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 55 | 5 | 0 | 60 | 91.67% | 0.00% |
| Zephyr | 53 | 7 | 0 | 60 | 88.33% | 0.00% |
| RT-Thread | 53 | 7 | 4 | 56 | 88.33% | 6.67% |

独立压力集仅用于记录复杂形态下的能力边界；表中保留原始统计值，不作阈值判定。

## 测试工程与 Ground Truth 交付状态

当前可复现工程由三套固定版本的公开 RTOS 源码加运行时注入 fixture 组成：每套工程的三类各 20 个正例和 20 个同类安全对照，因此 Web 平衡验收集共 3×3=9 个任务、360 个函数。fixture、函数名、文件、行范围及预期标签由脚本固定生成，而不是由模型提供。

- Ground truth 字段：`case_id`、`project`、`category`、`expected`、`function`、`file`、`line_start`、`line_end`、`defect_line`。Web 最新一轮位于 `artifacts/contract3/web-sonnet45-balanced/checkpoint.json` 的 `samples.*.rows[]`；该对象还含 `detected` 和 `matched_findings`，可供前端逐例展示。
- 本地运行会另外生成 `artifacts/contract3/balanced-cppcheck/ground_truth.json` 与 `result.json`，字段口径相同。当前仓库按“仅保留最新 Web 输出”的清理策略没有保留它们；运行下面的本地命令即可重新生成。
- 这套冻结来源和生成器可用于本项目复现，但**不是** NaturalCC 上游或验收方正式提供的最终验收工程/ground truth 包。若最终验收必须与外部指定样本逐字节一致，仍需要验收方提供或书面确认该工程、版本、注入位置和真值清单。

## 测试链路与适用边界

- 本地：通过产品 `security_analysis.cppcheck_scan` 适配层调用 Cppcheck；不执行目标程序。
- Web：与合同 5 共用本机 Web Agent API、线程、Run、events 和持久化引擎；固定 OpenRouter/Sonnet 配置，无模型 fallback。
- Web 运行按固定顺序串行执行，避免并发的限流、超时重试及账单归因干扰；未完成、超时、API 故障或无效 JSON 不计为安全、正确或任一统计桶。
- 平衡验收集证明规定注入形态下的检测统计；独立压力集展示未覆盖形态的边界。两者均不能单独证明完整上游工程覆盖、所有路径安全或增量分析性能。
- 多线程竞争/TSan、缓冲区溢出、内存泄漏、命令执行漏洞及预警准确率均属于指标 5，不在本指标 3 的统计接口或 Ground Truth 范围内。

## 运行与续跑

```bash
# 推荐本地验收：同时生成本地 result.json 与 ground_truth.json
uv run python scripts/benchmark_contract3_independent.py --profile balanced --workspace /tmp/naturalcc-contract3-independent-sources

# Web 模式：先按 README 启动 agent_web_api.py，再在另一终端执行
uv run python scripts/benchmark_contract3_web.py --suite balanced
uv run python scripts/benchmark_contract3_web.py --suite independent

# 不发模型请求，仅从已有原始工件刷新本报告
uv run python scripts/render_contract3_report.py
```

## 工件索引

- `artifacts/contract3/manifest.json`：固定上游版本、目标类别、Cppcheck ID/CWE 白名单。
- `artifacts/contract3/web-sonnet45-balanced/checkpoint.json`：当前最新 Web 评分、逐例 Ground Truth、预测和 `matched_findings`。
- `artifacts/contract3/*-cppcheck/ground_truth.json`、`result.json`：本地模式运行时生成的真值和结果；按当前清理策略不随仓库保留。
- `artifacts/contract3/web-sonnet45-balanced/runs/`：本机原始 Run events，体积较大且含运行历史，已由 `.gitignore` 排除；checkpoint 保留最近 Run ID 用于关联复核。
