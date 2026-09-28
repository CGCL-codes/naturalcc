# 合同指标 3：高频缺陷检测总报告

更新时间（UTC）：2026-09-28T10:32:19.483829+00:00

本报告是唯一的指标 3 Markdown 入口。各测试集的真值、原始结果、模型 Run 和覆盖状态保留在各自工件目录；不得择优挑选或平均不同测试集的分数。

## 统计口径与前后端字段契约

统计单位为单个注入函数或安全对照函数。后端/脚本只产出原始数值，不产出 `passed`、`verdict` 或阈值结论；前端若需展示达标状态，应基于下列字段和其配置阈值自行计算。

- `categories.array_oob`、`categories.string_overflow`、`categories.null_pointer`：每个对象均含 `tp`、`fn`、`fp`、`tn`、`detection_rate`、`false_positive_rate`。
- `overall`：将三类目标样本合并后的同名字段；不得与独立压力集或不同运行批次平均。
- `scored_groups`、`scored_functions`：有效评分组数和函数数；未完成、模型/API 失败、超时或无效 JSON 的组不进入 TP/FN/FP/TN。

接口现状：上述统计已由 `benchmark_contract3_web.py` 输出并写入 checkpoint/本报告；现有 `/api/agent/threads` 和 Run API 只返回单次 Agent Run，**尚未提供**可供前端直接请求的指标 3 聚合统计 HTTP 端点。若前端需要在线读取，应新增例如 `GET /api/benchmarks/contract3/latest` 的接口，并按本节字段返回数值。

定义：TP=正例被至少一个目标类别 Finding 命中；FN=正例没有目标类别 Finding；FP=安全对照被目标类别 Finding 命中；TN=安全对照没有目标类别 Finding。检出率=`TP/(TP+FN)`；误报率=`FP/(FP+TN)`。指标 3 不使用“预警准确率”字段（该字段属于指标 5）；若前端需要精度类展示，应另行明确采用 `TP/(TP+FP)`，不能把它称为检出率。

通俗示例：已注入数组越界且报告数组越界为 TP；已注入空指针调用但未报告为 FN；安全数组访问却报告数组越界为 FP；安全字符串复制且未报告字符串溢出为 TN。这里的“正/负”只针对当前测试类别和 Ground Truth，不表示文件中是否还有其他类型的问题。

## Finding 过滤规则

指标 3 只评分数组越界、字符串溢出、空指针调用三类目标 Finding。CWE-398、CWE-563 等代码质量诊断，以及目标函数范围以外的所有 Finding，可作为额外诊断保留，但绝不参与 TP/FN/FP/TN、检出率或误报率。

例如：数组越界正例只报告 CWE-563 未使用变量，不算 TP；数组安全对照只报告 CWE-398 风格问题，不算 FP。只有类别匹配且命中函数行范围的 Finding 才能改变当前指标统计。指标 5 应采用同样原则，只把缓冲区溢出、多线程竞争、内存泄漏和命令执行漏洞纳入其统计。

- Web：每个任务只接受与该任务类别相同的 `prediction.findings[].category`；Finding 的 `line` 必须落在对应真值函数的 `[line_start, line_end]` 内。类别不符的项写入 `out_of_scope_findings`，不评分；每个函数最终的匹配证据写入 `rows[].matched_findings`。
- 本地 Cppcheck：仅匹配 `manifest.json` 中该类别的 `cppcheck_ids` 或 `cwes`，再限定文件与函数行范围；原始扫描产生的其他诊断不进入 `matched_findings`。

## 当前结果总览

| 测试集 / 运行方式 | 完成状态 | 检出率 / 误报率 | 定位 |
|---|---|---|---|
| 平衡验收集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 87.78% / 0.00% | Web API 验收链路 |
| 平衡验收集本地 | 未保留 | — | 推荐本地验收口径 |
| 独立压力集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 83.33% / 0.00% | 能力边界 Web 链路 |
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
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：158/22/0/180。
- 工件：`artifacts/contract3/web-sonnet45-balanced/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 18/2/0/20 | `64ae44c1-bb48-4443-abfe-b231b624fe32` |
| freertos_kernel-null_pointer | scored | 40 | 15/5/0/20 | `2994eaed-1cd3-48cc-9a6b-cf3d5ab24968` |
| freertos_kernel-string_overflow | scored | 40 | 17/3/0/20 | `62c41ea8-c06e-4126-94b1-b80b6fea679e` |
| rt_thread-array_oob | scored | 40 | 18/2/0/20 | `33af3839-4d1d-4069-9fc6-ea616b099378` |
| rt_thread-null_pointer | scored | 40 | 16/4/0/20 | `7c29f923-b429-4643-95f3-5bdd1d8dc152` |
| rt_thread-string_overflow | scored | 40 | 20/0/0/20 | `9a0a4907-0828-4178-99fc-a3fe1f10583a` |
| zephyr-array_oob | scored | 40 | 17/3/0/20 | `cde51f8b-bc28-456c-b5c3-2b589a41f77d` |
| zephyr-null_pointer | scored | 40 | 17/3/0/20 | `93160df9-5704-4e57-b86f-46714f560491` |
| zephyr-string_overflow | scored | 40 | 20/0/0/20 | `05442bf2-804c-404b-9482-e55017ff4792` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 53 | 7 | 0 | 60 | 88.33% | 0.00% |
| 字符串溢出 | 57 | 3 | 0 | 60 | 95.00% | 0.00% |
| 空指针调用 | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| 总体 | 158 | 22 | 0 | 180 | 87.78% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 50 | 10 | 0 | 60 | 83.33% | 0.00% |
| Zephyr | 54 | 6 | 0 | 60 | 90.00% | 0.00% |
| RT-Thread | 54 | 6 | 0 | 60 | 90.00% | 0.00% |

上述表为本轮原始统计值；阈值比较由前端或验收方按其配置执行。

## 独立复杂压力集（能力边界）

该集用于暴露跨辅助函数、结构体指针、指针算术和复杂字符串流的边界；不能替换、平均或择优合并平衡验收集。

### 独立压力集 Web

Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。
- 状态：样本队列已处理完毕（未评分项保留在 checkpoint）。
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：150/30/0/180。
- 工件：`artifacts/contract3/web-sonnet45-independent/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 17/3/0/20 | `f750a591-6e4a-4500-a81a-6132b351ed09` |
| freertos_kernel-null_pointer | scored | 40 | 16/4/0/20 | `44f8a459-c772-4a7f-9e81-df1f3a2e9c24` |
| freertos_kernel-string_overflow | scored | 40 | 17/3/0/20 | `56e29ff8-8fbb-4f1f-99f9-61aad00cf727` |
| rt_thread-array_oob | scored | 40 | 18/2/0/20 | `aed74296-a9f6-4506-8fb3-8e763d828c3a` |
| rt_thread-null_pointer | scored | 40 | 16/4/0/20 | `f219b0ba-b603-465f-8a6e-c3ec34d8489e` |
| rt_thread-string_overflow | scored | 40 | 18/2/0/20 | `80b24b8a-17a2-4436-8da7-6a2bb5ac6037` |
| zephyr-array_oob | scored | 40 | 17/3/0/20 | `45ec3f51-a9c0-404c-9c41-d8380e73a730` |
| zephyr-null_pointer | scored | 40 | 16/4/0/20 | `b637f258-b4a7-45f3-b24f-c68587133366` |
| zephyr-string_overflow | scored | 40 | 15/5/0/20 | `981fca4c-b180-48b5-9c72-f6181b9924bf` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 52 | 8 | 0 | 60 | 86.67% | 0.00% |
| 字符串溢出 | 50 | 10 | 0 | 60 | 83.33% | 0.00% |
| 空指针调用 | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| 总体 | 150 | 30 | 0 | 180 | 83.33% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 50 | 10 | 0 | 60 | 83.33% | 0.00% |
| Zephyr | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| RT-Thread | 52 | 8 | 0 | 60 | 86.67% | 0.00% |

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
