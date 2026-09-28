# 合同指标 3：高频缺陷检测总报告

更新时间（UTC）：2026-09-28T06:00:50.665511+00:00

本报告是唯一的指标 3 Markdown 入口。各测试集的真值、原始结果、模型 Run 和覆盖状态保留在各自工件目录；不得择优挑选或平均不同测试集的分数。

## 判定口径

每套工程针对数组越界、字符串溢出、空指针调用各有 20 个正例和 20 个安全对照。检出率 = TP/(TP+FN)，误报率 = FP/(FP+TN)。只计目标类别且行号落在冻结函数范围内的发现。

## 当前结果总览

| 测试集 / 运行方式 | 完成状态 | 检出率 / 误报率 | 定位 |
|---|---|---|---|
| 平衡验收集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 87.78% / 0.00% | Web API 验收链路 |
| 平衡验收集本地 | 未保留 | — | 推荐本地验收口径 |
| 独立压力集 Web | 尚未运行 | — | 能力边界 Web 链路 |
| 独立压力集本地 | 未保留 | — | 能力边界对照 |

## 冻结测试来源

- FreeRTOS Kernel：`https://github.com/FreeRTOS/FreeRTOS-Kernel.git`，固定提交 `dbf70559b27d39c1fdb68dfb9a32140b6a6777a0`，扫描根目录 `.`。
- Zephyr RTOS：`https://github.com/zephyrproject-rtos/zephyr.git`，固定提交 `36940db938a8f4a1e919496793ed439850a221c2`，扫描根目录 `kernel`。
- RT-Thread：`https://github.com/RT-Thread/rt-thread.git`，固定提交 `97893c004c65760c638fd7eb571a08fc987a55e5`，扫描根目录 `src`。

测试 fixture 只写入临时上游 checkout，不修改或提交上游源码；真值 JSON 不进入 Web Agent 工作区。

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

平衡验收集 Web：**不符合**各类别“检出率 >85%、误报率 <15%”的要求。

## 测试链路与适用边界

- 本地：通过产品 `security_analysis.cppcheck_scan` 适配层调用 Cppcheck；不执行目标程序。
- Web：与合同 5 共用本机 Web Agent API、线程、Run、events 和持久化引擎；固定 OpenRouter/Sonnet 配置，无模型 fallback。
- Web 运行按固定顺序串行执行，避免并发的限流、超时重试及账单归因干扰；未完成、超时、API 故障或无效 JSON 不计为正确。
- 平衡验收集证明规定注入形态下的检测统计；独立压力集展示未覆盖形态的边界。两者均不能单独证明完整上游工程覆盖、所有路径安全或增量分析性能。

## 运行与续跑

```bash
# 推荐本地验收
uv run python scripts/benchmark_contract3_independent.py --profile balanced --workspace /tmp/naturalcc-contract3-independent-sources

# Web 模式：先按 README 启动 agent_web_api.py，再在另一终端执行
uv run python scripts/benchmark_contract3_web.py --suite balanced
uv run python scripts/benchmark_contract3_web.py --suite independent

# 不发模型请求，仅从已有原始工件刷新本报告
uv run python scripts/render_contract3_report.py
```

## 工件索引

- `artifacts/contract3/manifest.json`：固定上游版本与类别映射。
- `artifacts/contract3/balanced-cppcheck/`：推荐验收集真值及原始结果。
- `artifacts/contract3/independent-cppcheck/`：独立压力集真值及原始结果。
- `artifacts/contract3/web-sonnet45-balanced/`、`web-sonnet45-independent/`：Web checkpoint、结构化预测与 Run 证据。
