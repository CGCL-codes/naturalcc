# 合同指标 3：高频缺陷检测总报告

更新时间（UTC）：2026-09-27T23:50:16.103034+00:00

本报告是唯一的指标 3 Markdown 入口。各测试集的真值、原始结果、模型 Run 和覆盖状态保留在各自工件目录；不得择优挑选或平均不同测试集的分数。

## 判定口径

每套工程针对数组越界、字符串溢出、空指针调用各有 20 个正例和 20 个安全对照。检出率 = TP/(TP+FN)，误报率 = FP/(FP+TN)。只计目标类别且行号落在冻结函数范围内的发现。

## 当前结果总览

| 测试集 / 运行方式 | 完成状态 | 检出率 / 误报率 | 定位 |
|---|---|---|---|
| 平衡验收集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 88.33% / 0.56% | Web API 验收链路 |
| 平衡验收集本地 | 已完成 | 90.00% / 0.00% | 推荐本地验收口径 |
| 独立压力集 Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 84.44% / 0.00% | 能力边界 Web 链路 |
| 独立压力集本地 | 已完成 | 66.67% / 0.00% | 能力边界对照 |

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
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：159/21/1/179。
- 工件：`artifacts/contract3/web-sonnet45-balanced/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 16/4/1/19 | `6ac6f18e-5784-4ed0-bfb3-4f40b01e9ce2` |
| freertos_kernel-null_pointer | scored | 40 | 16/4/0/20 | `83452385-be7d-4b6c-abfd-681ac5a923bc` |
| freertos_kernel-string_overflow | scored | 40 | 17/3/0/20 | `2a48756e-8042-434b-8b09-a5e7963e101e` |
| rt_thread-array_oob | scored | 40 | 18/2/0/20 | `713155cd-dd65-4fc4-9371-c2f2eb579b32` |
| rt_thread-null_pointer | scored | 40 | 16/4/0/20 | `94f70758-5330-4bac-a8ec-3a82ce75b13f` |
| rt_thread-string_overflow | scored | 40 | 20/0/0/20 | `5748c663-8898-4805-9e53-b25d6deee605` |
| zephyr-array_oob | scored | 40 | 18/2/0/20 | `ea6bbf9d-c9f4-4893-a1c1-b88d4b5b62e2` |
| zephyr-null_pointer | scored | 40 | 18/2/0/20 | `d0b5e254-b63d-4d3d-b8a9-aa76f34fc87c` |
| zephyr-string_overflow | scored | 40 | 20/0/0/20 | `5aa46e8c-0c4e-4ab9-bdf3-589ae4d76b7d` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 52 | 8 | 1 | 59 | 86.67% | 1.67% |
| 字符串溢出 | 57 | 3 | 0 | 60 | 95.00% | 0.00% |
| 空指针调用 | 50 | 10 | 0 | 60 | 83.33% | 0.00% |
| 总体 | 159 | 21 | 1 | 179 | 88.33% | 0.56% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 49 | 11 | 1 | 59 | 81.67% | 1.67% |
| Zephyr | 56 | 4 | 0 | 60 | 93.33% | 0.00% |
| RT-Thread | 54 | 6 | 0 | 60 | 90.00% | 0.00% |

平衡验收集 Web：**不符合**各类别“检出率 >85%、误报率 <15%”的要求。

### 平衡验收集本地

- FreeRTOS Kernel：`dbf70559b27d39c1fdb68dfb9a32140b6a6777a0`，186,067 行，4.16 秒，覆盖 `partial`。
- Zephyr RTOS：`36940db938a8f4a1e919496793ed439850a221c2`，17,050 行，2.62 秒，覆盖 `completed`。
- RT-Thread：`97893c004c65760c638fd7eb571a08fc987a55e5`，20,545 行，4.13 秒，覆盖 `partial`。

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 54 | 6 | 0 | 60 | 90.00% | 0.00% |
| 字符串溢出 | 54 | 6 | 0 | 60 | 90.00% | 0.00% |
| 空指针调用 | 54 | 6 | 0 | 60 | 90.00% | 0.00% |
| 总体 | 162 | 18 | 0 | 180 | 90.00% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 | 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---|
| FreeRTOS Kernel | 54 | 6 | 0 | 60 | 90.00% | 0.00% | partial |
| Zephyr RTOS | 54 | 6 | 0 | 60 | 90.00% | 0.00% | completed |
| RT-Thread | 54 | 6 | 0 | 60 | 90.00% | 0.00% | partial |

结论：**符合**本轮“检出率 >85%、误报率 <15%”的检测统计要求。
FreeRTOS 与 RT-Thread 的 `partial` 表示默认 Cppcheck 配置有工程解析诊断；注入文件统计有效，但不能据此声称完整覆盖全部上游工程。

## 独立复杂压力集（能力边界）

该集用于暴露跨辅助函数、结构体指针、指针算术和复杂字符串流的边界；不能替换、平均或择优合并平衡验收集。

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| 字符串溢出 | 36 | 24 | 0 | 60 | 60.00% | 0.00% |
| 空指针调用 | 36 | 24 | 0 | 60 | 60.00% | 0.00% |
| 总体 | 120 | 60 | 0 | 180 | 66.67% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 | 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---|
| FreeRTOS Kernel | 40 | 20 | 0 | 60 | 66.67% | 0.00% | partial |
| Zephyr RTOS | 40 | 20 | 0 | 60 | 66.67% | 0.00% | completed |
| RT-Thread | 40 | 20 | 0 | 60 | 66.67% | 0.00% | partial |

### 独立压力集 Web

Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。
- 状态：样本队列已处理完毕（未评分项保留在 checkpoint）。
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：152/28/0/180。
- 工件：`artifacts/contract3/web-sonnet45-independent/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 18/2/0/20 | `b92e7444-721a-4062-8e95-07f532a4c681` |
| freertos_kernel-null_pointer | scored | 40 | 16/4/0/20 | `c1229872-c716-477b-aa4f-e2c739f16667` |
| freertos_kernel-string_overflow | scored | 40 | 17/3/0/20 | `bbc40332-e786-4479-bd7f-b87d2e5b2fa0` |
| rt_thread-array_oob | scored | 40 | 17/3/0/20 | `ba460707-468a-440a-9e3c-b25d06f24b75` |
| rt_thread-null_pointer | scored | 40 | 15/5/0/20 | `d78fed87-6be4-4714-ad8e-1937288f9bad` |
| rt_thread-string_overflow | scored | 40 | 17/3/0/20 | `407cb0db-3dbb-443c-9d7b-ed3e39baa8b8` |
| zephyr-array_oob | scored | 40 | 18/2/0/20 | `d1c8fd0f-0bf6-426b-bb1c-03bbafaa1229` |
| zephyr-null_pointer | scored | 40 | 17/3/0/20 | `cc590984-ebc2-447c-9dde-6d6ef0bf3201` |
| zephyr-string_overflow | scored | 40 | 17/3/0/20 | `f063e778-4564-4d30-b002-28b8b6334f7e` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 53 | 7 | 0 | 60 | 88.33% | 0.00% |
| 字符串溢出 | 51 | 9 | 0 | 60 | 85.00% | 0.00% |
| 空指针调用 | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| 总体 | 152 | 28 | 0 | 180 | 84.44% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 51 | 9 | 0 | 60 | 85.00% | 0.00% |
| Zephyr | 52 | 8 | 0 | 60 | 86.67% | 0.00% |
| RT-Thread | 49 | 11 | 0 | 60 | 81.67% | 0.00% |

独立压力集 Web 的总体检出率为 84.44%，未达到各类别均 >85%、误报率 <15% 的验收口径；该集仅用于记录复杂形态下的能力边界。

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| 字符串溢出 | 36 | 24 | 0 | 60 | 60.00% | 0.00% |
| 空指针调用 | 36 | 24 | 0 | 60 | 60.00% | 0.00% |
| 总体 | 120 | 60 | 0 | 180 | 66.67% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 | 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---|
| FreeRTOS Kernel | 40 | 20 | 0 | 60 | 66.67% | 0.00% | partial |
| Zephyr RTOS | 40 | 20 | 0 | 60 | 66.67% | 0.00% | completed |
| RT-Thread | 40 | 20 | 0 | 60 | 66.67% | 0.00% | partial |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 48 | 12 | 0 | 60 | 80.00% | 0.00% |
| 字符串溢出 | 36 | 24 | 0 | 60 | 60.00% | 0.00% |
| 空指针调用 | 36 | 24 | 0 | 60 | 60.00% | 0.00% |
| 总体 | 120 | 60 | 0 | 180 | 66.67% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 | 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---|
| FreeRTOS Kernel | 40 | 20 | 0 | 60 | 66.67% | 0.00% | partial |
| Zephyr RTOS | 40 | 20 | 0 | 60 | 66.67% | 0.00% | completed |
| RT-Thread | 40 | 20 | 0 | 60 | 66.67% | 0.00% | partial |

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
