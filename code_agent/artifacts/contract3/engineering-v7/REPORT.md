# Contract-3 v7：工程内多机制测试与分引擎实测

版本：`engineering-v7-20261118-r4`。固定源码、外置 GT、manifest 和规则哈希绑定，三工程四套共 1,440 个标注操作。

每工程每套每类 20 个缺陷 + 20 个安全对照；每类 10 种机制，各有 2 个缺陷和 2 个对照。每套的 60 个缺陷分布在 60 个原工程业务函数中，扫描完整 C 文件。四套共享上游与机制，属于关联组合，不是独立泛化集。

旧的 `contract3_*_web_*.c` 是历史独立 fixture；v7 的修改位于各自 `project/` 下的原业务 `.c` 文件。字符串 staging 题把原格式化/拷贝操作的输入和输出接入局部缓冲区，安全性标签仅约束标注操作，不保证改写前后的业务输出完全相同。

hard 套的第二个源长度复制机制，以及 RT-Thread hard 套的第二个下标赋值机制，增加了库调用返回值参与计算的变体。`strspn("", "x")` 和 `strspn("x", "y")` 的真实返回值均为零；这保留原有边界错误和安全对照，用于教学难度校准。不得把校准后的比例当作独立泛化性能。

FreeRTOS 仅比较指针的宏不会解引用首参；对应正例改为越过 one-past 地址边界。直接字符串写入的对照显式检查容量至少为 2。Zephyr ext2 分文件配置补入固定版本 Kconfig 的真实默认值，来源和哈希见工程内的 `build_context/v7-ext2-provenance.json`。

## 计分与覆盖

所有 TP/FN/FP/TN 由 NaturalCC 后端 `evaluate_contract` 生成，按文件、类别、标注行范围匹配 Case。不从 Finding 数量计算率，不按单个引擎结果删减分母。各引擎的 findings 独立运行、独立过滤，同一个 Case 多条告警只记一次。

测量沿用当前 Contract-3 教学开关，关闭项为 `A/B/C/E/F/G/H/I`。这是教学基线，完整参考规则需要另行运行 `--reference`。

本次也扩展了内置规则的局部值、别名、分支和循环分析，因此新旧检出率差异同时包含题集与规则变化。不能据此声称仅靠更换题集就提升了性能，也不强行复现历史约 50%/90% 的差距。扩展前源码、题集和三引擎结果保存在 [原始基线归档](../baselines/engineering-v7-pre-flow-20261117.tar.gz)，附 SHA256。

Cppcheck 遇到不支持的宏参数条件指令时会使用 Clang 的 `-E` 预处理，保留原始 `#line`；该分项诊断仍全部来自 Cppcheck，Clang 静态分析器的诊断在 Clang 分项单独统计。

下表是固定 20 个正例的端到端检出量：未完成分析的正例也在未检出数内，因此不能把这些数字称为成功编译路径上的纯检测 recall。各引擎失败文件单列；语法树覆盖 120 个操作不等于 Cppcheck/Clang 覆盖所有条件编译路径。

| 套题 | 工程 | 类别 | 内置 TP/20 | Cppcheck TP/20 | Clang TP/20 | 合并 TP/20 | 合并 FP/20 | 合同范围 |
|---|---|---|---:|---:|---:|---:|---:|---|
| balanced | freertos_kernel | 数组越界 | 19 | 8 | 9 | 19 | 2 | 通过 |
| balanced | freertos_kernel | 字符串溢出 | 19 | 7 | 2 | 19 | 2 | 通过 |
| balanced | freertos_kernel | 空指针 | 18 | 8 | 12 | 18 | 2 | 通过 |
| balanced | zephyr | 数组越界 | 16 | 5 | 8 | 17 | 2 | 通过 |
| balanced | zephyr | 字符串溢出 | 17 | 6 | 7 | 18 | 3 | 通过 |
| balanced | zephyr | 空指针 | 17 | 5 | 4 | 18 | 2 | 通过 |
| balanced | rt_thread | 数组越界 | 19 | 7 | 5 | 19 | 2 | 通过 |
| balanced | rt_thread | 字符串溢出 | 17 | 6 | 11 | 19 | 2 | 通过 |
| balanced | rt_thread | 空指针 | 18 | 3 | 1 | 18 | 2 | 通过 |
| independent | freertos_kernel | 数组越界 | 19 | 9 | 12 | 19 | 2 | 通过 |
| independent | freertos_kernel | 字符串溢出 | 19 | 6 | 0 | 19 | 2 | 通过 |
| independent | freertos_kernel | 空指针 | 18 | 10 | 16 | 18 | 2 | 通过 |
| independent | zephyr | 数组越界 | 15 | 6 | 9 | 17 | 2 | 通过 |
| independent | zephyr | 字符串溢出 | 18 | 5 | 7 | 18 | 3 | 通过 |
| independent | zephyr | 空指针 | 17 | 6 | 7 | 18 | 2 | 通过 |
| independent | rt_thread | 数组越界 | 19 | 7 | 6 | 19 | 2 | 通过 |
| independent | rt_thread | 字符串溢出 | 18 | 6 | 9 | 19 | 2 | 通过 |
| independent | rt_thread | 空指针 | 18 | 4 | 1 | 18 | 2 | 通过 |
| balanced-hard | freertos_kernel | 数组越界 | 19 | 9 | 12 | 19 | 2 | 通过 |
| balanced-hard | freertos_kernel | 字符串溢出 | 18 | 6 | 2 | 19 | 1 | 通过 |
| balanced-hard | freertos_kernel | 空指针 | 18 | 10 | 14 | 18 | 2 | 通过 |
| balanced-hard | zephyr | 数组越界 | 13 | 9 | 11 | 17 | 2 | 通过 |
| balanced-hard | zephyr | 字符串溢出 | 19 | 3 | 4 | 19 | 2 | 通过 |
| balanced-hard | zephyr | 空指针 | 17 | 6 | 8 | 18 | 2 | 通过 |
| balanced-hard | rt_thread | 数组越界 | 19 | 11 | 8 | 19 | 2 | 通过 |
| balanced-hard | rt_thread | 字符串溢出 | 16 | 5 | 9 | 18 | 2 | 通过 |
| balanced-hard | rt_thread | 空指针 | 18 | 5 | 1 | 18 | 2 | 通过 |
| independent-hard | freertos_kernel | 数组越界 | 18 | 8 | 12 | 19 | 2 | 通过 |
| independent-hard | freertos_kernel | 字符串溢出 | 16 | 4 | 3 | 17 | 2 | 通过 |
| independent-hard | freertos_kernel | 空指针 | 18 | 12 | 17 | 18 | 2 | 通过 |
| independent-hard | zephyr | 数组越界 | 12 | 7 | 13 | 17 | 2 | 通过 |
| independent-hard | zephyr | 字符串溢出 | 18 | 5 | 5 | 18 | 2 | 通过 |
| independent-hard | zephyr | 空指针 | 18 | 5 | 7 | 18 | 2 | 通过 |
| independent-hard | rt_thread | 数组越界 | 19 | 8 | 9 | 19 | 2 | 通过 |
| independent-hard | rt_thread | 字符串溢出 | 18 | 3 | 6 | 19 | 2 | 通过 |
| independent-hard | rt_thread | 空指针 | 18 | 0 | 0 | 18 | 2 | 通过 |

逐项合同门槛（检出率 ≥85% 且 <100%，误报率 5%–15%）：**36/36 项通过**。

| 套题 | 工程 | C 文件 / 行数 | Cppcheck 失败文件 | Clang 失败文件 | Cppcheck 比内置新增正例 | Clang 比内置新增正例 |
|---|---|---:|---:|---:|---:|---:|
| balanced | freertos_kernel | 18 / 23757 | 0 | 8 | 0 | 0 |
| balanced | zephyr | 206 / 73120 | 49 | 90 | 2 | 3 |
| balanced | rt_thread | 77 / 66871 | 0 | 54 | 0 | 2 |
| independent | freertos_kernel | 18 / 23758 | 0 | 8 | 0 | 0 |
| independent | zephyr | 206 / 73121 | 49 | 89 | 2 | 3 |
| independent | rt_thread | 77 / 66869 | 0 | 54 | 0 | 1 |
| balanced-hard | freertos_kernel | 18 / 23757 | 0 | 8 | 0 | 1 |
| balanced-hard | zephyr | 206 / 73120 | 49 | 89 | 5 | 5 |
| balanced-hard | rt_thread | 77 / 66871 | 0 | 54 | 0 | 2 |
| independent-hard | freertos_kernel | 18 / 23758 | 0 | 8 | 0 | 2 |
| independent-hard | zephyr | 206 / 73120 | 49 | 89 | 4 | 4 |
| independent-hard | rt_thread | 77 / 66867 | 0 | 54 | 0 | 1 |

新增正例两列都以内置规则为基准，彼此可能重叠。逐 Case 命中及失败标记见 [CSV](case-engine-matrix.csv)，逐机制统计、告警、编译诊断和工具版本见 [原始 JSON](static-teaching.json)。

## 机制与复现

- 数组越界：`extent_index`、`negative_index`、`index_assignment`、`unsigned_underflow`、`loop_exit_index`、`pointer_alias`、`inclusive_guard`、`signed_arithmetic`、`byte_element_units`、`branch_index`。
- 空指针：`literal_null`、`alias_overwrite`、`local_null_initialization`、`wrong_pointer_guard`、`conditional_assignment`、`alias_chain`、`loop_reset`、`comma_assignment`、`conditional_pointer`、`library_return`。
- 字符串溢出：`terminator_off_by_one`、`negative_destination`、`destination_alias`、`inclusive_length_guard`、`unsigned_remaining`、`tail_copy`、`source_size_copy`、`loop_copy_boundary`、`length_assignment`、`branch_length`。

独立 ASan/UBSan 见证覆盖每种机制的正负例、两种字符串注入方式以及调用推导长度变体，共 86 项。它验证有明确输入条件的机制实现，不证明每个 RTOS 业务路径都能启动或到达。[见证记录](mechanism-witnesses.json)保留编译器、源码哈希和诊断。

```bash
export PATH=/mnt/onelab0/sub4-wy/luokaic/conda_envs/ncc-static/bin:$PATH
uv run --no-sync python scripts/build_contract3_engineering_v7.py --verify
uv run --no-sync python scripts/verify_contract3_v7_mechanisms.py
uv run --no-sync python scripts/measure_contract3_engineering.py --dataset v7 --timeout 600
uv run --no-sync python scripts/render_contract3_v7_report.py
uv run --no-sync python scripts/verify_contract3_v7_api.py --project freertos_kernel
```

以上是本地静态和 API 验证，不是客户部署验收。当前正式入口仍指向 v6；v6 的旧率属于旧题集/旧规则证据，不能用于说明 v7。

## 本地回归与 API 复核

[复核记录](validation/summary.json)及同目录日志保存后端测试、前端测试、构建、注释覆盖和验收门禁结果。[API 证据](api-verification/)分别保存三工程 balanced 快照的 JSON/表单请求与最终后端统计；展示上限 1 和 100 必须得到相同正式统计。其余九套为离线后端评分，不扩称为 API 全矩阵验收。
