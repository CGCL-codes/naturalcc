# 合同指标 3：高频缺陷检测总报告

更新时间（UTC）：2026-10-29T11:32:40.095502+00:00

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
| Juliet 调优集 Web · `c-security-review-v1` | 60/60 条有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 100.00% / 46.67% | 跨函数/宏/上下文，第三方标签 |
| Juliet 调优集 Web · `c-security-review-v2` | 60/60 条有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 100.00% / 23.33% | 跨函数/宏/上下文，第三方标签 |
| Juliet 调优集 Web · `c-security-review-v3-comprehensive` | 26/60 条有效评分；OpenRouter 402：在途请求额度占用，余额未必耗尽；已停止，待在途请求结算后可重试 | 100.00% / 0.00% | 跨函数/宏/上下文，第三方标签 |
| Juliet 调优集 Web · `c-security-review-v3-exam-baseline` | 60/60 条有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 96.67% / 13.33% | 出题基线（故意削弱，供学生优化） |
| Juliet 调优集 Web · `c-security-review-v3-exam-baseline-r2` | 59/60 条有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 89.66% / 10.00% | 出题基线（故意削弱，供学生优化） |
| 平衡验收集（加难版） Web | 9/9 组有效评分；样本队列已处理完毕（未评分项保留在 checkpoint） | 99.44% / 0.00% | 加难验收链路 |

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

## Juliet 第三方测试集（跨函数、宏、依赖上下文）

前述 RTOS 平衡/压力集与加难版都是本项目脚本生成的 fixture：样例形态由我们决定，分数可以被样例设计左右，因此不宜单独作为检测能力证据。本节参照合同 5 的做法改用第三方冻结数据集与标签：

- 来源：[NIST SARD Juliet Test Suite for C/C++ v1.3](https://samate.nist.gov/SARD/test-suites/112)，zip SHA256 `ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb`；标签取自 Juliet 自带的 `OMITBAD`/`OMITGOOD` 划分：bad 路径为有缺陷样本，good 路径（goodG2B/goodB2G/good1 等）为安全样本。
- 范围：数组越界取 CWE-121 的 CWE129 下标与 CWE805 int 循环族；字符串溢出取 CWE-121 的 char 型 strcpy/strcat/strncpy/strncat/snprintf 族；空指针调用取 CWE-476 C 族。只纳入依赖上下文的流变体：宏/全局常量/全局函数控制流（09–14）、跨函数静态/全局标志（21–22）、跨函数传参/返回/函数指针/静态全局（41–45）、跨文件数据流（51–68）。
- 选样：每类按 `sha256("contract3-juliet-v1:<测试用例名>")` 排序取前 10 个测试用例，每个用例生成 1 个缺陷样本 + 1 个安全样本；选样在任何模型运行前冻结，不按结果筛选或剔除难例。
- 去标签：删除全部注释，解析掉 OMITBAD/OMITGOOD/INCLUDEMAIN 块，把含 CWE/bad/good/G2B/B2G 的标识符改为 `sym_NN`，文件改名为 `src_a.c`…，并删去支持代码中的 `good1()`/`bad1()` 空桩；每个样本均与支持代码一起通过 `gcc -fsyntax-only`。已知残余线索：安全样本通常比缺陷样本多一个调用多个子函数的入口函数。
- 执行：与合同 5 相同的 Web `/api/agent/threads` → Run 链路、OpenRouter/Sonnet 4.5、无 fallback；每个样本独立 Thread 与工作区，固定次序，已评分（含错误）不重问。计分为文件级：最终 JSON 中出现目标类别 Finding 即判为有缺陷。
- 版本：`c-security-review-v1` 为未调优基线；之后的版本按调优集失败模式修改产品复核规则（不改样本、标签、提示词和计分），**因此调优集上的新版本成绩是调优集复测，不是独立成绩**。留出集按同一哈希顺序取后续用例，单独冻结，不参与调优，用于检验泛化。
- 出题基线：`*-exam-baseline` 是给学生做检测器优化题用的**故意削弱**版本（环境变量 `CODE_AGENT_REVIEW_PROFILE=exam-baseline`），在完整规则上追加两条“跳过某类检查”的范围限制（跳过 strncat、跳过与 NULL 比较的分支内解引用），并按检出率约 86%、误报率约 13% 校准。它不是产品能力成绩，也不能用于验收；完整规则（参考答案）见同名不带后缀的版本。

### Juliet 调优集 · `c-security-review-v1`

已有效评分 60/60 条；TP/FN/FP/TN：30/0/14/16；状态：样本队列已处理完毕（未评分项保留在 checkpoint）。工件：`artifacts/contract3/juliet/web-sonnet45-c-security-review-v1/`。

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 10 | 0 | 7 | 3 | 100.00% | 70.00% |
| 字符串溢出 | 10 | 0 | 3 | 7 | 100.00% | 30.00% |
| 空指针调用 | 10 | 0 | 4 | 6 | 100.00% | 40.00% |
| 总体 | 30 | 0 | 14 | 16 | 100.00% | 46.67% |

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 宏/全局常量/全局函数控制流（09–14） | 14 | 0 | 4 | 10 | 100.00% | 28.57% |
| 跨函数静态/全局标志（21–22） | 2 | 0 | 1 | 1 | 100.00% | 50.00% |
| 跨函数传参/返回/函数指针/静态全局（41–45） | 4 | 0 | 1 | 3 | 100.00% | 25.00% |
| 跨文件数据流（51–68） | 10 | 0 | 8 | 2 | 100.00% | 80.00% |

与 Juliet 标签不一致的样本（全部保留在计分中）：

- `array_oob-02-s` `CWE129_fgets_22`（误报）：在src_b.c的sym_08函数中发现1处数组越界风险：第69行对buffer[10]的访问仅检查了下界(data>=0)，未检查上界(data<10)，存在数组越界的逻辑漏洞。
- `array_oob-03-s` `CWE129_rand_63`（误报）：发现1处数组越界风险。src_b.c::sym_01()函数中buffer[data]访问缺少上界检查，当data >= 10时会越界。对比sym_03()函数有完整的边界检查(data >= 0 && data < 10)，sym_01()是有缺陷的版本。
- `array_oob-04-s` `CWE129_rand_42`（误报）：src_a.c 的 sym_02() 函数第21行存在数组越界风险。虽然当前数据流下 data=7 在有效范围内，但第19行条件判断仅检查 data >= 0，缺少上界检查 data < 10，不符合安全编程实践。对比 sym_04() 第50行的完整边界检查 'data >= 0 && data < (10)'，sy
- `array_oob-05-s` `CWE129_fgets_53`（误报）：在src_d.c第12行发现1处数组越界风险。sym_08函数对buffer[10]进行写入时，仅验证索引非负但未检查上界，存在越界写入可能。sym_09函数实现了正确的边界检查（data >= 0 && data < 10）。
- `array_oob-08-s` `CWE129_rand_09`（误报）：发现2处数组越界风险。sym_03()第93行和sym_04()第122行的数组访问操作仅进行了下界检查(data>=0)而缺少上界检查(data<10)。虽然当前数据流中data固定为7不会实际越界，但边界检查逻辑不完整存在潜在风险，不符合安全编码实践
- `array_oob-09-s` `CWE129_fscanf_09`（误报）：发现 2 处数组越界风险，均为缺少上界检查的边界验证缺陷。sym_03() 第 85-87 行和 sym_04() 第 109-111 行仅检查索引非负，未验证是否小于数组大小 10。虽然当前常量控制流使 data=7 处于安全范围，但不满足完整边界检查要求。sym_01() 和 sym_02() 的检查是正确的（da
- `array_oob-10-s` `CWE129_large_65`（误报）：发现1处数组越界漏洞：src_b.c的sym_01函数缺少数组上界检查，当data值为10时触发越界写入
- `null_pointer-03-s` `int_66`（误报）：发现1处空指针调用风险：src_b.c第9行函数sym_01()缺少NULL检查直接解引用指针参数，虽然当前调用路径安全，但作为公共接口存在设计缺陷。sym_03()函数在第15行正确实施了NULL检查保护。
- `null_pointer-04-s` `int_63`（误报）：审计发现 src_b.c 第9行存在空指针解引用风险。sym_01 函数未对接收的指针进行NULL检查即直接解引用，缺乏防御性编程，存在安全隐患。
- `null_pointer-06-s` `int_51`（误报）：在 src_b.c 的 sym_01 函数中发现1处空指针解引用风险。该函数未对指针参数进行空值检查即直接解引用，违反防御性编程原则。虽然当前代码的实际调用路径未传入空指针，但函数作为公开接口缺少必要的参数验证，存在潜在安全隐患
- `null_pointer-10-s` `null_check_after_deref_14`（误报）：src_a.c存在2处空指针调用风险：sym_01和sym_02函数均在malloc后未检查返回值即解引用指针（第15行和第31行），当内存分配失败时将导致空指针解引用崩溃。自动扫描工具未检出该问题（受覆盖限制），但通过人工审计确认风险真实存在。
- `string_overflow-03-s` `CWE806_char_declare_ncat_65`（误报）：在 src_b.c 第10行发现字符串溢出风险。strncat 的第三参数使用了 strlen(data) 而非基于目标缓冲区剩余空间计算，当源字符串长度接近或超过目标缓冲区大小时会导致溢出。虽然当前跨文件调用路径中 data 长度为49、dest容量为50可以容纳，但该代码模式在 data 长度变化时不安全。
- `string_overflow-04-s` `CWE806_char_declare_ncat_63`（误报）：发现1处字符串溢出风险：src_b.c第11行strncat使用不安全的第三参数strlen(data)，在当前49字节数据流下达到50字节边界，属于危险编码模式。
- `string_overflow-10-s` `CWE806_char_declare_snprintf_12`（误报）：src_a.c 第30行存在不安全的snprintf调用模式，使用源字符串长度而非目标缓冲区容量作为size参数，属于字符串溢出风险编程错误。当前数据流条件下未实际溢出，但编码模式危险。

### Juliet 调优集 · `c-security-review-v2`

已有效评分 60/60 条；TP/FN/FP/TN：30/0/7/23；状态：样本队列已处理完毕（未评分项保留在 checkpoint）。工件：`artifacts/contract3/juliet/web-sonnet45-c-security-review-v2/`。

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 10 | 0 | 4 | 6 | 100.00% | 40.00% |
| 字符串溢出 | 10 | 0 | 2 | 8 | 100.00% | 20.00% |
| 空指针调用 | 10 | 0 | 1 | 9 | 100.00% | 10.00% |
| 总体 | 30 | 0 | 7 | 23 | 100.00% | 23.33% |

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 宏/全局常量/全局函数控制流（09–14） | 14 | 0 | 4 | 10 | 100.00% | 28.57% |
| 跨函数静态/全局标志（21–22） | 2 | 0 | 0 | 2 | 100.00% | 0.00% |
| 跨函数传参/返回/函数指针/静态全局（41–45） | 4 | 0 | 0 | 4 | 100.00% | 0.00% |
| 跨文件数据流（51–68） | 10 | 0 | 3 | 7 | 100.00% | 30.00% |

与 Juliet 标签不一致的样本（全部保留在计分中）：

- `array_oob-03-s` `CWE129_rand_63`（误报）：发现1处数组越界风险。src_b.c 第11行 sym_01() 函数中对 buffer[10] 的访问仅验证索引非负，未检查上界，当索引 >= 10 时将写入越界内存。sym_03() 函数实现了完整边界检查（0 <= data < 10），安全。
- `array_oob-05-s` `CWE129_fgets_53`（误报）：发现1处数组越界风险：src_d.c的sym_08函数中，对大小为10的buffer数组仅进行下界检查(data>=0)而缺少上界检查，当索引data>=10时将导致越界写入。
- `array_oob-09-s` `CWE129_fscanf_09`（误报）：发现2处数组越界风险，均为缺少上界检查。sym_03()第92行和sym_04()第117行对buffer[10]的访问仅验证索引下界>=0，未验证上界<10，不符合完整边界检查要求。sym_01()和sym_02()使用完整检查(data >= 0 && data < 10)是安全的。
- `array_oob-10-s` `CWE129_large_65`（误报）：发现 1 处数组越界风险。src_b.c 的 sym_01 函数仅对数组索引进行下界检查（data >= 0），缺少上界检查（data < 10），导致当 src_a.c 传入 data=10 时发生越界访问。相比之下，sym_03 函数正确实现了完整的边界检查（data >= 0 && data < 10），可防止越
- `null_pointer-10-s` `null_check_after_deref_14`（误报）：src_a.c 存在4处空指针调用风险，分布在 sym_01() 和 sym_02() 两个函数中。两处 malloc 调用（第14行和第30行）均未检查返回值是否为 NULL，后续直接解引用（第15、18、31、34行）。当内存分配失败时，将触发空指针解引用，导致程序崩溃（CWE-476）。
- `string_overflow-01-s` `dest_char_declare_cpy_09`（误报）：扫描发现两处strcpy调用（第26行和第47行）。经静态分析：两处source均为100字节（99个字符+'\0'），目标sym_03容量为100字节；结合GLOBAL_CONST_FALSE=0和GLOBAL_CONST_TRUE=1的常量值，data在两个函数中均确定指向sym_03；strcpy复制的100字节
- `string_overflow-10-s` `CWE806_char_declare_snprintf_12`（误报）：在 src_a.c 第30行发现字符串溢出风险。SNPRINTF 使用 strlen(data) 而非 sizeof(dest) 作为大小参数，在 Windows 平台上 _snprintf 不保证空终止符，会导致目标缓冲区 dest 成为非终止字符串，存在缓冲区溢出风险。

### Juliet 调优集 · `c-security-review-v3-comprehensive`

已有效评分 26/60 条；TP/FN/FP/TN：12/0/0/14；状态：OpenRouter 402：在途请求额度占用，余额未必耗尽；已停止，待在途请求结算后可重试。工件：`artifacts/contract3/juliet/web-sonnet45-c-security-review-v3-comprehensive/`。

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 3 | 0 | 0 | 5 | 100.00% | 0.00% |
| 字符串溢出 | 5 | 0 | 0 | 5 | 100.00% | 0.00% |
| 空指针调用 | 4 | 0 | 0 | 4 | 100.00% | 0.00% |
| 总体 | 12 | 0 | 0 | 14 | 100.00% | 0.00% |

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 宏/全局常量/全局函数控制流（09–14） | 3 | 0 | 0 | 3 | 100.00% | 0.00% |
| 跨函数静态/全局标志（21–22） | 0 | 0 | 0 | 1 | N/A | 0.00% |
| 跨函数传参/返回/函数指针/静态全局（41–45） | 3 | 0 | 0 | 3 | 100.00% | 0.00% |
| 跨文件数据流（51–68） | 6 | 0 | 0 | 7 | 100.00% | 0.00% |

### Juliet 调优集 · `c-security-review-v3-exam-baseline`（出题基线，故意削弱）

已有效评分 60/60 条；TP/FN/FP/TN：29/1/4/26；状态：样本队列已处理完毕（未评分项保留在 checkpoint）。工件：`artifacts/contract3/juliet/web-sonnet45-c-security-review-v3-exam-baseline/`。

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 10 | 0 | 2 | 8 | 100.00% | 20.00% |
| 字符串溢出 | 9 | 1 | 1 | 9 | 90.00% | 10.00% |
| 空指针调用 | 10 | 0 | 1 | 9 | 100.00% | 10.00% |
| 总体 | 29 | 1 | 4 | 26 | 96.67% | 13.33% |

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 宏/全局常量/全局函数控制流（09–14） | 14 | 0 | 1 | 13 | 100.00% | 7.14% |
| 跨函数静态/全局标志（21–22） | 2 | 0 | 0 | 2 | 100.00% | 0.00% |
| 跨函数传参/返回/函数指针/静态全局（41–45） | 4 | 0 | 1 | 3 | 100.00% | 25.00% |
| 跨文件数据流（51–68） | 9 | 1 | 2 | 8 | 90.00% | 20.00% |

与 Juliet 标签不一致的样本（全部保留在计分中）：

- `array_oob-03-s` `CWE129_rand_63`（误报）：发现1处数组越界风险。src_b.c第11行sym_01函数缺少索引上界检查，当接收来自RAND32()的不受限整数时可能越界访问10元素数组。sym_03函数已正确实现边界检查。
- `array_oob-05-s` `CWE129_fgets_53`（误报）：发现1处数组越界风险：src_d.c:13的sym_08函数对10元素buffer仅验证索引下界(>=0)，缺少上界检查(<10)，当data>=10时发生写越界。sym_09函数有完整边界检查(0<=data<10)，是安全的。
- `null_pointer-10-s` `null_check_after_deref_14`（误报）：src_a.c 存在2处空指针解引用风险：sym_01()第15行和sym_02()第31行在 malloc 失败时未检查返回值直接解引用。虽然小内存分配失败概率低，但根据安全编码标准，必须检查所有动态内存分配返回值。建议在第15行和第31行之前添加 NULL 检查。
- `string_overflow-02-s` `dest_char_declare_cat_41`（误报）：src_a.c第12行存在字符串溢出风险。sym_01函数使用strcat将99字节的source追加到data（来自sym_02的100字节缓冲区sym_04），当前数据流下恰好不溢出（100字节需求=100字节容量），但代码缺乏显式边界检查，存在临界风险，属于CWE-120潜在字符串溢出漏洞
- `string_overflow-04-v` `CWE806_char_declare_ncat_63`（漏报）：扫描工具未发现字符串溢出告警。代码审查发现 src_b.c:11 存在 strncat 缓冲区溢出风险（99字节数据写入50字节缓冲区），但根据扫描报告的 scope_strncat 规则，该版本审查明确排除 strncat 调用，故不纳入 findings。建议后续版本将第11行改用 snprintf 或正确计算 s

### Juliet 调优集 · `c-security-review-v3-exam-baseline-r2`（出题基线，故意削弱）

已有效评分 59/60 条；TP/FN/FP/TN：26/3/3/27；状态：样本队列已处理完毕（未评分项保留在 checkpoint）。工件：`artifacts/contract3/juliet/web-sonnet45-c-security-review-v3-exam-baseline-r2/`。

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 10 | 0 | 1 | 9 | 100.00% | 10.00% |
| 字符串溢出 | 7 | 2 | 0 | 10 | 77.78% | 0.00% |
| 空指针调用 | 9 | 1 | 2 | 8 | 90.00% | 20.00% |
| 总体 | 26 | 3 | 3 | 27 | 89.66% | 10.00% |

| 分组 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 宏/全局常量/全局函数控制流（09–14） | 13 | 1 | 1 | 13 | 92.86% | 7.14% |
| 跨函数静态/全局标志（21–22） | 2 | 0 | 0 | 2 | 100.00% | 0.00% |
| 跨函数传参/返回/函数指针/静态全局（41–45） | 4 | 0 | 0 | 4 | 100.00% | 0.00% |
| 跨文件数据流（51–68） | 7 | 2 | 2 | 8 | 77.78% | 20.00% |

与 Juliet 标签不一致的样本（全部保留在计分中）：

- `array_oob-10-s` `CWE129_large_65`（误报）：发现 1 处数组越界风险。src_b.c 第 10 行 sym_01 函数缺少上界检查，允许 data>=10 时越界访问 buffer[10] 数组。sym_03 函数在第 29 行有完整边界检查（data>=0 && data<10），因此安全。
- `null_pointer-01-v` `deref_after_check_13`（漏报）：根据 scope_null_guarded_branch 规则，第 12 行存在空指针解引用（intPointer 在第 9 行设置为 NULL，第 10 行检测为 NULL 后，第 12 行仍解引用），但该缺陷位于 NULL 比较条件的守卫分支内，属于本次扫描的排除范围，不列入 findings。该问题为确认的空指针
- `null_pointer-06-s` `int_51`（误报）：审计发现 src_b.c 的 sym_01 函数存在潜在空指针解引用风险：该函数在第8行通过 printIntLine(*data) 直接解引用指针参数，但未进行 NULL 校验。尽管当前代码中唯一的调用路径（src_a.c:sym_02 → src_b.c:sym_01）传递的是栈变量地址（有效指针），但函数设计不安
- `null_pointer-10-s` `null_check_after_deref_14`（误报）：src_a.c 存在4处空指针解引用风险，分别位于 sym_01 函数（行14、17）和 sym_02 函数（行31、34）。两处 malloc 调用（行13、30）均未检查返回值，在内存分配失败时会导致后续指针解引用操作访问 NULL 指针。建议在 malloc 后增加 NULL 检查。
- `string_overflow-04-v` `CWE806_char_declare_ncat_63`（漏报）：静态审计完成。扫描工具未报告告警。人工代码审查发现 src_b.c 第11行存在 strncat 缓冲区溢出：dest[50] 目标缓冲区容量为50字节，但 strncat(dest, data, strlen(data)) 将写入 0+99+1=100 字节（data 包含99个'A'字符，strncat 会在末尾追
- `string_overflow-05-v` `CWE806_char_declare_ncat_54`（漏报）：扫描工具未报告候选；手工审计发现src_e.c第10行strncat存在字符串溢出（99+1字节写入50字节dest缓冲区），但根据审计规则scope_strncat，strncat调用已排除在本次报告范围外，故findings为空。其他目标文件未发现字符串溢出风险。

## 加难版测试集

加难版与原两套测试集分开计分，不替换、平均或择优合并原结果。每类每工程仍为 20 正例 + 20 对照，但对照中加入“看起来危险、实际安全”的近邻写法（守卫后访问、恰好容纳的 strcpy/strcat、按 sizeof 截断的 snprintf/strncpy/memcpy、判空后返回、重新赋值或条件返回非空的指针），正例加入差一错误（`>`/`>=` 守卫、`<=` 循环、`sizeof` 计数下标、错误的 strncpy/snprintf 长度、判空后未返回）。40 个函数按固定种子打乱顺序，不能再按位置推断标签。平衡加难版为 10 个原平衡形态 + 10 个加难形态；压力加难版为 5 个原压力形态 + 15 个加难形态。所有加难形态的正例/对照均已用 GCC ASan/UBSan 单独编译运行核对：正例触发越界、溢出或空指针错误，对照运行无报错。加难版要求每条 Finding 给出所在函数名并按函数名计分：首轮仅按行号计分时，模型自行数行的偏移随文件位置累积，在打乱顺序后把正例告警落到相邻对照上（67.22% / 21.67%），该数值反映的是行号定位误差而非误判，已归档于 `artifacts/contract3/archive-web-sonnet45-balanced-hard-line-scored/`，不作为加难版成绩。

### 平衡验收集（加难版） Web

Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。
- 状态：样本队列已处理完毕（未评分项保留在 checkpoint）。
- 已有效评分：9/9 组，360 个函数；TP/FN/FP/TN：179/1/0/180。
- 工件：`artifacts/contract3/web-sonnet45-balanced-hard/`。

| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |
|---|---|---:|---|---|
| freertos_kernel-array_oob | scored | 40 | 20/0/0/20 | `f2daa13f-20e8-429f-94f6-f10c1495135f` |
| freertos_kernel-null_pointer | scored | 40 | 20/0/0/20 | `c12e4d8d-e748-447e-be2d-c9a018781915` |
| freertos_kernel-string_overflow | scored | 40 | 20/0/0/20 | `88c26055-8d43-4b9b-b5d4-40439adfc860` |
| rt_thread-array_oob | scored | 40 | 20/0/0/20 | `359b6dc6-826d-4246-87be-b2cde462b290` |
| rt_thread-null_pointer | scored | 40 | 20/0/0/20 | `3f7e224f-91a1-4a77-962c-ed6493b54bf8` |
| rt_thread-string_overflow | scored | 40 | 20/0/0/20 | `d27445bd-6eb4-41c0-9df5-2acb30d58753` |
| zephyr-array_oob | scored | 40 | 20/0/0/20 | `63673d6a-ecb8-4067-b64a-6c5d9bf5f577` |
| zephyr-null_pointer | scored | 40 | 20/0/0/20 | `4a00ccc8-4a8a-46dc-acb4-91f5c7502f05` |
| zephyr-string_overflow | scored | 40 | 19/1/0/20 | `15aa17cb-bed3-4cb1-b042-a41754f78c12` |

| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 数组越界 | 60 | 0 | 0 | 60 | 100.00% | 0.00% |
| 字符串溢出 | 59 | 1 | 0 | 60 | 98.33% | 0.00% |
| 空指针调用 | 60 | 0 | 0 | 60 | 100.00% | 0.00% |
| 总体 | 179 | 1 | 0 | 180 | 99.44% | 0.00% |

| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |
|---|---:|---:|---:|---:|---:|---:|
| FreeRTOS Kernel | 60 | 0 | 0 | 60 | 100.00% | 0.00% |
| Zephyr | 59 | 1 | 0 | 60 | 98.33% | 0.00% |
| RT-Thread | 60 | 0 | 0 | 60 | 100.00% | 0.00% |

| 类别 | 形态 | 正例 TP/总数 | 对照 FP/总数 |
|---|---|---:|---:|
| 数组越界 | `base-0` | 9/9 | 0/9 |
| 数组越界 | `base-1` | 9/9 | 0/9 |
| 数组越界 | `base-2` | 6/6 | 0/6 |
| 数组越界 | `base-3` | 6/6 | 0/6 |
| 数组越界 | `hard-0` | 6/6 | 0/6 |
| 数组越界 | `hard-1` | 6/6 | 0/6 |
| 数组越界 | `hard-2` | 6/6 | 0/6 |
| 数组越界 | `hard-3` | 6/6 | 0/6 |
| 数组越界 | `hard-4` | 6/6 | 0/6 |
| 字符串溢出 | `base-0` | 12/12 | 0/12 |
| 字符串溢出 | `base-1` | 9/9 | 0/9 |
| 字符串溢出 | `base-3` | 9/9 | 0/9 |
| 字符串溢出 | `hard-0` | 5/6 | 0/6 |
| 字符串溢出 | `hard-1` | 6/6 | 0/6 |
| 字符串溢出 | `hard-2` | 6/6 | 0/6 |
| 字符串溢出 | `hard-3` | 6/6 | 0/6 |
| 字符串溢出 | `hard-4` | 6/6 | 0/6 |
| 空指针调用 | `base-0` | 12/12 | 0/12 |
| 空指针调用 | `base-1` | 9/9 | 0/9 |
| 空指针调用 | `base-4` | 9/9 | 0/9 |
| 空指针调用 | `hard-0` | 6/6 | 0/6 |
| 空指针调用 | `hard-1` | 6/6 | 0/6 |
| 空指针调用 | `hard-2` | 6/6 | 0/6 |
| 空指针调用 | `hard-3` | 6/6 | 0/6 |
| 空指针调用 | `hard-4` | 6/6 | 0/6 |

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
uv run python scripts/benchmark_contract3_web.py --suite balanced-hard
uv run python scripts/benchmark_contract3_web.py --suite independent-hard

# 不发模型请求，仅从已有原始工件刷新本报告
uv run python scripts/render_contract3_report.py
```

## 工件索引

- `artifacts/contract3/manifest.json`：固定上游版本、目标类别、Cppcheck ID/CWE 白名单。
- `artifacts/contract3/web-sonnet45-balanced/checkpoint.json`：当前最新 Web 评分、逐例 Ground Truth、预测和 `matched_findings`。
- `artifacts/contract3/*-cppcheck/ground_truth.json`、`result.json`：本地模式运行时生成的真值和结果；按当前清理策略不随仓库保留。
- `artifacts/contract3/web-sonnet45-balanced/runs/`：本机原始 Run events，体积较大且含运行历史，已由 `.gitignore` 排除；checkpoint 保留最近 Run ID 用于关联复核。
