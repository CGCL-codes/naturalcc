# 合同三 v6：Clang 问题复核

核查版本：`engineering-v6-20261114`，正式代码提交 `09096e72653408aaf55694aa3959a4413a9d8de4`。

**刚补足每套 60 个缺陷的正式 v6 仍有甲方反馈的 Clang 问题。** 缺陷数量修正和编译配置修复是两件事，不能从静态比例合格推断 Clang 已完整覆盖。

## 正式 v6 的实际结果

- 每工程、每套、每类 20 个缺陷 + 20 个对照，总计 60 个缺陷 + 60 个对照。重新运行原版检查，36/36 类别比例通过，1,440/1,440 标注操作可计分。
- Zephyr 未附完整生成配置及 compile_commands，实际上下文为 `inferred`、数据库为空、编译条目 0、直接宏 0。
- 对当前源码真实执行 Clang，`kernel/atomic_c.c` 仍有 `CONFIG_MP_MAX_NUM_CPUS` 未定义及 `atomic_get` / `atomic_ptr_get` 重定义；`kernel/banner.c` 仍有 CPU 配置未定义及 `zephyr/syscall_list.h` 缺失。
- 原始证据：[正式 v6 两文件 Clang 扫描](artifacts/contract3/clang-v6-review/original-clang-evidence.json)。标注位置可计分不等于每个引擎或完整工程编译成功。

## 配置修复候选及未采用原因

已用固定 Zephyr 上游提交真实运行 CMake/Kconfig/syscall/devicetree 生成步骤，生成 native64 和 atomic-c 两套配置；提供可搬移的分文件编译数据库及相应头文件。候选的四套 C 源码和 GT 与正式版本逐字节相同。

两个已报文件在候选中均通过真实 Clang 扫描，搬移目录回归也通过。候选 Python 回归 611 项、前端 46 项及生产构建通过，但不能据此判定满足全部合同约束：

| 检查项 | 候选实际结果 |
|---|---|
| 每工程每套缺陷数 | 仍为三类各 20 个，共 60 个 |
| 逐类教学指标 | 34/36 项通过 |
| Zephyr independent 空指针 | TP/FN/FP/TN = 20/0/1/19，检出率 **100%**，误报率 5% |
| Zephyr independent-hard 空指针 | TP/FN/FP/TN = 20/0/1/19，检出率 **100%**，误报率 5% |
| 四套完整 C 文件 Clang 语法成功数 | 113/206、114/206、113/206、114/206，仍为部分覆盖 |

上面两项从 FN 变为 TP 的都是 `zephyr-null_pointer-21`（`kernel/init.c:386`）。配置补齐后，Clang 识别了原来漏报的实际空指针操作，因此达不到“检出率 <100%”的教学要求。没有删除诊断或修改分类来人为维持旧比例。

其余语法失败涉及不同功能、板级配置或外部模块依赖；借用主机分析参数的文件明确标为 `derived-source-audit-profile`，没有冒充真实 CMake 选中条目。详细记录：[复核汇总](artifacts/contract3/clang-v6-review/review.json)、[候选逐文件语法诊断](artifacts/contract3/clang-v6-review/candidate-syntax-audit.json)。

**候选未替换正式 v6，也未作为合格修复推送。** 正式 v6 的源码、GT、manifest、规则及静态成绩均保持原版；甲方旧 ZIP 未修改。完整候选源码、原始实测矩阵及复现材料保存在本机 `/home/sub4-wy/wangchen/ncc/甲方/clang复核/v6候选/`。本次只记录核查证据，不能宣称“Clang 已全部修好、所有指标仍同时达标”。
