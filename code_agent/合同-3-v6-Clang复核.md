# 合同三 v6：Clang 配置修复与教学指标复核

当前为 `engineering-v6-20261114` 的 Clang 配置修订。原版提交 `09096e7` 的问题及未采用候选仍保留在 [历史核查证据](artifacts/contract3/clang-v6-review/review.json)，不用于当前成绩。

## 修复内容

四套 Zephyr 均补入真实生成的 native64 / atomic-c 配置、syscall/devicetree 头文件以及工程内相对路径 `compile_commands.json`。`atomic_c.c` 使用 C 原子实现配置，其他文件按记录的分析配置加载，避免把原子内联实现和 C 实现混用。编译数据库搬移后仍可直接识别，206 个条目，参数无丢弃；配置宏经 `-imacros autoconf.h` 加载。

真实 Clang 对四套 `kernel/atomic_c.c`、`kernel/banner.c`、`kernel/init.c` 均返回 completed，已消除甲方指出的原子函数重定义、CPU 配置宏未定义和 syscall 头文件缺失。见 [逐套实际扫描](artifacts/contract3/engineering-v6/clang-reported-files-validation.json)。

## 教学指标与实际缺陷

仅补齐配置时，Zephyr 压力集与压力加难集的空指针检出率曾达到 100%，该候选未交付。本次在这两套原有 `z_impl_device_init` 的同一个业务操作上，将零指针来源调整为 `strspn("x", "y")` 的返回值；返回值确为零，转指针后发生空指针访问。保留原有业务函数、case_id 和操作行，没有增加样例函数，也没有减少缺陷数量。

独立 sanitizer 见证确认该操作存在真实空指针访问：[原始记录](artifacts/contract3/engineering-v6/literal-span-witness.json)。完整参考规则对普通 ASCII 字面量实现 strspn 零结果推理，学生 E-off 配置保留该语义任务；规则不读取 GT、工程名、case_id 或目标比例，不过滤原始工具告警。相关生成器、规则边界和缓存版本均有代码注释，GT 与源码哈希同步更新。

| 项目 | 最新实测 |
|---|---|
| 每工程、每套缺陷数量 | 三类各 20 个，共 60 个；另有三类各 20 个安全对照 |
| 三工程 × 四套 × 三类 | 36/36 项均满足数量与比例要求 |
| 每类检出率 | 19/20 = 95%，满足 ≥85% 且 <100% |
| 每类误报率 | 1/20 或 2/20 = 5%–10%，满足 5%–15% |
| 标注操作计分 | 1,440/1,440，720 个缺陷 + 720 个对照 |
| Python / 前端回归 | 616 / 46 项通过，生产构建通过 |

实际插件另行复测三个 balanced 工程及两套 Zephyr 压力题，共五组合、600/600 操作通过，见 [插件原始记录](artifacts/contract3/engineering-v6/product-plugin-validation.json)。其余七组合仅做离线矩阵验证。

所有十二组合的三个引擎均重新执行，没有复用旧比例或旧原始结果。哈希和数量区间统一通过 `scripts/check_contract3_teaching.py` 核查。见 [最新静态原始矩阵](artifacts/contract3/engineering-v6/static-teaching.json)及 [总报告](合同-3-result.md)。这是教学校准题集的静态结果，不作为独立泛化成绩或 Web 模型成绩。

## 仍存在的解析边界

**完整 Zephyr Clang 仍是部分覆盖。** 四套 syntax-only 通过数依次为 balanced 113/206、independent 114/206、balanced-hard 113/206、independent-hard 114/206。其余文件涉及不同功能、板级配置或外部模块依赖；完整失败列表保留在 [逐文件语法审计](artifacts/contract3/engineering-v6/zephyr-clang-syntax-audit.json)。没有把 120 个标注操作可计分当成全部 C 文件 Clang 编译成功。

`build_context/provenance.json` 区分真实 CMake 条目与 `derived-source-audit-profile`，后者仅借用主机分析参数，不表示被真实板级构建选中。配置生成步骤、固定上游和复现命令见 [配置说明](artifacts/contract3/engineering-v6/context-recipes/README.md)。未重启现有后端、未运行 Web 模型、未部署到甲方机器，原有 v5 ZIP 未修改。
