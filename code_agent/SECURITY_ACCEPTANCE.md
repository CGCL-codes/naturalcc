# 指标3、指标5联调说明

这次补齐了七类分类、基于真值的单次任务统计、C/C++ 命令执行与共享变量竞争检查，以及可复现的测试工程。原有 `/api/run` 和 Agent 工具都可以使用。

## 环境与调用

后端安装项目依赖、Cppcheck 和 Clang。此次实测版本为 Cppcheck 2.13.0、Clang 18.1.3、Linux x86_64。不需要大模型 API Key，也不需要运行待测工程。Ubuntu 可安装 `cppcheck clang-18`；其他发行版安装对应软件包，确保服务进程的 `PATH` 可以找到 `cppcheck` 和 `clang`。

`feature=vulnerability_detection` 的 `feature_config` 示例：

```json
{
  "analyzer": "comprehensive",
  "scan_type": "high_risk",
  "ground_truth_file": "ground_truth.json",
  "scan_scope": "targets",
  "rule_profile": "c_cpp",
  "severity_threshold": "medium",
  "max_findings": 1000,
  "auto_fix": false
}
```

请求顶层的 `project_dir` 是后端本机工程绝对路径，`target_files` 必须包含该 `ground_truth.json` 标注的全部源文件。下方检查脚本会自动填写，无需手工复制几十个文件名。指标3改用 `scan_type=frequent_defects`。没有标注的真实项目不传 `ground_truth_file`，仍返回分类后的发现项与覆盖情况，统计状态为 `not_evaluated`，不会编造检出率。

`analyzer=auto` 保持原来的“内置规则 + 可用的 Cppcheck”；`builtin` 不启动外部分析器；`cppcheck` 要求安装 Cppcheck；新增 `comprehensive` 要求 Cppcheck 和 Clang。Clang 使用 `core/unix/cplusplus` 加两个实验性越界规则 `alpha.security.ArrayBoundV2`、`alpha.unix.cstring.OutOfBounds`，用于补充指针偏移、拼接和内存复制边界检查。实验规则可能有误报，已在 `coverage.experimental_checkers` 和相关 Finding 的 `experimental` 中标出。具体检查器见 [Clang 官方说明](https://clang.llvm.org/docs/analyzer/checkers.html)。

Agent 模式用 `vulnerability_detection.analyze`，参数可加 `analyzer=comprehensive` 和上述统计参数，仍需当前 Run 的 execute 审批。普通 `vulnerability_detection` 工具只做内置检查；两者都不会自动编译运行待测程序。

## 返回字段与分类

读取 `/api/run` 的 NDJSON 最后一条 `type=done` 事件中的 `artifacts`；Agent 工具返回对应的 `data` 字段。平台后端需透传参数及这些字段。

| 字段 | 含义 |
| --- | --- |
| `findings[].rule_id` | 原始 CWE/规则编码，保留兼容 |
| `findings[].category` / `category_label` | 稳定类别 ID / 中文名称 |
| `findings[].metric_eligible` | 该规则结果有可用于用例匹配的证据；不是人工确认标记 |
| `findings[].classification_basis` | 分类依据 |
| `findings[].analyzer` / `analyzer_id` | 具体分析器与检查规则 |
| `finding_summary` | 阈值筛选后的完整候选数、实际返回数、是否截断 |
| `contract_statistics.category_counts` | 当前指标各类别候选数、可匹配发现项数；不是 TP |
| `contract_statistics.overall` / `by_category` | 总体 / 各类别的 TP、FN、FP、TN 和比率 |
| `contract_statistics.cases` | 每个用例的匹配记录与计分结果 |
| `contract_statistics.unscored_target_finding_count` | 目标类中位于标注范围外的发现项数，不偷偷算成 TP/TN |
| `coverage` / `contract_statistics.coverage_incomplete` | 引擎状态、失败文件、覆盖限制 |

七类采用互斥的统计口径：

| 扫描类型 | 类别 ID | 中文名称 | 主要证据 / 常见 CWE |
| --- | --- | --- | --- |
| frequent_defects | array_oob | 数组越界 | 下标或指针访问越界诊断；CWE-119/125/787/788 等 |
| frequent_defects | string_overflow | 字符串溢出 | 字符串复制、拼接、格式化操作的越界诊断；CWE-120/787 等 |
| frequent_defects | null_pointer | 空指针调用 | 空指针解引用诊断；CWE-476 |
| high_risk | buffer_overflow | 缓冲区溢出 | `memcpy/memmove/memset/read/recv/fread` 等原始字节操作的越界诊断；CWE-119/787 等 |
| high_risk | data_race | 多线程竞争 | 重叠线程共享变量的冲突访问，或 TSan race 报告；CWE-362 |
| high_risk | memory_leak | 内存泄漏 | 丢失分配对象的诊断；CWE-401 |
| high_risk | command_execution | 命令执行漏洞 | 输入字符串传递到 `system/popen/_popen`；CWE-78 |

CWE 表示缺陷体系，不是一对一的业务类别。实际映射版本为 `contract-categories-v1`，按分析器规则、诊断类型和报错位置的操作确定。不能仅见到 CWE-787 就同时算数组、字符串和缓冲区三次。无法明确分类的项保留为其他诊断。

CWE-398、CWE-563 等质量问题及非当前指标的类别不参与统计。只凭 `strcpy` 等 API 名称产生的风险提示仍可展示，但 `metric_eligible=false`，不拿安全的常量复制去凑 TP/FP。Clang 的 `unix.Malloc` 也可能报告重复释放、释放后使用，只有类型为 `Memory leak` 才归入内存泄漏。

## 统一计算口径

统计单位是一个已标注用例，以“规范化的相对文件路径 + 类别 + 行范围”匹配。相同用例被多个分析器命中，只计一次。正例命中为 TP，正例未命中为 FN，安全对照命中为 FP，安全对照未命中为 TN。

| 字段 | 公式 |
| --- | --- |
| `detection_rate`，检出率 | TP / (TP + FN) |
| `warning_accuracy`，预警准确率（precision） | TP / (TP + FP) |
| `false_positive_rate`，误报率（FPR） | FP / (FP + TN) |
| `false_discovery_rate`，告警中错误占比 | FP / (TP + FP) |
| `accuracy`，整体分类准确率 | (TP + TN) / 全部标注用例数 |

比率返回 0～1 的数值，分母为零返回 `null`。总体按用例累加 TP/FN/FP/TN 后计算，不平均各类别百分比。无安全对照就没有 FPR；预警准确率与整体准确率是两个不同字段。

先使用所有阈值筛选后的候选做统计，再按 `max_findings` 截断展示列表，增量缓存也遵循这个规则。阈值本身会影响哪些告警被视为检出，因此对比实验必须保持阈值和分析器配置一致。没有“指标是否通过”字段，前端自行判断阈值。

真值文件最小结构如下；`source_sha256` 必须对应当前源文件，修改源码后旧真值会被拒绝，防止错用行号：

```json
{
  "schema_version": 1,
  "scan_type": "high_risk",
  "source_sha256": {"sample.c": "完整文件的 SHA-256"},
  "cases": [
    {"case_id": "cmd-01", "category": "command_execution", "file": "sample.c",
     "function": "invoke", "line_start": 3, "line_end": 9, "expected": "defect"},
    {"case_id": "cmd-02", "category": "command_execution", "file": "sample.c",
     "function": "invoke_fixed", "line_start": 11, "line_end": 17, "expected": "control"}
  ]
}
```

同文件同类别的标注范围不能重叠；`case_id` 不可重复；路径不可逃出工程。真值只在扫描完成后交给计分器，检测规则不读取标签。必需分析器未运行、标注文件解析失败或源码哈希不符时，不生成有效检出率。未标注区域的告警单独报告，不代表已验证整个项目。

## 测试工程与复现

在 `code_agent/` 下运行：

```bash
uv run python scripts/prepare_security_acceptance.py
uv run python scripts/check_security_acceptance.py
```

准备脚本下载并校验固定上游提交，生成 `artifacts/security-acceptance/workspaces/`：

| 目录 | 源码版本 | 样例 |
| --- | --- | --- |
| freertos_kernel | FreeRTOS Kernel V11.1.0，`dbf70559b27d39c1fdb68dfb9a32140b6a6777a0` | 指标3每类20处缺陷 + 每类20处安全对照 |
| zephyr | Zephyr v3.7.0，`36940db938a8f4a1e919496793ed439850a221c2` | 同上 |
| rt_thread | RT-Thread v5.2.1，`97893c004c65760c638fd7eb571a08fc987a55e5` | 同上 |
| high_risk | 上述 FreeRTOS 源码副本 | 指标5每类10处缺陷 + 每类10处安全对照，含 C 和 C++ |

每套目录含 `naturalcc_cases/` 和 `ground_truth.json`，总计 220 个缺陷用例、220 个安全对照。三套指标3使用相同的固定注入形态，不能当作三份独立泛化样本。测试文件单独放置，不修改上游业务源码，也不加入上游构建。

`suite.json` 记录上游版本、扫描目录和工程规模。此次三套上游源码分别计得 **30,090 / 212,624 / 190,571** 个语句节点，均超过一万。计数使用 Tree-sitter 的语句节点，排除空行、注释、声明和块包裹节点，保留不同预处理分支；同时记录解析有误的文件数。这是公开、可复查的源码规模口径，不能替代指定硬件配置下的编译统计。

检查脚本默认只扫描标注测试文件，以验证接口、分类和指标；结果写入 `artifacts/security-acceptance/result.json`。添加 `--whole-project` 才扩展到 OS 源码，缺头文件、条件编译和超时等会明确反映在覆盖状态中。没有编译数据库的默认配置不保证完整解析整个 OS 工程。

通过实际 Web 接口复现：

```bash
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
# 在另一终端执行，服务与脚本应能访问同一份工程路径：
uv run python scripts/check_security_acceptance.py --api-url http://127.0.0.1:7860
```

也可把随附 `security-test-projects.tar.gz` 解压到后端主机，通过 `--workspace /解压目录/workspaces` 指定位置；脚本按当前位置构造绝对路径。不需要重新下载源码。离线仅测接口时，可用 `prepare_security_acceptance.py --fixtures-only --output /tmp/security-smoke`，这一路径不包含 OS 源码，不宣称满足工程规模要求。

本次固定样例在综合扫描下：指标3每工程 TP=60、FN=0、FP=0、TN=60；指标5 TP=40、FN=0、FP=0、TN=40。具体规则、行号、耗时和匹配记录见结果 JSON。这是联调回归结果；检查规则开发时已经看过这些形态，不能当作独立测评，也不自动构成双方最终验收结论。已有 `合同-3-result.md`、`合同-5-result.md` 的历史实验保持独立，不合并或择优计分。

已同步 `feb0a67` 的合同3 Web 基准：该提交保存的平衡集检出率/误报率为 87.22%/0%，独立压力集为 89.44%/2.22%。这条链路由模型审计，固定传 `scan_type=frequent_defects`，不把真值提供给扫描工具，而由脚本在模型回答后匹配。工具返回 `not_evaluated` 表示没有进行工具内真值计分，不代表本轮模型基准失败。其统计和上面的综合扫描实测分开保存；同步代码与重生成报告不会重新发起模型测评。

## 多线程与 TSan

默认静态竞争检查识别：同一文件中，通过 `pthread_create` 或 `std::thread` 启动的重叠线程，对同一普通全局标量的读写/写写冲突，并检查共同 mutex。已 join 的线程、只读访问、已识别的原子变量和共同锁对照不报同类竞争。复杂指针别名、结构体共享对象、跨文件线程入口、自定义锁、宏和复杂条件路径仍需进一步检查。

TSan 是可选动态证据，后端不会自动运行用户项目。要验证附带的竞争样例，可在支持 TSan 的 Linux 环境执行：

```bash
cd artifacts/security-acceptance/workspaces/high_risk
gcc -fsanitize=thread -g -O1 -pthread -DRUN_CASE \
  "$PWD/naturalcc_cases/case_0021.c" -o /tmp/naturalcc-race
/tmp/naturalcc-race 2>tsan.log
```

随后传 `sanitizer_report=tsan.log`，并保持源文件路径与生成日志时一致。检测到竞争时程序可能以非零状态退出，需看日志中的 `WARNING: ThreadSanitizer: data race`。`case_0031.c` 是对应加锁对照，可用相同方式编译。

当前验证主机上实测 TSan 报 `FATAL: ThreadSanitizer: unexpected memory mapping`，因此没有声称动态验证通过。此类日志现在返回 `coverage.status=failed`，不会当成“零竞争”。需要在兼容的主机/虚拟机重跑动态验证；上述静态检查和接口联调不依赖 TSan 成功启动。TSan 的编译参数及平台支持见 [LLVM 官方文档](https://clang.llvm.org/docs/ThreadSanitizer.html)。
