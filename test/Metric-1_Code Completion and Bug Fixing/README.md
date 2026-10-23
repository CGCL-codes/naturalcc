# 指标 1：代码补全与错误修复测试指南

本文说明如何使用本目录中的 benchmark 样例测试 NaturalCC，以及如何把“运行结束”与“修复正确”区分开。**Agent 显示 `completed` 只表示本次运行结束，不代表代码正确或官方评测通过。**

## 1. Benchmark 与样例组织

本测试池选择了 4 个有一定难度、覆盖不同代码任务的 benchmark，每类暂选 10 个样例：

| 任务类型 | Benchmark | 目录 | 主要测试内容 |
|---|---|---|---|
| 代码补全 | SecRepoBench | `CodeCompletion/SecRepoBench` | 在真实仓库上下文中补全代码，并验证编译、相关单元测试和安全性 |
| 错误修复 | Defects4C | `BugFix/Defects4C` | 修复 C/C++ 缺陷，重点覆盖漏洞修复 |
| 错误修复 | SWE-bench Multilingual | `BugFix/SWE-bench-Multilingual` | 根据 issue 修复真实项目问题 |
| 代码重构 | SWE-Atlas Refactoring | `Refactoring/SWE-Atlas-Refactoring` | 在保持行为的同时改进代码结构和工程质量 |

另外，本目录按样例池的启发式复杂度分数整理了两个便于抽样的集合：

- `easy-batch/`：相对简单的 10 个样例。
- `hard-batch/`：相对困难的 10 个样例。

easy/hard 是本地抽样分档，**不是 benchmark 官方难度等级**。分数用于本样例池内的相对排序，不代表通过率。两个 batch 是复制件；正式结果应记录原始 case ID，勿把同一个 case 当作两道不同题。

### 样例文件各自的作用

- `TASK.md`：给 Agent 的任务描述。
- `workspace/`：Agent 执行任务时可读写的工作区。测试脚本会复制一份到独立的 run 目录，保留原样例。
- `case.json`：样例 ID、benchmark 和选择信息等元数据，不是测试器。
- 官方 evaluator/测试：负责判定补丁是否正确；并非每个本地 case 文件夹都带有它。

注意：测试脚本要求 case 目录中同时有 `TASK.md` 和 `workspace/`。SWE-bench Multilingual 与 SWE-Atlas 中部分样例目前只有任务卡；SWE-Atlas hard-batch 的 `ATLAS-RF-06` 也没有准备好的 workspace，不能直接运行。SecRepoBench/Defects4C 的单文件 workspace 可用于初步 Agent 试跑，但不等于完整官方评测环境。

## 2. 一次规范测试的流程

### 第一步：准备模型与 NaturalCC

1. 在 NaturalCC UI/服务端确认 API 服务已启动，默认地址为 `http://127.0.0.1:7860`。
2. 在运行 Python 的同一个 PowerShell 窗口准备 API Key。不要把密钥写进 README、命令历史或结果文件：

~~~powershell
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) { throw '当前 PowerShell 没有 DEEPSEEK_API_KEY' }
Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/health' -TimeoutSec 10
~~~

`test/naturalcc_benchmark.py` 会从当前进程读取 `DEEPSEEK_API_KEY`（或 `OPENROUTER_API_KEY`），并在创建 run 时传给 NaturalCC。模型名和 Base URL 必须与该 Key 对应的服务商配置一致；如果认证返回 401，应先解决 Key/账户认证问题再正式测。

3. 检查 UI/模型配置中的模型与上下文窗口。脚本的 `--max-input-tokens` 是**本次 run 的累计输入预算**，不是模型上下文窗口大小。实际上下文窗口以 run 的 `events.json` 中 `runtime_model_config.context_window_tokens` 为准。

### 第二步：先 dry-run 检查样例

以下命令请从仓库根目录运行。它直接检查 easy-batch 中的 D4C-10，不调用模型、不修改文件：

~~~powershell
python test/naturalcc_benchmark.py --dataset-dir 'test/Metric-1_Code Completion and Bug Fixing/easy-batch/D4C-10-7722455726bec8c53458a32851d2a87982cf0eac___gd_gd' --codegraph off --provider deepseek --model deepseek-v4-pro --base-url 'https://api.deepseek.com/v1' --dry-run
~~~

检查输出中的 case、workspace 和 targets 是否正确。对没有 `workspace/` 的任务，先准备完整 pinned 仓库和运行环境；不要只删除 `--dry-run` 就开跑。

### 第三步：执行 Agent run

单题试跑的命令如下。确认 API、模型和 workspace 正确后，再把 `--dataset-dir` 换成 `easy-batch` 或 `hard-batch`，并用 `--limit 10` 批量运行。

~~~powershell
python test/naturalcc_benchmark.py --dataset-dir 'test/Metric-1_Code Completion and Bug Fixing/easy-batch/D4C-10-7722455726bec8c53458a32851d2a87982cf0eac___gd_gd' --codegraph off --provider deepseek --model deepseek-v4-pro --base-url 'https://api.deepseek.com/v1' --max-compaction-calls 16 --max-llm-calls 30 --max-tool-calls 60 --max-input-tokens 1200000 --max-output-tokens 240000 --max-seconds 1800 --auto-approve-all
~~~

批量示例：把上面命令中的 `--dataset-dir` 改成 `'test/Metric-1_Code Completion and Bug Fixing/easy-batch'`，再加 `--limit 10`。测试 CodeGraph 开关对性能的影响时，对同一组 case 分别运行 `--codegraph off` 和 `--codegraph on`，其他模型、预算、提示词和样例保持一致。

### 重要参数

| 参数 | 含义 |
|---|---|
| `--dataset-dir` | case 文件夹，或包含多个 case 子文件夹的目录；每个 case 需要 `TASK.md` 和 `workspace/`。 |
| `--codegraph off/on` | 关闭/启用 CodeGraph。比较时应对同一 case 做成对运行。 |
| `--limit N` | 最多运行 N 个 case；省略或设为 `0` 表示全部。case 按名称排序。 |
| `--dry-run` | 只检查输入和运行计划，不请求模型、不修改文件。 |
| `--provider`、`--model`、`--base-url` | 指定模型提供商、模型名和 API 地址；必须与 Key 匹配。 |
| `--max-llm-calls` | 每个 case 的模型调用预算。 |
| `--max-compaction-calls` | 每个 case 的上下文压缩维护调用预算，不是上下文窗口大小。 |
| `--max-tool-calls` | 每个 case 的工具调用预算。 |
| `--max-input-tokens`、`--max-output-tokens` | 每个 case 的累计输入/输出 token 预算；不等同于模型单次上下文窗口。 |
| `--max-seconds` | 每个 case 的最长运行时间（秒）。 |
| `--target-file` | 指定 Agent 目标文件的相对路径；可重复传入多个文件。 |
| `--output-dir` | 指定结果根目录；默认写入 `test/benchmark_runs/`。 |
| `--auto-approve-all` | 自动批准 run 请求的写入、执行、网络、Git 写操作。仅在可信样例和隔离 workspace 中使用；不要指向日常开发目录。 |

## 3. 如何判定“改正确了”

不要只看 UI 的 `completed`、Agent 最后的文字或文件有变化。每个 case 分开记录：

1. **是否产出补丁**：比较 run 前后的文件哈希/差异；没有产品代码补丁，记为“未产出补丁”。
2. **是否改了测试**：检查所有变更文件；benchmark 测试和 evaluator 不应由 Agent 修改。正式验证最好在 Agent 无权写入的干净副本/容器中进行。
3. **本地检查**：在适用时检查编译、格式/静态检查和开发者测试。通过本地检查不自动等于官方通过。
4. **官方评测**：将补丁应用到对应 benchmark 的干净基础版本，运行该 benchmark 官方 evaluator。只有实际评测通过才记录 `official_pass=true`。
5. **基础设施错误**：镜像拉取失败、评测器未启动、缺少依赖等记为“未评分/基础设施失败”，不能当作 Agent 通过或代码失败。

当前样例池的官方验证要求因 benchmark 而异：

- Defects4C：使用官方评测服务，或完整项目与官方测试环境；本地函数片段不是官方评分。
- SWE-bench Multilingual：使用任务对应的仓库版本与测试 harness。标准 SWE-bench 通过 Docker 应用 patch 并运行测试；输出中的 resolved/unresolved 才是正确性结果。[官方评测说明](https://www.swebench.com/SWE-bench/guides/evaluation/)
- SecRepoBench：需要完整仓库；官方评估包括仓库编译、相关开发者单元测试和 OSS-Fuzz PoC 安全测试，官方仓库提供 `run_eval.py`。[官方仓库](https://github.com/ai-sec-lab/SecRepoBench)
- SWE-Atlas Refactoring：按其 Harbor 沙箱与 refactoring 评估流程执行；保留程序化检查和 rubric 分项结果。[官方仓库](https://github.com/scaleapi/SWE-Atlas)

## 4. 指标与报告

建议报告至少包含：

- **补丁产出率** = 有产品代码补丁的 case 数 / Agent 实际运行数。
- **官方通过率** = 官方 evaluator 通过数 / 官方实际评分数。另列未评分数和基础设施失败数；未评分不能填 0。
- **本地构建/测试通过率**：单独列示，不替代官方通过率。
- **测试篡改数**：Agent 修改 benchmark 测试或 evaluator 的 case 数。
- **漏洞密度**：统一扫描器、版本、规则集和统计范围；按“确认漏洞数 × 100 / 生成代码非空 LOC”计算。只有实际扫描的 case 才有分数；未扫描保持未评分。保留原始扫描报告（如 SARIF）并记录人工复核结果。
- **效率**：输入/输出 tokens、模型/工具调用次数、耗时和费用。
- **CodeGraph 对比**：报告 off/on 的同题配对官方结果，以及 token、耗时变化；不能只比较 Agent 的自我评价。

### 导入 evaluator 和扫描器结果

每次 Agent 运行会在 `test/benchmark_runs/<run-name>/` 下生成 `results.jsonl`、`summary.json`，并为每题保存 `events.json`、`manifest.json`、workspace 和运行状态。官方评测输出应另存为 JSONL，例如：

~~~json
{"case_id":"D4C-10-7722455726bec8c53458a32851d2a87982cf0eac___gd_gd","codegraph":"off","official_pass":true,"official_metric":"official_resolved","official_value":1,"official_result_source":"官方评测报告路径/运行ID"}
~~~

安全扫描结果另存为 JSONL：

~~~json
{"case_id":"D4C-10-7722455726bec8c53458a32851d2a87982cf0eac___gd_gd","codegraph":"off","generated_nonblank_loc":24,"self_check_vulnerability_findings":0,"security_result_source":"扫描器/版本/规则集/报告路径"}
~~~

字段必须来自真实 evaluator/扫描器输出。`official_pass=false` 只用于“已经正式评分但未通过”；未运行或基础设施失败时不要导入该 case，让它保持未评分。漏洞数为 0 只表示扫描完成且未发现符合规则的漏洞，**不表示没扫描**。

把 `<run目录>` 换成实际路径后汇总：

~~~powershell
python test/summarize_naturalcc_benchmark.py 'test/benchmark_runs/<run-name>/results.jsonl' --official-results 'test/benchmark_runs/<run-name>/official_results.jsonl' --security-results 'test/benchmark_runs/<run-name>/security_results.jsonl' --output 'test/benchmark_runs/<run-name>/summary.metrics.json'
~~~

`summary.metrics.json` 中的 `official_pass_rate` 只按有官方布尔结果的题计算；脚本也会保留未评分 case 的区分。SWE-Atlas 等输出多维 rubric 分数的评测，建议同时保存原始分项结果，并预先定义通过阈值后再映射为 `official_pass`，不要丢弃原始分数。

## 5. 结果解读底线

- `completed` = Agent 流程结束，**不等于**修复成功。
- 有 diff = Agent 修改了文件，**不等于**修改正确。
- 编译/本地测试通过 = 局部验证通过，**不等于**官方 benchmark 通过。
- 没有官方评测结果 = 正确性尚未评分；没有安全扫描结果 = 漏洞密度尚未评分。
- 报告结论以官方 evaluator 和可复现的扫描报告为依据，不以 Agent 的自述为依据。

