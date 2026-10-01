# 指标 1/3/5 离线运行指南

本指南说明 NaturalCC 侧的离线运行路径和可复现的 smoke 检查。平台后端仍需按接口契约传递 Agent 请求或 `/api/run` 参数，并保存、展示返回的 artifacts；本指南不替代平台集成验收。指标阈值是否达标由合同验收方依据正式数据和口径判断。

## 运行路径与职责

| 指标 | NaturalCC 调用路径 | 模型要求 | NaturalCC 返回内容 |
|---|---|---|---|
| 1 | `/api/agent/runs` Agent Runtime | Agent Gateway 与 Aider 都要使用同一台机器上的本地 Ollama。覆盖 Pipeline 的 `code_completion` 和通用 `aider.edit` 等 Aider 编辑入口。 | Agent 运行事件、工具结果和修改文件；模型需能按 Agent 工具协议发出结构化 tool calls。 |
| 3、5 | `/api/run` 的 `vulnerability_detection` 插件 | `auto_fix=false` 时为本地扫描，不调用模型，也不启动 Aider。 | 候选 findings、coverage；请求 `scan_type` 后返回 `contract_statistics`，没有有效真值或覆盖不足时状态为 `not_evaluated`。 |

指标 1 的 Ollama API 地址只允许本机 HTTP loopback，并且以 `/v1` 结尾。Aider 子进程由 NaturalCC 自动生成 `OLLAMA_API_BASE`，去掉 `/v1`；`aider.edit` 与 Pipeline Aider 路径都会使用 Run 的本地模型配置。不要为 Web Agent 手工设置 `OLLAMA_API_BASE`。

指标 3/5 扫描模式可选 `builtin`、`auto`、`cppcheck`、`comprehensive`。`builtin` 不运行外部分析器；`auto` 使用内置规则及可用的 Cppcheck；`cppcheck` 需要 Cppcheck；`comprehensive` 需要 Cppcheck 和 Clang。自然语言描述的七类缺陷不代表七类已被完整覆盖：内置竞争检查、Clang 实验规则、语言解析和分析器 coverage 都有边界，见[安全检测联调说明](SECURITY_ACCEPTANCE.md)。

平台后端负责实现 `/api/agent/runs`、`/api/run` 的调用和参数透传，并传递、保存及展示返回的 `findings`、`finding_summary`、`coverage`、`contract_statistics` 等 artifacts。NaturalCC 提供扫描候选与状态；合同指标的正式数据、ground truth、计分及达标判定由指标/验收负责人负责。CWE 是诊断标识，不能直接当作合同类别或 TP/FP 标签。

## 断网前准备

以下准备应在目标运行机器上联网完成；断网后继续使用已安装的环境和本地服务。若要迁移到另一台隔离机器，应按组织流程提前转移并验证 Python 环境、系统库、分析器、前端产物和 Ollama 模型文件。

1. **NaturalCC 运行依赖**：安装 Python 3.12、`uv`、系统 `libclang`，然后在 `code_agent/` 执行 `uv sync`。锁定环境包含 Aider 和 Python `clang==18.1.8`；C/C++ 解析还需要兼容的系统 `libclang`。不使用知识图谱时无需额外安装 CodeGraph CLI。
2. **本地扫描器**：安装 Cppcheck。若要使用 `analyzer=comprehensive`，另外安装 Clang 静态分析器 CLI，并确认服务进程的 `PATH` 能找到 `cppcheck` 和 `clang`。普通 `/api/run` 内置扫描不要求模型。指标 3/5 的综合 smoke 脚本要求这两个外部分析器。
3. **Tokenizer 资源**：确认仓库内 `code_agent/resources/deepseek_v3_tokenizer/tokenizer.json` 和 `tokenizer_config.json` 已存在且有效。Agent 运行时从该目录离线计数；文件缺失时不能依靠字符数回退。不要在断网后运行会下载资源的 `scripts/install_deepseek_tokenizer.py`。
4. **前端（仅需 Web UI 时）**：提前执行 `cd code_agent/webui && npm ci && npm run build`。仅使用 API 时无需 Node.js/npm。
5. **Ollama（指标 1）**：提前安装并启动 Ollama，执行 `ollama pull qwen2.5-coder:7b`，确认 `ollama list` 能看到所选模型。该默认模型需要与本地 Agent 工具调用能力相匹配；换用其他模型时，用目标模型实际 smoke，不要仅按名称推断支持情况。
6. **测试工程（可选）**：`scripts/prepare_security_acceptance.py` 的完整模式会下载固定上游源码，应在联网时准备并保留 `artifacts/security-acceptance/workspaces/`。离线 fixture 模式不下载源码，但只验证随附小型固定样例，不满足目标操作系统工程验收。

`uv sync`、Tokenizer 安装、npm 安装和 `ollama pull` 都可能访问网络。冷启动离线安装不是本指南已验证的场景；应先在最终目标机上准备好这些依赖，再断网运行。

## Ollama 配置与检查

将 `.env.example` 中的 Agent 配置替换为以下值，或在启动服务的同一 shell 中设置环境变量。代码只自动读取环境变量，不会因为 `.env.example` 存在就加载它。

```bash
export CODE_AGENT_PROVIDER=ollama
export CODE_AGENT_MODEL=qwen2.5-coder:7b
export CODE_AGENT_API_BASE=http://127.0.0.1:11434/v1
export CODE_AGENT_CONTEXT_WINDOW_TOKENS=8192
export CODE_AGENT_OUTPUT_RESERVE_TOKENS=2048
```

`CODE_AGENT_CONTEXT_WINDOW_TOKENS` 只控制 NaturalCC 的请求预算，不会修改 Ollama 服务端的上下文长度。应确保模型支持范围、Ollama 实际 `num_ctx`、输入预算、输出预留、provider framing 和安全余量相互匹配。若需固定服务端上下文，可用 Ollama Modelfile 创建本地模型标签，例如：

```text
FROM qwen2.5-coder:7b
PARAMETER num_ctx 8192
```

保存为 `Modelfile` 后执行 `ollama create qwen2.5-coder-local -f Modelfile`，然后把 `CODE_AGENT_MODEL` 改为 `qwen2.5-coder-local`。具体语法与模型限制见 [Ollama Modelfile 文档](https://docs.ollama.com/modelfile)。

启动前在服务所在机器检查：

```bash
ollama list
curl --fail http://127.0.0.1:11434/api/tags
uv run aider --version
cppcheck --version
clang --version
```

如果只做指标 1，可跳过 Cppcheck/Clang 检查；如果只做指标 3/5，完全可以不启动 Ollama。默认模型名 `qwen2.5-coder:7b` 可被环境变量覆盖；上下文窗口示例值 8192 也必须按部署实例核实。项目 tokenizer 是随仓库提供的 DeepSeek tokenizer，对 Ollama 请求只是 token 预算估算，并非 Ollama 模型原生 tokenizer。

## 离线 smoke 步骤

### 指标 1：Agent Gateway 与 Aider

1. 确认目标代码工作区在服务进程可访问的本机路径。启动 Ollama 和 NaturalCC Web 服务：`uv run python agent_web_api.py --host 127.0.0.1 --port 7860`。
2. 在 Web UI 新建 Agent Thread，确认 Provider 为 Ollama、Model 为已拉取的本地 tag、Base URL 为 `http://127.0.0.1:11434/v1`。若通过平台 API 创建 Thread/Run，应显式传入相同的 `runtime_model_config`，并确保 Agent Gateway 与 Ollama 在同一运行主机上。
3. 在临时副本中提交一个小型代码修改任务，让 Agent 读取目标文件并调用 `code_completion` 或 `aider.edit`。按 UI/API 返回执行审批后继续运行。检查 Run 事件中有 tool call 和成功的 Aider 结果，并核对改动 diff；仅收到模型文本而没有工具调用，不算完成此 smoke。
4. 记录模型 tag、Ollama 上下文配置、Run ID、调用的工具、结果状态和 diff。小参数模型（例如 1.5B）最多用于连通性/工具调用 smoke，不能证明合同代码质量或指标 1 达标。

### 指标 3/5：本地 `/api/run` 扫描

1. 若使用固定 fixture，在 `code_agent/` 执行：

   ```bash
   uv run python scripts/prepare_security_acceptance.py --fixtures-only --output artifacts/offline-smoke
   ```

2. 启动 NaturalCC 服务，然后在另一终端针对准备好的 workspace 执行：

   ```bash
   uv run python scripts/check_security_acceptance.py \
     --workspace artifacts/offline-smoke \
     --output artifacts/offline-smoke-result.json \
     --api-url http://127.0.0.1:7860
   ```

   此脚本通过 `/api/run` 调用扫描器，使用 `analyzer=comprehensive`、`auto_fix=false` 和 fixture 的真值文件；因此该检查需要 Cppcheck 与 Clang，但不需要模型。fixture 结果只证明固定样例上的接口/回归行为，不能外推为合同指标或目标工程结果。

3. 若扫描没有有效的合同 ground truth，可调用 `/api/run` 并省略 `ground_truth_file`。查看 NDJSON 最后一条 `type=done` 的 `artifacts`：`contract_statistics.status` 应为 `not_evaluated`，这是没有计分标签时的状态，不是零误报或零漏报。提供真值后，也只有在 schema、`scan_type`、扫描文件和当前源文件 SHA-256 均匹配，且相关分析器覆盖足以计分时才会计算有效统计。
4. 检查每个 `coverage` 项和 `contract_statistics.coverage_incomplete`。保留原始响应和日志；不把空 finding 列表、分析器缺失或 TSan 未运行解释成“已证明没有漏洞”。

TSan 字段只读取工作区内已有的预生成报告；NaturalCC 不在 `/api/run` 中编译或运行待测程序，也不会替用户生成 ThreadSanitizer 报告。动态报告的产生需要另行在支持 TSan 的环境完成。

## 常见失败定位

| 现象 | 检查方式 |
|---|---|
| Agent Gateway 连接失败或配置返回 422 | 确认 Ollama 正运行、模型已拉取；Base URL 必须是本机 HTTP loopback 且以 `/v1` 结尾。远程 Ollama 地址不符合当前 NaturalCC 校验。 |
| Agent 有文本回复但没有工具执行 | 检查所选模型是否在真实 smoke 中发出结构化 tool calls；查看 Run 事件中的 model/tool 调用和模型响应。换模型后重新 smoke。 |
| `aider.edit` 或 Pipeline 编辑失败 | 检查目标文件可访问、`uv sync` 中的 Aider 可执行、模型 tag 和 Tool Run 配置。查看工具结果及 `aider.log`。Web Agent 路径由代码生成 Aider 的本机 `OLLAMA_API_BASE`，应检查 Gateway Base URL，不要手工拼错变量。 |
| coverage 显示 `unavailable`、`failed` 或 `partial` | 检查配置的 analyzer、服务 `PATH`、Cppcheck/Clang 可执行和解析日志。`auto` 下外部 Cppcheck 不可用会降低覆盖；`cppcheck`/`comprehensive` 的必需分析器失败时扫描可能失败或统计不可用。 |
| 合同统计显示 `not_evaluated` | 核实请求是否省略真值文件。正式统计需提供与当前扫描源码 hash 一致的、符合 schema 的 ground truth；不要复用源码修改前的行号/哈希。 |
| TSan 报告导致 coverage 失败 | 该报告记录了运行时失败或不是有效 race 报告；NaturalCC 只导入报告，不会重跑程序。按 [安全检测联调说明](SECURITY_ACCEPTANCE.md) 在兼容环境重新生成报告。 |

## 验收边界

本指南描述可执行的离线部署与 smoke 流程，不是验收结果。本次文档更新没有在 openEuler 实机运行服务，也没有执行真实 Ollama 模型推理；测试中的 mock/单元回归不能替代这两项验证。固定 fixture 不是独立 ground truth 或正式工程样本。缺少经确认的正式 ground truth 时，指标统计应保持 `not_evaluated`；不得据候选 findings 推导 TP/FN/FP，也不得宣称合同阈值通过。
