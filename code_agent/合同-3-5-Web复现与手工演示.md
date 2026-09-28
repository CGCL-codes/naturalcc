# 合同指标 3、5：Web 复现与手工演示

本文说明合同指标 3 与指标 5 的 Web Agent 复现方式，以及在网页中手工演示静态漏洞检测的操作步骤。

## 1. 两种运行方式

两项指标的正式 Web 测试都走同一条链路：本机 Web Agent API 创建对话、发送消息、启动 Run、读取事件并持久化结果。脚本不是直接请求模型，也不依赖浏览器点击自动化。

| 项目 | 指标 3 | 指标 5 |
|---|---|---|
| 测试对象 | 3 套 RTOS 工程中的 3 类注入缺陷 | 40 个 CASTLE C 语言样本 |
| Web Run | 9 个工程×类别组 | 40 个独立样本组 |
| 入口脚本 | `scripts/benchmark_contract3_web.py` | `scripts/benchmark_contract5_web.py` |
| 结果报告 | `合同-3-result.md` | `合同-5-result.md` |

正式验收应使用脚本。网页界面用于展示 Agent 如何读取代码、调用只读扫描工具、展示 Run 事件和生成结论；手工网页对话不会自动写入验收统计。

## 2. 启动 Web 服务

在终端一进入 `code_agent/`，配置模型并启动服务。不要把密钥写入源文件、文档或聊天记录；使用自己的环境变量值。

```bash
cd /home/sub4-wy/wangchen/ncc/naturalcc/code_agent

export CODE_AGENT_PROVIDER="openrouter"
export CODE_AGENT_MODEL="anthropic/claude-sonnet-4.5"
export CODE_AGENT_API_BASE="https://openrouter.ai/api/v1"
export OPENROUTER_API_KEY="你的 OpenRouter 密钥"

# 首次运行或前端有变更时执行
cd webui && npm run build && cd ..

uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

浏览器访问 `http://127.0.0.1:7860/`。服务运行期间不要关闭第一个终端。

## 3. 正式复现

另开一个终端，进入同一目录并配置 `OPENROUTER_API_KEY`。指标 3 有两套不能混合计分的测试集：`balanced` 是平衡验收集，`independent` 是独立复杂压力集。

```bash
cd /home/sub4-wy/wangchen/ncc/naturalcc/code_agent
export OPENROUTER_API_KEY="你的 OpenRouter 密钥"

# 指标 3：Web 平衡验收集
uv run python scripts/benchmark_contract3_web.py --suite balanced

# 指标 3：Web 独立复杂压力集
uv run python scripts/benchmark_contract3_web.py --suite independent

# 指标 5：Web Agent 测试
uv run python scripts/benchmark_contract5_web.py
```

各脚本会保存已评分结果；再次执行会跳过已评分样本，不会为同一结果重复发起模型调用。指标 3 不发起模型请求、只根据已有工件刷新总报告：

```bash
uv run python scripts/render_contract3_report.py
```

指标 5 使用下面命令离线刷新已有结果：

```bash
uv run python scripts/benchmark_contract5_web.py --report-only
```

## 4. 网页手工演示

手工演示前，目标 C 文件必须位于 `Project root` 所指向的本地目录中。

1. 打开 `http://127.0.0.1:7860/`。
2. 顶部运行模式选择 **Agent**，不要选择 **Pipeline**。
3. 点击右上角 **Settings**。
4. 在 **Workspace → Project root** 填写或选择包含目标 C 文件的目录，点击刷新按钮 **Apply workspace**。
5. 在 **Model** 中选择 **Provider: OpenRouter**，并将 **Model** 设为 `anthropic/claude-sonnet-4.5`。若服务启动时已设置环境变量，网页中的 **API key** 可留空。
6. 回到聊天框，输入 `@文件名`，从候选列表点选目标 `.c` 文件；也可粘贴绝对路径后点击 **Add context**。
7. 输入以下审计请求：

```text
请只读审计当前上下文文件中的数组越界、字符串溢出和空指针调用。
先完整阅读源码，再调用 vulnerability_detection 辅助检查。
不要修改、编译、运行文件，也不要执行 shell。
请按类别、行号、触发条件列出发现；没有发现时明确说明。
```

8. 点击输入框右侧纸飞机图标 **Send**。
9. 点击顶部 **Run details**，查看状态、模型/工具用量、事件和最终答案。

本场景仅需只读分析。若 Run 出现 **Approve execute** 或 **Approve write**，不要批准；点击 **Reject** 或 **Cancel**，因为编译、执行和修改代码都不属于本次静态检测授权范围。

## 5. 结果与边界

- 指标 3 的正式统计、Run 标识和测试集边界以 `合同-3-result.md` 及 `artifacts/contract3/` 下的冻结工件为准。
- 指标 5 的正式统计以 `合同-5-result.md` 和 `artifacts/contract5/` 为准。
- 手工对话可用于演示功能，但不能替代脚本生成的验收分数。
- 结果中的 RTOS 注入文件仅用于测试，不能据此声称已经覆盖或证明整个上游工程无漏洞。
