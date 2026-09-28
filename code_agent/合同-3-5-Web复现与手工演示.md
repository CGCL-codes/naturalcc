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

## 2. 启动 Web 服务（OpenRouter 经 Clash）

先在另一终端启动 Clash。然后在**同一个终端**进入 `code_agent/`，一次性加载凭据、配置模型和 Clash 代理，最后只启动一次服务。不要把密钥写入源文件、文档或聊天记录。

```bash
cd /home/sub4-wy/wangchen/ncc/naturalcc/code_agent

export CODE_AGENT_PROVIDER="openrouter"
export CODE_AGENT_MODEL="anthropic/claude-sonnet-4.5"
export CODE_AGENT_API_BASE="https://openrouter.ai/api/v1"

# 本机凭据已配置在 ~/.bashrc；新开的 Bash 会自动加载。
# 若当前终端早于该配置，执行一次：source ~/.bashrc

# OpenRouter 经 Clash；本机网页/API 直连，不送入 Clash
export HTTP_PROXY="http://127.0.0.1:7899"
export HTTPS_PROXY="http://127.0.0.1:7899"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
unset ALL_PROXY all_proxy

# 首次运行或前端有变更时执行
cd webui && npm run build && cd ..

uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

浏览器访问 `http://127.0.0.1:7860/`。服务运行期间不要关闭第一个终端。

### 2.1 使用 Clash 访问 OpenRouter

代理变量已包含在第 2 节唯一的启动命令中。若服务先前已在未设置代理的环境中启动，先在旧服务终端按 `Ctrl+C` 停止它，再从第 2 节的完整命令块重新执行；不要只补设代理变量后直接复用旧服务。

启动后可在另一个终端验证 Clash 的 HTTP 代理连通性：

```bash
curl --silent --show-error --proxy http://127.0.0.1:7899 \
  --connect-timeout 10 --max-time 20 -o /dev/null \
  -w 'OpenRouter through Clash: HTTP %{http_code}\n' \
  https://openrouter.ai/api/v1/models
```

成功时会显示 HTTP 200。实际模型请求时，Clash 日志应出现 `openrouter.ai:443` 并显示其选中的节点。若之前因 403 中断指标 3，代理生效后先只补跑失败组：

```bash
uv run python scripts/benchmark_contract3_web.py --suite balanced --retry-failed
```

## 3. 正式复现

另开一个终端，进入同一目录。`OPENROUTER_API_KEY` 已由 `~/.bashrc` 自动提供；若当前终端尚未加载该文件，先执行 `source ~/.bashrc`。指标 3 有两套不能混合计分的测试集：`balanced` 是平衡验收集，`independent` 是独立复杂压力集。

```bash
cd /home/sub4-wy/wangchen/ncc/naturalcc/code_agent
source ~/.bashrc  # 仅当前终端未自动加载凭据时需要

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

指标 3 的 Web 脚本会在终端逐组打印“排队/运行/已保存”状态；如果结果已存在，也会明确打印“跳过（已有结果）”。结束时会打印各类别 TP/FN/FP/TN、检出率、误报率、判定以及报告和 checkpoint 路径，因此无需只打开 Markdown 才能查看结果。使用 `--report-only` 时不调用模型，只打印已有结果摘要：

```bash
uv run python scripts/benchmark_contract3_web.py --suite balanced --report-only
```

如需重新请求模型并产生一轮新的指标 3 Web 结果，服务和密钥均已配置时使用 `--force-rerun`。该选项会对选中的组创建新 Run、保留旧 Run 记录，并以本轮完整完成的结果更新当前统计；会消耗模型额度。

```bash
# 9 组完整平衡验收集，结束时输出本轮新的完整得分
uv run python scripts/benchmark_contract3_web.py --suite balanced --force-rerun
uv run python scripts/benchmark_contract3_web.py --suite independent  --force-rerun

# --limit 1 仅作连通性冒烟，不能作为完整新得分
uv run python scripts/benchmark_contract3_web.py --suite balanced --limit 1 --force-rerun
```

指标 5 使用下面命令离线刷新已有结果：

```bash
uv run python scripts/benchmark_contract5_web.py --report-only
```

## 4. 指标 3 网页手工演示

网页演示展示产品的只读漏洞检测能力；它不替代第 3 节脚本产生的正式验收统计。手工演示前，目标 C 文件必须位于 **Project root** 所指向的本地目录中。

### 4.1 演示前准备

可以使用自己的 C 文件，也可复用运行指标 3 脚本后生成的 fixture。后者适合现场展示三类风险：

| 演示类别 | Project root 示例 | 添加到上下文的文件 |
|---|---|---|
| 数组越界 | `/tmp/naturalcc-contract3-balanced-sources/freertos_kernel` | `contract3_balanced_web_array_oob.c` |
| 字符串溢出 | `/tmp/naturalcc-contract3-balanced-sources/freertos_kernel` | `contract3_balanced_web_string_overflow.c` |
| 空指针调用 | `/tmp/naturalcc-contract3-balanced-sources/freertos_kernel` | `contract3_balanced_web_null_pointer.c` |

这些 fixture 仅在先执行过 `benchmark_contract3_web.py` 后存在；若目录被清理，可改用自己准备的 `.c` 文件。不要在网页中向 Agent 提供正负例标签、参考行号或真值 JSON。

### 4.2 网页按钮与上下文操作

1. 打开 `http://127.0.0.1:7860/`。
2. 顶部运行模式选择 **Agent**，不要选择 **Pipeline**。
3. 点击右上角 **Settings**。
4. 在 **Workspace → Project root** 填写或选择上表中的工程目录，点击刷新图标 **Apply workspace**。
5. 在 **Model** 中选择 **Provider: OpenRouter**，将 **Model** 设为 `anthropic/claude-sonnet-4.5`。服务启动终端已配置 `OPENROUTER_API_KEY` 时，网页 **API key** 输入框留空即可。
6. 关闭 Settings，回到聊天输入框。输入 `@contract3_balanced_web_array_oob.c`（或上表其他文件），在候选项中点选该文件；也可粘贴绝对路径后点击 **Add context**。
7. 确认聊天框上方已显示选中的上下文文件，再输入审计请求。
8. 点击输入框右侧纸飞机图标 **Send**。这会创建一个新的 Agent 对话和 Run。

### 4.3 三类风险的可发送提示词

一次演示一个类别，便于观众核对 Agent 的分类、行号和理由。下面三段提示词任选其一发送；目标文件应与类别对应。

**数组越界：**

```text
请只读审计当前上下文 C 文件，只检测数组越界风险。
先完整阅读源码，再调用 vulnerability_detection 辅助检查；扫描无告警不等于安全。
逐个检查数组长度、下标、循环边界、指针算术和经辅助函数计算的索引。
不要修改、编译、运行文件，不执行 shell，不搜索外部资料。
请按“风险类别、真实行号、触发条件、代码证据”列出发现；没有发现时明确说明。
```

**字符串溢出：**

```text
请只读审计当前上下文 C 文件，只检测字符串或缓冲区溢出风险。
先完整阅读源码，再调用 vulnerability_detection 辅助检查；扫描无告警不等于安全。
逐个检查目标缓冲区长度及 strcpy、sprintf、strcat、memcpy 和包装函数的复制长度，计算结尾 NUL 字节。
不要修改、编译、运行文件，不执行 shell，不搜索外部资料。
请按“风险类别、真实行号、触发条件、代码证据”列出发现；没有发现时明确说明。
```

**空指针调用：**

```text
请只读审计当前上下文 C 文件，只检测空指针调用或解引用风险。
先完整阅读源码，再调用 vulnerability_detection 辅助检查；扫描无告警不等于安全。
逐个追踪 0/NULL 初始化、条件分支、结构体指针成员、辅助函数传参和最终解引用点。
不要修改、编译、运行文件，不执行 shell，不搜索外部资料。
请按“风险类别、真实行号、触发条件、代码证据”列出发现；没有发现时明确说明。
```

如需一次性展示三类，可将第一句改成“检测数组越界、字符串溢出和空指针调用三类风险”，但分类演示建议仍分别运行三次。

### 4.4 观察过程、结果与审批

1. 点击顶部 **Run details**，查看状态是否从 `queued`、`running` 变为 `completed`。
2. 在事件中确认 Agent 已读取上下文文件，并调用 `vulnerability_detection`；在聊天最终答案中核对类别、行号和理由。
3. 若结果需要展示给验收人员，可展开 **Run details** 中的用量、工具事件和最终回答；它们是该手工演示的过程证据。
4. 本场景只允许静态只读分析。若出现 **Approve execute** 或 **Approve write**，不要批准，点击 **Reject** 或 **Cancel**。编译、执行和修改代码均不属于本次授权范围。
5. 手工对话完成后，正式指标分数仍以第 3 节脚本输出、`合同-3-result.md` 和冻结 checkpoint 为准。

## 5. 结果与边界

- 指标 3 的正式统计、Run 标识和测试集边界以 `合同-3-result.md` 及 `artifacts/contract3/` 下的冻结工件为准。
- 指标 5 的正式统计以 `合同-5-result.md` 和 `artifacts/contract5/` 为准。
- 手工对话可用于演示功能，但不能替代脚本生成的验收分数。
- 结果中的 RTOS 注入文件仅用于测试，不能据此声称已经覆盖或证明整个上游工程无漏洞。
