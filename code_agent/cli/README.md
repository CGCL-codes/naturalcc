# naturalcc CLI

`naturalcc` 是一个面向本地项目的 coding agent。它在项目目录中创建一次性任务或多轮 REPL 会话，使用 OpenAI-compatible Chat Completions 模型，并通过受限工具读取、搜索和修改工作区。

## 安装与启动

```bash
cd code_agent/cli
bun install
export OPENAI_API_KEY=sk-...
bun run naturalcc -- "检查项目并运行测试"
```

需要 PATH 中存在可执行的 `rg`（ripgrep）。启动时会用 `rg --no-config --version` 做健康检查；当前版本不会下载或打包 ripgrep。

不带任务且连接到 TTY 时启动 REPL：

```bash
bun run naturalcc
```

同一个 REPL 会话复用当前 `CodingAgentSession` 和 `AgentThread`。`Esc` 中断当前回合；`Ctrl-C` 在空闲时需要连续两次才退出。非 TTY 空输入只显示帮助，不启动 Ink。

## 配置

环境变量：

| 变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` | 必填的模型 API key；也可用 `--api-key` 覆盖 |
| `OPENAI_MODEL` | 默认模型；未设置时为 `gpt-5.5` |
| `OPENAI_BASE_URL` | 可选的 OpenAI-compatible endpoint；也可用 `--base-url` 覆盖 |

命令行选项：

| 选项 | 用途 |
| --- | --- |
| `--model <name>` | 模型名称 |
| `--api-key <key>` | API key |
| `--base-url <url>` | OpenAI-compatible endpoint |
| `--project-dir <dir>` | 工作区目录，默认当前目录 |
| `--max-turns <n>` | 单回合最多执行的模型/工具循环次数 |
| `--no-tool-results` | 隐藏成功工具结果；错误结果仍显示 |
| `--approval <mode>` | `prompt`、`deny` 或 `allow` |
| `--yes` | 将审批模式设为 `allow` |

无显式配置时，TTY 默认使用 `prompt`，非 TTY 默认使用 `deny`。缺少 API key、工作区无效或 ripgrep 健康检查失败会以配置错误退出码 `2`；中断退出码为 `130`，其他任务失败退出码为 `1`。

## 工具与安全边界

模型可使用以下结构化工具：

- `list_files`：通过 ripgrep 递归列出工作区内文件；默认遵循 `.gitignore`，不包含隐藏文件。`recursive` 默认是 `true`；显式设为 `false` 时只列出目标目录的直接文件。
- `read_file`：读取工作区内有界的 UTF-8 文本文件。
- `search_text`：在工作区内进行字面字符串搜索；默认遵循 `.gitignore`，不包含隐藏文件。
- `apply_patch`：按精确上下文应用 Codex envelope 补丁。使用 `*** Begin Patch`/`*** End Patch` 包住 `*** Add File: path`、`*** Update File: path` 或 `*** Delete File: path` 指令；Update hunk 首选裸 `@@`，也兼容完整的 `@@ -old[,count] +new[,count] @@`，每条 hunk body 必须以空格、`+` 或 `-` 开头（或使用字面量 `\\ No newline at end of file` marker）。它不会接受任意完整 `git diff` 的 `---/+++` 文件头，也不会模糊匹配上下文。
- `bash`：在工作区目录执行本地 shell 命令。

`includeIgnored` 和 `includeHidden` 只对列表/搜索工具显式放宽 ripgrep 的默认过滤。文件工具拒绝绝对路径、目录遍历和逃逸到工作区外的符号链接；二进制、NUL 和无效 UTF-8 内容不会作为文本交给模型。所有工具输出都有大小限制；bash 还有超时、取消和 POSIX 进程组清理。

`bash` 与 `apply_patch` 默认需要审批。prompt 模式会展示由 canonical command/patch 生成的真实指令预览，以及模型提供的 `Reason`；Ink 中默认选 `No`，用 `↑/↓` 和 `Enter` 确认，`Esc` 中断，单独输入 `y/n` 不会直接批准。one-shot/readline 使用相同的标题、指令和理由，并以 `[y/N]` 默认拒绝。每个需要审批的模型工具调用都必须提供非空、最多 500 个 Unicode 字符的 `reason`；reason 只说明必要性，不会改变实际执行参数。审批只是 agent 层策略，不是操作系统沙箱：允许 bash 后，命令仍拥有当前用户赋予进程的本地权限。`--yes` 或 `--approval allow` 只应在明确可信的工作区使用。

REPL 会通过标准化 streaming 增量显示模型文本；文本未完整结束或回合被取消时，半截 assistant/tool-call 不会提交到 AgentThread。每个工具调用在 REPL 历史中只有一条可原位更新的消息，默认不显示 callId/requestId；one-shot 的 stdout 只输出成功提交的最终 assistant_message，工具和审批状态写入 stderr，工具结果只写最终状态；取消或错误不会留下半截 stdout。工具结果和审批状态会保留在本次 REPL 的内存 UI 历史中，不依赖临时 streaming 展示；当前没有磁盘 transcript 或持久化历史。

REPL 中的 Assistant 文本使用受限 Markdown 渲染：代码块、标题、列表、强调、引用、链接和表格会转换为终端可读的纯文本/样式；危险链接、HTML、ANSI/OSC 和控制字符不会执行。`read_file`、`bash`、`list_files`、`search_text`、`apply_patch` 的结果按已知 JSON 结构展示，stdout/stderr 和文件内容保持预格式化，不做全局反转义。工具详情只保留有界的内存投影，并明确标记 source truncation；不会重新访问工作区。

当前不提供 `@` 文件引用、正则搜索、NaturalCC provider、磁盘 transcript 或远程沙箱能力。

## REPL 命令

| 命令 | 用途 |
| --- | --- |
| `/help` | 显示帮助 |
| `/clear` | 同时清除 UI 消息和 agent 对话历史 |
| `/exit` | 退出 REPL |

## 开发与验证

```bash
cd code_agent/cli
bun run typecheck
bun test
```

查看版本或 CLI 帮助：

```bash
bun run naturalcc -- --version
bun run naturalcc -- --help
```
