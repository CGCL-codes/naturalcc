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

同一个 REPL 会话复用当前 `CodingAgentSession` 和 `AgentThread`。`Esc` 关闭补全或中断当前回合；运行中的 `Ctrl-C` 中断当前回合。空闲时，非空输入的第一次 `Ctrl-C` 会清空输入并显示退出提示，下一次退出；空输入的第一次 `Ctrl-C` 显示提示，下一次退出。非 TTY 空输入只显示帮助，不启动 Ink。

每个新 `CodingAgentSession` 会读取一次根目录 `AGENTS.md`/`CLAUDE.md`，并采集一次有界的 `git status --short` 初始化快照；该 Git 快照会明确标注采样时刻，之后不会因新回合、工具修改或压缩自动刷新，`/clear` 成功创建新 Session 时才重新采集。工作区身份在 Session 内固定。日期和时区在每次实际模型请求的准备边界检查，仅当它们变化时以框架 system 上下文更新；更新不写入用户/Assistant/工具历史，也不会触发额外模型请求。需要实时 Git 状态时，应让 agent 通过原有受审批工具查询。

回合运行期间键入的可打印文本保留为有界 type-ahead（上限 256 个 Unicode 字符：按 scalar 值计数截断，不拆分代理对；控制键与孤立代理对不排队，`Ctrl-C`/`Esc` 中断会清空队列），并在输入行恢复交互时一次性提交，因此终端帧与输入行重新挂载之间到达的 `/exit` 等按键不会被丢弃；审批菜单或 `/clear` 期间的键入保持原有不可决策语义，不进入 type-ahead。

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
| `--max-turns <n>` | 单回合模型请求硬上限，默认 `32`，显式值必须为 `1..256` 的整数 |
| `--no-tool-results` | 隐藏成功工具结果；错误结果仍显示 |
| `--approval <mode>` | `prompt`、`deny` 或 `allow` |
| `--yes` | 将审批模式设为 `allow` |

无显式配置时，TTY 默认使用 `prompt`，非 TTY 默认使用 `deny`。缺少 API key、工作区无效或 ripgrep 健康检查失败会以配置错误退出码 `2`；中断退出码为 `130`，其他任务失败退出码为 `1`。输出预算配置在两个入口（app `resolveCodingAgentConfig()` 与兼容 `createRuntimeConfig()`）共享同一规范最小值 `192`：低于它的预算装不下最小 canonical 工具结果 envelope，会在配置期失败；直接注入 ToolRegistry/context 的底层通用下限仍是 `96`，由各工具对超长行等情形诚实 fail closed。

`--max-turns` 按一次用户请求发起的全部模型请求计数（包括维护与收尾），是绝对天花板，不保证一定运行到该数值；工具批次中的多个调用只占一次模型请求。当前内部 segment 候选为 32，并预留最后一次无工具请求生成未完成的 `PauseCheckpoint`，所以通常至多先发送 31 次 non-closing 请求，第 32 次用于收尾；防循环、上下文或其他已启用预算也可能更早停止。缺少可信 context/token capability 时，即使显式配置 `--max-turns 64`，当前实现也不会自动越过首个 32-request segment，而会保守收尾。REPL 以暂停警告交还控制权，one-shot 将检查点写入 stderr 并退出 `1`，这不是成功 `done`。显式 `--max-turns 1` 没有关闭请求；如果唯一模型响应要求工具，任务以 `max_turns_reached` 结束且不会生成检查点，已执行工具造成的工作区变化不会被回滚。

## 工具与安全边界

工具错误通过原有结果向模型返回稳定的细分原因、字段规则和可用的修复提示；提示受工具输出预算约束，预算不足时省略可选细节。参数错误在审批前拒绝；文件不存在、权限或 I/O 失败按实际执行阶段报告，不能视为同一种参数错误。提示不会自动修改参数或执行重试，重复错误仍受原防循环规则约束；取消、审批及可能已发生的文件修改语义不变。

模型可使用以下结构化工具：

- `list_files`：发现、列举、筛选或统计项目文件时优先使用本工具，不要用 `bash`/`find`/`rg` 等价完成同样的工作。通过 ripgrep 递归列出工作区内文件；默认遵循 `.gitignore`，不包含隐藏文件。`recursive` 默认是 `true`；显式设为 `false` 时只列出目标目录的直接文件。`mode:"list"`（默认）返回 `{ entries, truncated }`，只有 `truncated:false` 时 `entries.length` 才是完整匹配数；`mode:"count"` 不返回路径，只在计数完整时返回 `{ mode:"count", count, complete:true }`，且不能与 `limit` 同时使用。它不会列出被永久排除的 `.git` 目录。
- `read_file`：先配合 `search_text` 定位，再读取工作区内有界的 UTF-8 连续行页。`startLine` 可是任意安全正整数；省略范围时默认返回最多 200 行，单页最多 300 行，也可用 `maxLines` 控制页大小。显式 `endLine` 与 `maxLines` 不能同时使用；`complete:false` 时必须使用结果中的 `nextStartLine` 继续，不能把分页结果当作完整文件。结果同时包含 `startLine`、`endLine`、`totalLines`、`content`、`complete` 和 `nextStartLine`（未完成时）。仅当显式传 `includePagePlan:true` 且当前页未到 EOF 时，结果额外携带 `suggestedContinuationRanges`（最多 3 个尚未读取的连续后续范围，不含当前页）、`pagePlanComplete` 与 `nextPlanStartLine`（未覆盖 EOF 时为最后建议行 +1）：每个建议范围都按真实 `JSON.stringify` 长度证明可独立装入输出预算，计划 metadata 本身计入当前页预算（必要时按整行缩短当前页）；当前页已 `complete:true`、无法生成至少一个合法后续范围或预算不足时整体省略计划字段。计划只是基于读取时快照的建议：工作区被修改后应重新搜索/读取并忽略旧计划，后续调用始终以执行时的文件内容为准。
- `search_text`：在工作区内搜索文本；`mode:"literal"`（默认）按字面字符串匹配，`mode:"regex"` 显式使用 ripgrep 的默认 Rust regex engine。regex 仍是单行、有超时和输出预算的搜索，不启用 PCRE2/auto、lookaround、backreference 或跨行匹配；不支持的表达式会作为可修复的工具参数错误返回。query 最多 2,000 个 UTF-16 code units 且不超过 4,096 UTF-8 bytes。默认遵循 `.gitignore`，不包含隐藏文件。
- `apply_patch`：按精确上下文应用 Codex envelope 补丁。使用 `*** Begin Patch`/`*** End Patch` 包住 `*** Add File: path`、`*** Update File: path` 或 `*** Delete File: path` 指令；Update hunk 首选裸 `@@`，也兼容完整的 `@@ -old[,count] +new[,count] @@`，每条 hunk body 必须以空格、`+` 或 `-` 开头（或使用字面量 `\\ No newline at end of file` marker）。它不会接受任意完整 `git diff` 的 `---/+++` 文件头，也不会模糊匹配上下文。
- `bash`：在工作区目录执行本地 shell 命令，适合运行测试、构建、项目脚本等结构化工具无法表达的工作。查找、列举、搜索或统计项目文件时优先使用 `list_files`/`search_text`，不要仅因熟悉 `find`/`rg`/`grep` 绕过等价的结构化工具。
- `update_task_plan`：按需管理当前 run 的显式任务清单，并有界查询当前 Session 内已经冻结的历史 run 快照。简单问答、局部解释和单一明确操作默认不建清单；一旦合法建立清单，正常收尾前必须声明当前 revision 的全部阶段完成。历史快照只读，不能复活旧计划或授权重放旧工具。`list_snapshots`/`read_snapshot` 为每条历史标注 `sourceReadable`/`sourceChars`：原始 prompt 仍在已提交 Thread 中时可再用同一 `snapshotId` 加 `sourceOffset`/`sourceLimit`（每页 ≤4000 字符，`sourceExcerpt.nextOffset` 续读）分页读取原文；已回滚、清除或 dispose 的来源如实返回不可读，不悄悄改查其他消息。

另提供可由 app 显式注入的候选只读工具 `git_changes`（`summary | files | patch`，默认 `scope:"both"`）。它区分 staged、unstaged、untracked，并只对一个逻辑文件返回连续 patch 页；非 Git 目录明确返回 `applicable:false`，子模块 worktree 不递归观察，指向子模块内部的 patch 请求会明确拒绝。该候选会拒绝 `core.worktree` 重定向、可执行 filter、external diff/textconv、动态 include、fsmonitor 和自定义 attributes；实际 git-dir/work-tree 由可信 discovery 固定，所有 Git 子进程都禁用 pager、交互、lazy fetch 与隐式子模块检查。Git 的 NUL 元数据、quoted path、patch header/hunk 会严格校验，畸形结果不会被投影成可信变更。当前默认 builtin 尚未注册该候选：该工具仍为可选集成，默认关闭，因此普通 CLI 不会在 system instruction 中宣称它可用。显式集成必须由 app 注入 `GitChangesBackend`，不能由 Session 或模型参数提供 workspace/repository 身份。

显式注册后，模型指引按需求选查询：数量汇总用 `summary`，路径/动作/metadata 清单用 `files`，已知单文件差异直接 `patch`，未知范围的全仓内容总结先定位再读相关 patch；untracked 正文用 `read_file`。用户要求全部修改时仍须覆盖所需范围。明确的 mode/rename/binary 等字段可支持对应事实，不要求无条件用 Bash 重查；unknown、不适用和未观察子模块不能当成确定或 clean。`complete:true` 只代表当前 mode/scope/path 查询完整，不代表任务完成或验证通过。待读页沿 `nextCursor` 继续；stale/observation changed 后从首页重查，不混新旧页。要求都有来源、范围足够且无缺页/冲突/已知新变化时，停止无目的重复取证，转入回答或下一阶段；缺失、变化、显式交叉验证和修改后测试仍可复查，合法 Bash 能力及原审批保留，不能绕过安全拒绝。这是模型指引，不是自动路由/结束器；不保证减少模型请求或 token 消耗。

`includeIgnored` 和 `includeHidden` 只对列表/搜索工具显式放宽 ripgrep 的默认过滤。文件工具拒绝绝对路径、目录遍历和逃逸到工作区外的符号链接；二进制、NUL 和无效 UTF-8 内容不会作为文本交给模型。所有工具输出都有大小限制；bash 还有超时、取消和 POSIX 进程组清理。

`bash` 与 `apply_patch` 默认需要审批。prompt 模式会展示由 canonical command/patch 生成的真实指令预览，以及模型提供的 `Reason`；Ink 中默认选 `No`，用 `↑/↓` 和 `Enter` 确认，`Esc` 中断，单独输入 `y/n` 不会直接批准。one-shot/readline 使用相同的标题、指令和理由，并以 `[y/N]` 默认拒绝。每个需要审批的模型工具调用都必须提供非空、最多 500 个 Unicode 字符的 `reason`；reason 只说明必要性，不会改变实际执行参数。审批只是 agent 层策略，不是操作系统沙箱：允许 bash 后，命令仍拥有当前用户赋予进程的本地权限。`--yes` 或 `--approval allow` 只应在明确可信的工作区使用。

REPL 会通过标准化 streaming 增量显示模型文本；文本未完整结束或回合被取消时，半截 assistant/tool-call 不会提交到 AgentThread。UI 完整接收每条 delta，但只合并短时间内的 React 展示刷新（默认 75 ms；首个 delta 立即可见）；terminal、完成段落、工具、审批和取消不会被该刷新间隔延迟。每个工具调用在 REPL 历史中只有一条可原位更新的消息，默认不显示 callId/requestId；one-shot 只有收到 `done` 才把最终 assistant 内容写入 stdout，工具、审批和 `paused` 检查点写入 stderr，工具结果只写最终状态；取消或错误不会留下半截 stdout。工具结果和审批状态会保留在本次 REPL 的内存 UI 历史中，不依赖临时 streaming 展示；当前没有磁盘 transcript 或持久化历史。

REPL 中的 Assistant 文本使用受限 Markdown 渲染：代码块、标题、列表、强调、引用、链接和表格会在 streaming 中按当前完整的有界来源重投影为终端样式；危险链接、HTML、ANSI/OSC 和控制字符不会执行。表格在表头分隔行到齐后复用完成态的同一布局，后续 cell 到达或终端 resize 时从完整有界来源重投影。表格保留 Markdown 的左/中/右对齐和 cell 内样式，按终端显示宽度补齐列并在列内换行；空间不足时先按稳定列序号显示一次完整表头映射，再让每个数据行按相同列序号展示字段，因此空/重复表头仍可区分，长表头也不会在每条数据中重复；cell 内容不会被静默截短。已经完成的普通 Assistant 段落和最终回复默认完整展示已保留的内容，不按工具摘要的 40 行/6,000 字符门槛折叠；运行中的长内容只在统一动态区域按物理行选择最新连续窗口，正文不插入隐藏计数或 head/tail 拼接，空间不足时状态行固定显示 `Streaming… · showing latest output`。完成后正文一次性进入 stable history。Markdown renderer 仍有 100,000 字符的硬 source 上限，真实 source truncation 会明确标注。`read_file`、`bash`、`list_files`、`search_text`、`apply_patch`、`git_changes` 的结果按已知 JSON 结构展示，stdout/stderr 和文件内容保持预格式化，不做全局反转义。`read_file` 的专用 presenter 严格解析 canonical 分页结果：摘要显示路径、绝对行范围、总行数和 continuation 提示（`(continue at N)`、`(EOF)`、空文件 `(empty file)`），正文保持预格式化；违反分页不变量（如 `nextStartLine !== endLine + 1`、`complete:true` 仍带 `nextStartLine`、兼容 `truncated !== !complete`）的结果按有界 raw JSON fallback 展示。`complete:false` 是 canonical continuation，不会被 UI 当作不可恢复的 source truncation；只有 UI 自身的展示裁剪才标记 truncation，且投影字段不会反写 canonical ToolResult。工具详情只保留有界的内存投影，并明确标记 source truncation；不会重新访问工作区。

`list_files` 的专用 presenter 当前面向 list 结果；count 结果仍会以安全、有界的通用 JSON 形式展示。这只是 UI 展示降级，不改变回填给模型和 AgentThread 的 canonical count 结果。

`update_task_plan` 的结构化结果把任务阶段直接映射为 `[ ]`、`[>]`、`[✓]`，不会从 Assistant 自由文本推断清单状态；`[✓]` 仅表示模型声明的阶段进度，不表示质量已验证。每个已受理 run 在终止后冻结一份有界、进程内快照；新 run 可读取仍保留的历史，但总是从无活动清单开始，`/clear`、替换或 dispose 后旧 generation 不再可用。

正在编辑的输入框（上横线/编辑行/下横线，极窄时单行降级）与已提交的用户消息正文共用同一"用户输入 surface"：深灰背景 `#262626` + 浅色正文 `#D8DEE9`（进程内固定 palette，无主题切换/自动亮暗探测）；框架生成的 `>` 提示符保持绿色，用户正文不再整行绿色；时间戳、状态栏、footer、overlay、Assistant 与工具结果背景不变。背景铺到各自区域现有的安全可绘制宽度（Ink 布局的 UI 填充，不写入消息/输入值/模型内容），换行、行数与光标语义不变；`NO_COLOR` 下关闭新增前/背景样式并保留原文本结构。

长对话会在**安全点**自动压缩：当可压缩历史（已完成 run 的完整工具 exchange 与当前 run 已保存的完整 exchange）超过历史额度一半的下限软目标时，Runner 在上一个工具批次完全回填后、下一次模型请求前，用一次**禁用工具的维护请求**生成结构化摘要（FACTS/FAILURES/UNFINISHED/SUGGESTIONS/UNCERTAINTY 五段，推断显式标注）。摘要只替换**派生请求视图**中被覆盖的旧 exchange（一次一条摘要、只认严格超集替换、最新一个完整 exchange 保持原文），原始消息仍完整保留在 AgentThread 中，revision/游标/收据不变。外壳裁剪（输入超过 400k 请求壳时）由 Thread 核心在派发前最终签发：engine 只提交选中的 unit ID，消息/计数/顺序由核心从自身冻结记录重建，provider 输入、prepare 与安装绑定同一最终收据；收据登记按 Thread 隔离（每线程有界登记簿，其他线程的压缩不会挤掉本线程在途来源）。净收益用共同 final serializer 计量（真实摘要投影 + JSON 转义 + request-only 段落 + 冻结的尾部框架消息），安装后的摘要在历史额度中占软槽（优先于等量旧原文，但不会绕过整请求硬上限、也从不被逐出）。结构修复（每源至多 1 次）携带被拒正文的有限摘录与验证诊断，同源重发；瞬态失败（超时/传输）不锁定来源。维护请求与普通、修复、收尾请求共享同一绝对及 segment 准入总账；每 run ≤2 轮、≤3 次维护调用、单次 30s、累计 60s 等小限额不会因 segment 续期重置。维护失败（超时/传输/无净压缩/坏结构）只降级回旧的合法请求视图，不制造新错误终态，同一确定来源不重试。`/clear` 的新 generation 使旧摘要失效。摘要可能遗漏细节；需要精确信息时应重新读取原始来源。默认 `--max-turns 32` 不变。

当前不提供 `@` 文件引用、NaturalCC provider、磁盘 transcript 或远程沙箱能力。

## 渲染模型

REPL 输出分为三个区域：

- **Stable region**：已 finalize、不会再原位改变的 turn 记录（user message、完成的 Assistant segment、最终 assistant 正文、tool 与审批审计）在 run 收口后按稳定 batch ID 一次性写入终端历史；已写入的字节不会被应用主动擦除或重写。
- **Dynamic region**：当前 turn 仍可能变化的内容——由完整有界 Assistant source 解析出的统一 Markdown/table 物理行，再由活动视窗选取最新连续行；空间不足时复用单行灰色 streaming 状态提示。这里还包含 running/denied tool、审批菜单与当前回合 `Working for` 计时。
- **Interactive region**：唯一 session footer（仅空闲时，稳定的唯一 ID，graceful exit 时 final once）、命令输入与局部交互。

事件经 delivery envelope（session generation、runId、run 内严格递增 sequence）进入 UI：重复 sequence 幂等忽略，倒退/跳号/旧 generation/同 sequence 不同 payload 只产生有界内部诊断，不更新任何 owner。`/clear` 丢弃尚未冻结的 boundary/footer 并以新 stable generation remount；**已经写入终端原生 scrollback 的历史字节无法被应用撤回**——本应用保证的是“不再主动擦除并重写已稳定历史”，不控制终端模拟器自身的 scroll-on-output 设置。渲染使用锁定的 Ink `7.0.3` 标准 renderer（`incrementalRendering: false`）+ `exitOnCtrlC: false`；稳定性来自 stable/dynamic 分区与显式 resize owner，而不是增量渲染优化；live 计时刷新为 500 ms。

流式展示链只有一个来源方向：`AgentEvent` 先进入 message controller 的 run/generation owner，UI refresh owner 合并脏 delta，单个 streaming normalizer 保留当前受 100,000 字符硬上限约束的 source，既有 Markdown parser 和共享 table layout 生成 styled rows，`ActiveTurn` 最后才按整帧预算选取可见行，`MessageColumn` 直接消费这些行。只有最后一步可以暂时隐藏内容；可见尾窗不会反拼成 Markdown，也不会回写 AgentThread。没有计时状态变化的 event 不会触发额外 UI 刷新；普通计时 tick 或同源父组件重渲染也不会重建未变的 Markdown/table rows。代码块语言标签与正文共用同一物理行换行和帧预算。segment/terminal 立即撤销 pending refresh，`/clear`、run/generation 切换和 unmount 使旧 callback 失效；resize 用同一完整有界 source 和新 width generation 重投影。

## 工具折叠与详情查看

工具 presenter（bash、read_file、list_files、search_text、apply_patch、git_changes）默认折叠到 40 逻辑行 / 6,000 字符（先到者生效，`UiPresentationLimits` 集中管理）；被折叠的工具记录保存在**有界详情存储**里，空闲时按 `Ctrl+O` 用独立面板翻阅。普通已完成 Assistant 文本不创建折叠摘要或重复详情副本。两种工具截断状态严格区分：

- **viewCollapsed**（可恢复）：折叠只是默认摘要；完整有界记录留在详情存储（单条 ≤ 100,000 字符且 ≤ 4,000 行双上限，行分隔计入成本；整存储 ≤ 1,000,000 字符，按最旧 turn 先逐出），`Ctrl+O` 可翻阅。
- **sourceTruncated**（不可恢复）：上游 canonical 输出被截断（工具输出预算、display 构建上限或详情存储自身 clip），提示 `Output was truncated at the source; all retained output is shown`，无法从 UI 恢复。

对于可展开的工具记录，详情存储是 retained 行的**唯一持有者**：stable 消息只带轻量引用（计数 + 双截断态 + record ID），消息提交 owner 按 generation+recordId 一次性接纳，React 重复渲染不改变缓存事实，逐出真正释放内存。普通 Assistant 直接由其 canonical `content` 生成 stable Markdown，不进入详情存储；presenter 分派经 `ToolPresenterRegistry`：未知工具用受限 literal fallback，presenter 抛错只降级不改写工具成功事实，重复注册在装配期失败。

详情查看器是 interactive region 的动态 overlay：`Ctrl+O` 打开最近一条折叠记录，`PgUp/PgDn/Home/End` 翻页，`Tab/Shift+Tab` 切换记录，`Esc` 或 `Ctrl+O` 关闭；仅在空闲时可用，运行中的 turn、审批菜单或 clear 事务拥有键盘时不会打开，已打开的会被自动关闭。查看器按**视觉行**（wrap 后）分页，长逻辑行可翻页到达而非永久横向截断；面板行数计入整帧 ledger，打开/翻页/关闭从不重写已提交的 stable 历史。输入行的 value/cursor/completion 状态在打开前快照、重挂载时同步恢复（abc 光标在 a 后，开/关后输入 X 得到 aXbc）；同一 stdin chunk 内的 `Ctrl+O`+文字不会泄漏进输入行。`/clear` 的新 generation 会使存储全部失效。

### apply_patch diff 投影

`apply_patch` 成功时，工具 executor 额外产出**中性 display facts**（`ToolExecutionResult.display`，成功路径独有）：带旧/新行号的上下文/增/删记录。UI 以单一右对齐灰色行号列（删除行取旧行号，新增/上下文取新行号）+ 单字符标记列（`+`/`-`/空格/`⋮`）投影为 `行号 │ 标记 源文本`，行号列与源文本之间只有这一个 UI 分隔空格。每个修改组默认只保留前后各 1 行上下文，隐藏区域用唯一一个 `⋮` 行表示；正文硬预算为 40 个物理可见行与共享 6,000 字符（先到者生效）。新增/删除行使用绿/红前景+深色底色，行号、`⋮` 与标题为中性灰；`NO_COLOR` 下投影与纯文本完全一致。

`display` 是 UI 本地、一次性的展示事实：它搭 UI-facing 的 tool_result 事件流（与输出同样受 `showToolResults`/错误可见性约束）进入 presenter，但 canonical `content`（回填模型与 AgentThread）不变，`display` 不会进入 ModelToolResultMessage、AgentThread 或模型 token。任何 display facts 结构漂移（未知 kind、缺失该 kind 必需行号、超过 executor 上限的记录数、`facts:null`）都会严格解析失败并降级回 通用表格展示。diff 行号来自成功执行使用的 canonical hunk 游标：removed 投影**原始**旧行号（跨 hunk 保留原始行身份），added/context 投影最终新行号。物理行/字符账本由真实内容宽度驱动：grapheme-safe 软换行、续行空 gutter、标题/`⋮`/notice 同一计费行集，Ink 渲染、NO_COLOR plainText 与 整帧布局账本 消费同一事实；颜色层只装饰——行号/分隔线灰，亮红绿与深底只覆盖 marker/分隔空格/正文，SGR 逐物理行闭合。

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
```

查看版本或 CLI 帮助：

```bash
bun run naturalcc -- --version
bun run naturalcc -- --help
```
