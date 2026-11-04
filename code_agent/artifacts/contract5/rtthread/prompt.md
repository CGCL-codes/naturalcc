# 合同 5 · 单文件审计提示词（命令提示文本）

脚本 `scripts/benchmark_contract5_web.py` 对 ground truth 中的每个待检文件发起一个独立 Web Agent 对话，把下面的模板中的 `{file}` 替换为该文件的工程相对路径（例如 `src/thread.c`）后作为用户消息发送；工作区为 `artifacts/contract5/rtthread/dataset/project/`，ground truth 不在工作区内。

只有 `<!-- prompt -->` 与 `<!-- /prompt -->` 之间的文本会被发送；修改提示词后 checkpoint 的协议签名会变化，不能与旧结果混合续跑。

<!-- prompt -->
请审计当前工作区（RT-Thread 内核与组件工程）中的文件 `{file}`，只检测以下四类高危风险：

- buffer_overflow（缓冲区溢出，CWE-787/120）：数组或缓冲区的越界写/读，长度校验缺失、差一错误、格式化输出无长度限制。
- data_race（多线程竞争，CWE-362）：线程之间或线程与中断之间共享的状态在锁外或无同步地被读写，包括临界区过早退出、检查与使用分离。
- memory_leak（内存泄漏，CWE-401）：rt_malloc / rt_calloc / rt_strdup / RT_KERNEL_MALLOC / rt_object_allocate 等取得的内存在某条可达路径上失去引用而未释放。
- command_execution（命令执行漏洞，CWE-78）：操作者之外来源的字符串（目录项名、文件内容、其他子系统传入的设备名/路径等）未经校验被拼进命令行并交给 msh_exec() / system() 执行。

请先用工作区读取工具完整阅读该文件。**该工具默认只返回前 20000 个字符就静默截断**，而本文件可能有十几万字符，因此调用时必须显式传入足够大的 `max_chars`（例如 `max_chars: 400000`），并检查返回数据中的 `truncated` 字段：只要它为 true，就加大 `max_chars` 重新读取，直到拿到整个文件。缺陷可能位于文件任意位置，包括最后几百行，只审开头会漏掉大部分问题。需要时可查看同工程内的头文件或被调用函数，以理解 RT-Thread 的锁、对象、内存 API 的语义。
读完文件后**必须**调用一次只读的 vulnerability_detection 工具（参数：target_files:["{file}"]、scan_scope:"targets"、scan_type:"high_risk"、severity_threshold:"low"、max_findings:1000），并在回答中逐条核对它返回的每一处告警：确认成立的写进 findings，判定不成立的在 summary 里说明理由。工具告警必须用源码证据核实，不能直接照抄；工具没有告警也不代表文件安全，仍需自行通读判断。
本次仅静态审计：不要修改文件、编译、运行程序或执行 shell，不调用需要 execute/write 授权的工具，不搜索外部资料、数据集或答案。
只报告有具体代码依据的上述四类风险；空指针、未初始化、整数溢出、风格问题等其他类型不要报告，也不要归入这四类。不要凭函数名或注释猜测，每一处都要给出数据流、边界、所有权或同步关系的证据。
每个独立问题报告一条：function 填问题所在函数的真实名称，line 填该文件中问题所在行的真实 1-based 行号，reason 写具体证据与触发条件。

最终回答只输出 JSON（可使用唯一的 json 代码块），结构为：
{"findings":[{"category":"buffer_overflow","function":"rt_kprintf","line":373,"reason":"具体证据与触发条件"}],"summary":"简要结论"}
category 只能为 buffer_overflow、data_race、memory_leak、command_execution 之一；确认没有这四类风险时 findings 为 []。以上 JSON 仅为格式示例，不代表该文件实际存在漏洞。
<!-- /prompt -->
