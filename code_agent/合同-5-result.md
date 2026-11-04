# 合同第 5 项：操作系统底层程序高危代码检测实测结果

更新时间（UTC）：2026-11-04T12:11:02.962449+00:00

> **本轮不是合同成绩。** 使用的是 `deepseek/deepseek-chat`，并非合同要求的 Claude/GPT 级模型，本轮仅用于验证数据集与测试链路（提示词、读取覆盖、JSON 解析、site 计分、报告生成）是否正确。合同成绩须用 `anthropic/claude-sonnet-4.5` 或更强的 Claude/GPT 模型重跑，结果写入 `web-<model>/` 独立目录。

数据集：RT-Thread v5.2.1（提交 `97893c004c65`）子集，332 个文件、源码 108524 行；待检文件 34 个、合计 29821 行；每类注入缺陷 10 处，共 40 处缺陷 + 43 处安全对照。数据集与真值见 `artifacts/contract5/rtthread/dataset/`，构建方法见 `scripts/build_contract5_dataset.py`。

模型：OpenRouter / `deepseek/deepseek-chat`；状态：待检文件队列已处理完毕（未评分项见进度表）；已尝试 33/34 个文件，已有效评分 30/34 个。

**这是部分文件的阶段性结果，不是完整数据集的得分。**API/额度错误、未完成 Run、输出格式错误不计入 TP/TN/FP/FN。

检测系统的静态层可单独评估（不调模型、不联网）：`uv run python scripts/measure_contract5_static.py`；其成绩与本轮端到端成绩的差异分析见 `合同-5-数据集说明.md` 第 5.2、5.3 节。

## 指标（site 级：每处注入缺陷 / 安全对照函数为一个样本）

| 指标 | 公式 | 当前结果 |
|---|---|---|
| 预警准确率（合同口径） | TP/(TP+FP) | 100.00% (3/3) |
| 误报率（合同口径） | FP/(FP+TN) | 0.00% (0/35) |
| 检出率 | TP/(TP+FN) | 10.71% (3/28) |
| 正确率 | (TP+TN)/N | 60.32% (38/63) |
| TP / FN / FP / TN | — | 3 / 25 / 0 / 35 |

合同阈值：预警准确率 > 90%，误报率 < 8%；甲方期望数据集难度使正确率落在 85%–90%。本脚本只输出原始数值，不输出达标结论。

| 类别 | TP | FN | FP | TN | 检出率 | 预警准确率 | 误报率 | 正确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 缓冲区溢出 | 0 | 7 | 0 | 8 | 0.00% | N/A | 0.00% | 53.33% |
| 多线程竞争 | 0 | 6 | 0 | 9 | 0.00% | N/A | 0.00% | 60.00% |
| 内存泄漏 | 0 | 5 | 0 | 7 | 0.00% | N/A | 0.00% | 58.33% |
| 命令执行漏洞 | 3 | 7 | 0 | 11 | 30.00% | 100.00% | 0.00% | 66.67% |
| 总体 | 3 | 25 | 0 | 35 | 10.71% | 100.00% | 0.00% | 60.32% |

| 缺陷难度 | 缺陷数 | 已评分 | 检出 | 检出率 |
|---|---:|---:|---:|---:|
| easy | 16 | 12 | 2 | 16.67% |
| medium | 16 | 10 | 0 | 0.00% |
| hard | 8 | 6 | 1 | 16.67% |

## 文件级判定

文件级：findings 非空视为预测有漏洞；真值 verdict=vulnerable 的文件至少含一处注入缺陷，clean 文件不含缺陷（可含安全对照）。

| 指标 | 当前结果 |
|---|---|
| 文件级正确率 | 50.00% (15/30) |
| 文件级误报率（clean 文件被报有漏洞） | 7.69% (1/13) |
| 文件级 TP / FN / FP / TN | 3 / 14 / 1 / 12 |

## 与真值不一致的 site

以下均保留在上述计分中；人工复核意见不修改标签，也不产生另一套更高的成绩。

| site | 文件 | 函数 | 类别 | 难度 | 判定 | 真值说明 |
|---|---|---|---|---|---|---|
| `BOF-09` | `components/dfs/dfs_v2/src/dfs.c` | `dfs_normalize_path` | 缓冲区溢出 | hard | 漏报 FN | 为拼接 "目录/文件" 分配的 path_len 少算 1 字节（应为 +2：一个 '/' 加结尾 '\0'），而随后的 rt_snprintf 仍以 strlen(directory)+strlen(filename)+2 为长度上限写入该堆缓冲，造成 1 字节堆越界写。需要对照分配长度与 snprintf 的长度参 |
| `CE-06` | `components/dfs/dfs_v2/src/dfs_fs.c` | `dfs_remount` | 命令执行漏洞 | hard | 漏报 FN | fullpath 来自调用方给的挂载路径，data 是重挂载时上层透传的文件系统私有数据块（由其它子系统填充），两者都不是 shell 操作者输入；这里跨子系统地把它们拼进命令行交给 system，污点要跨多层调用才能看清。 |
| `CE-04` | `components/dfs/dfs_v2/src/dfs_fs.c` | `dfs_mount` | 命令执行漏洞 | medium | 漏报 FN | device_name 与规整后的挂载路径 fullpath 由上层挂载请求（可能来自自动挂载/热插拔等子系统）传入，而非 shell 操作者当场键入；此处未校验就拼成命令行交给 system。 |
| `CE-05` | `components/dfs/dfs_v2/src/dfs_fs.c` | `dfs_umount` | 命令执行漏洞 | medium | 漏报 FN | mnt->fullpath 是挂载表中保存的路径，由此前的挂载过程写入，属于其它子系统提供的数据；卸载时把它拼进命令行交给 system 执行。 |
| `CE-07` | `components/dfs/dfs_v2/src/dfs_fs.c` | `dfs_mkfs` | 命令执行漏洞 | easy | 漏报 FN | fs_name 与 device_name 是 mkfs 请求参数，格式化设备时可能由脚本或上层自动流程传入，而非操作者当场键入；此处直接拼成命令行交给 system。 |
| `CE-08` | `components/dfs/dfs_v2/src/dfs_fs.c` | `dfs_statfs` | 命令执行漏洞 | easy | 漏报 FN | fullpath 是规整后的查询路径，由上层调用（可能是服务/脚本）给出，不是 shell 操作者输入；statfs 把它拼进命令行交给 system。 |
| `LEAK-09` | `components/dfs/dfs_v2/src/dfs_mnt.c` | `dfs_mnt_unref` | 内存泄漏 | hard | 漏报 FN | 引用计数归零释放 mnt 时，dfs_mnt_create() 中 rt_strdup 的 fullpath 不再 rt_free，只置空指针。 |
| `LEAK-10` | `components/dfs/dfs_v2/src/dfs_pcache.c` | `dfs_aspace_release` | 内存泄漏 | hard | 漏报 FN | 页缓存地址空间销毁时释放了 fullpath 却只置空 pathname，_dfs_aspace_create() 中 rt_strdup 的 pathname 泄漏。 |
| `LEAK-08` | `components/dfs/dfs_v2/src/dfs_seq_file.c` | `dfs_seq_release` | 内存泄漏 | medium | 漏报 FN | 释放 seq 结构前只把 seq->buf 置空，dfs_seq_read() 中 dfs_seq_alloc() 分配的读缓冲区不再释放。 |
| `BOF-05` | `components/drivers/ipc/ringbuffer.c` | `rt_ringbuffer_put` | 缓冲区溢出 | medium | 漏报 FN | rt_ringbuffer_put 原本在剩余空间 size 小于请求长度 length 时把 length 截断到 size；删除该截断后，后续 rt_memcpy 按调用方 length 写入，可超过环形缓冲剩余容量造成越界写。 |
| `RACE-05` | `components/drivers/ipc/workqueue.c` | `_workqueue_cancel_work` | 多线程竞争 | easy | 漏报 FN | 共享状态：work->list 链表与 work->flags/workqueue 字段。取消时的链表摘除与字段清零被移出锁，与工作线程取节点、其它提交者并发。 |
| `LEAK-04` | `components/drivers/ipc/workqueue.c` | `rt_workqueue_create` | 内存泄漏 | easy | 漏报 FN | 工作线程创建失败的错误路径不再 RT_KERNEL_FREE(queue)，已分配的 rt_workqueue 结构丢失。 |
| `RACE-04` | `components/drivers/ipc/workqueue.c` | `rt_workqueue_urgent_work` | 多线程竞争 | easy | 漏报 FN | 共享状态：工作队列 queue->work_list 链表。链表摘除/插入被移到自旋锁之外，与工作线程 _workqueue_thread_entry 持锁取节点并发。 |
| `BOF-04` | `components/drivers/serial/dev_serial.c` | `_serial_int_rx` | 缓冲区溢出 | easy | 漏报 FN | 软件接收 FIFO 的读指针回绕判断由 >= 改成 >，当 get_index 自增到等于 bufsz 时不再归零，下一轮用 rx_fifo->buffer[get_index] 读取时越界访问 buffer[bufsz]。 |
| `CE-10` | `components/finsh/msh.c` | `msh_exec_module` | 命令执行漏洞 | medium | 漏报 FN | pg_name 是在文件系统 /bin 路径中检索、补全得到的模块文件路径，取决于磁盘内容；加载模块前把它拼进命令行交给 system 执行。 |
| `BOF-08` | `components/finsh/msh.c` | `msh_auto_complete_path` | 缓冲区溢出 | medium | 漏报 FN | 补全路径时把候选目录项名 full_path 以其完整长度拷回 index（指向调用方 path 缓冲的中段），而非受控的公共前缀长度 min_length；当候选名较长而 path 剩余空间不足时越界写命令行缓冲。 |
| `CE-02` | `components/finsh/msh_file.c` | `directory_setattr` | 命令执行漏洞 | medium | 漏报 FN | full_path 同样来自 readdir 的目录项名，是外部文件系统数据；递归设置属性时把它拼进命令行并经 msh_exec 执行，等于让磁盘上的文件名决定要运行的 shell 命令。 |
| `BOF-03` | `components/finsh/shell.c` | `finsh_get_prompt` | 缓冲区溢出 | easy | 漏报 FN | finsh_get_prompt 把用户通过 finsh_set_prompt 设置的自定义提示符用无界 strcpy 拷入固定缓冲 finsh_prompt[RT_CONSOLEBUF_SIZE+1]，自定义串长度不受限，可越界写该栈/静态缓冲。 |
| `BOF-10` | `components/finsh/shell.c` | `finsh_thread_entry` | 缓冲区溢出 | hard | 漏报 FN | 命令行输入缓冲 shell->line[FINSH_CMD_SIZE+1] 的两处越界保护被同时放宽：前置丢弃判断由 >= 改为 >，尾部清行判断由 >= 改为 >= FINSH_CMD_SIZE+1，使 line_position 可达到 FINSH_CMD_SIZE+1，随后以该下标写字符时越界。需跨两处保护联立分 |
| `RACE-08` | `src/memheap.c` | `rt_memheap_free` | 多线程竞争 | medium | 漏报 FN | 共享状态：memheap 块头 magic 与 heap->available_size。标记空闲与可用字节累加被移到获取 heap->lock 之前，脱离互斥保护。 |
| `LEAK-03` | `src/mempool.c` | `rt_mp_create` | 内存泄漏 | medium | 漏报 FN | 乘法回绕检查放在 rt_malloc 之后：检查失败时删除对象并返回，但 mp->start_address 指向的堆内存未 rt_free。 |
| `RACE-02` | `src/mempool.c` | `rt_mp_free` | 多线程竞争 | easy | 漏报 FN | 共享状态：内存池空闲块计数 mp->block_free_count。自增被移到加锁之前，与 rt_mp_alloc 中锁内的 block_free_count-- 构成非原子读改写。 |
| `BOF-02` | `src/object.c` | `rt_object_get_name` | 缓冲区溢出 | easy | 漏报 FN | rt_object_get_name 收到调用方缓冲 name 及其容量 name_size，却用不带长度的 rt_strcpy 拷贝对象名，忽略了 name_size，当对象名长度接近 RT_NAME_MAX 而调用方缓冲较小时越界写。 |
| `RACE-06` | `src/thread.c` | `rt_thread_wakeup_set` | 多线程竞争 | medium | 漏报 FN | 共享状态：thread->wakeup_handle 的 func 与 user_data。func 的写入被移出调度锁，而 rt_thread_wakeup 在调度锁内读取 func 并清空后回调。 |
| `RACE-01` | `src/timer.c` | `rt_timer_stop` | 多线程竞争 | easy | 漏报 FN | 共享状态：定时器所在的全局 timer skip-list 与 timer->parent.flag。自旋锁被提前释放，_timer_remove() 的链表摘除和标志清除落在锁外；另一核的 rt_timer_start/rt_timer_check 可同时操作同一链表。 |

范围外告警（不落在任何注入/对照函数上、不计入 site 级指标）：10 条；明细见 checkpoint。

## 逐文件进度

| 文件 | 行数 | 真值 | 缺陷/对照 | 状态 | 预测告警数 | site TP/FN/FP/TN | 文件判定 | Run |
|---|---:|---|---|---|---:|---|---|---|
| `components/dfs/dfs_v2/src/dfs.c` | 1047 | 有缺陷 | 1/1 | scored | 4 | 0/1/0/1 | 正确 | `a04d2f6f` |
| `components/dfs/dfs_v2/src/dfs_dentry.c` | 391 | 无缺陷 | 0/1 | scored | 0 | 0/0/0/1 | 正确 | `fda4acf2` |
| `components/dfs/dfs_v2/src/dfs_file.c` | 2427 | 有缺陷 | 3/2 | failed | — | — | — | `22f1429c` |
| `components/dfs/dfs_v2/src/dfs_posix.c` | 1468 | 无缺陷 | 0/1 | scored | 4 | 0/0/0/1 | 错误 | `fbeb2799` |
| `components/dfs/dfs_v2/src/dfs_fs.c` | 600 | 有缺陷 | 5/1 | scored | 0 | 0/5/0/1 | 错误 | `5c08c03f` |
| `components/dfs/dfs_v2/src/dfs_vnode.c` | 147 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `cfe84b58` |
| `components/dfs/dfs_v2/src/dfs_mnt.c` | 473 | 有缺陷 | 1/1 | scored | 0 | 0/1/0/1 | 错误 | `97c01259` |
| `components/drivers/ipc/completion_up.c` | 226 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `f703e5e3` |
| `components/dfs/dfs_v2/src/dfs_pcache.c` | 1536 | 有缺陷 | 1/0 | scored | 0 | 0/1/0/0 | 错误 | `87ba30d4` |
| `components/drivers/ipc/dataqueue.c` | 459 | 无缺陷 | 0/1 | scored | 0 | 0/0/0/1 | 正确 | `ff07a053` |
| `components/dfs/dfs_v2/src/dfs_seq_file.c` | 402 | 有缺陷 | 1/1 | scored | 0 | 0/1/0/1 | 错误 | `01a4ef8f` |
| `components/drivers/ipc/pipe.c` | 803 | 无缺陷 | 0/2 | scored | 0 | 0/0/0/2 | 正确 | `eda9b7d3` |
| `components/drivers/ipc/ringbuffer.c` | 466 | 有缺陷 | 1/3 | scored | 0 | 0/1/0/3 | 错误 | `250c6201` |
| `components/drivers/ipc/ringblk_buf.c` | 604 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `ae94ed84` |
| `components/drivers/ipc/workqueue.c` | 469 | 有缺陷 | 3/1 | scored | 0 | 0/3/0/1 | 错误 | `efd2c52b` |
| `components/finsh/msh_parse.c` | 96 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `f1c7241b` |
| `components/drivers/serial/dev_serial.c` | 1575 | 有缺陷 | 1/1 | scored | 0 | 0/1/0/1 | 错误 | `cc3b7ac4` |
| `components/libc/compilers/common/cstdlib.c` | 197 | 无缺陷 | 0/1 | scored | 0 | 0/0/0/1 | 正确 | `918d634b` |
| `components/finsh/cmd.c` | 1131 | 有缺陷 | 1/3 | scored | 1 | 1/0/0/3 | 正确 | `3e83d342` |
| `components/utilities/ulog/ulog.c` | 1602 | 无缺陷 | 0/3 | scored | 0 | 0/0/0/3 | 正确 | `cd533903` |
| `components/finsh/msh.c` | 1018 | 有缺陷 | 2/2 | scored | 0 | 0/2/0/2 | 错误 | `4f09bf5a` |
| `src/clock.c` | 261 | 无缺陷 | 0/0 | failed | — | — | — | `6c24367c` |
| `components/finsh/msh_file.c` | 1159 | 有缺陷 | 3/5 | scored | 4 | 2/1/0/5 | 正确 | `fd883e4e` |
| `src/components.c` | 286 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `e325742a` |
| `components/finsh/shell.c` | 834 | 有缺陷 | 2/0 | scored | 0 | 0/2/0/0 | 错误 | `40f3dce7` |
| `src/irq.c` | 157 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `3577a2d3` |
| `src/ipc.c` | 4045 | 有缺陷 | 8/4 | 未测试 | — | — | — | `` |
| `src/scheduler_up.c` | 571 | 无缺陷 | 0/0 | scored | 0 | 0/0/0/0 | 正确 | `5a15df3a` |
| `src/kservice.c` | 1162 | 有缺陷 | 1/2 | unscored | — | — | — | `551754c1` |
| `src/memheap.c` | 998 | 有缺陷 | 1/0 | scored | 0 | 0/1/0/0 | 错误 | `5ad8a37d` |
| `src/mempool.c` | 419 | 有缺陷 | 2/1 | scored | 0 | 0/2/0/1 | 错误 | `bba4c9cd` |
| `src/object.c` | 775 | 有缺陷 | 1/2 | scored | 0 | 0/1/0/2 | 错误 | `54202067` |
| `src/thread.c` | 1145 | 有缺陷 | 1/2 | scored | 0 | 0/1/0/2 | 错误 | `f3c16c77` |
| `src/timer.c` | 872 | 有缺陷 | 1/2 | scored | 0 | 0/1/0/2 | 错误 | `99c44291` |

## 测试链路、额度与证据

- 模型：OpenRouter / `deepseek/deepseek-chat`；无模型 fallback。可选的 GPT/Claude 更强模型用 `--model` 指定，并在 Web 后端启动时把运行时默认模型设为同一值；不同模型的结果写入不同的 `web-<model>/` 目录，不混合。
- 链路：Web `/api/agent/threads` → 新消息 → Run `/step` → 读取状态和 events；与页面共用持久化 Agent 引擎，不是脚本直接询问模型。
- 每个待检文件一个独立对话；工作区为整个注入后的工程目录，target_files 只给该文件；不向 Agent 提供 ground truth、注入位置或正负标签，真值文件不在工作区内。
- 静态只读审计：允许读取源码与调用只读扫描工具；不编译、不执行、不授权 write/execute 工具（等待审批的工具调用一律拒绝）。
- 固定次序：有缺陷文件与无缺陷文件交替，各自按路径排序；不按模型表现筛文件。每个 Run 预算：`{"max_llm_calls": 14, "max_tool_calls": 36, "max_input_tokens": 600000, "max_output_tokens": 24000, "max_seconds": 1800, "max_compaction_calls": 8}`。
- 提示词模板：`artifacts/contract5/rtthread/prompt.md`；计分：category 相同且 function 同名或 line 落在函数范围内即命中。
- 完整预测、断点与配置：`artifacts/contract5/rtthread/run-deepseek-prompt-v1/checkpoint.json`；Run 事件日志保存在同目录 `runs/`（不入库）。
- 脚本启动时核对工程全部文件与 ground truth 的 SHA256，防止在被改动的工程上计分。

## 额度记录

- 首次余额：$21.464472；最近余额：$21.258252（查询时间 2026-11-04T12:08:01.504511+00:00）。
- 两次查询之间的差值可能包含期间其他请求，不等同于本实验精确账单；余额入账可能延迟，在途占用也可能触发 402。

## 运行与续跑

启动 Web 服务（运行时默认模型与 `--model` 一致）并导出 OPENROUTER_API_KEY 后：

```bash
uv run python scripts/benchmark_contract5_web.py                 # 全部待检文件，默认续跑
uv run python scripts/benchmark_contract5_web.py --limit 2       # 低成本冒烟：前 2 个文件
uv run python scripts/benchmark_contract5_web.py --print-prompts # 不联网，打印每个文件的提示词
uv run python scripts/benchmark_contract5_web.py --report-only   # 不联网，重新生成本报告
```

脚本保留已评分结果（包括错误），不会反复测到答对；仅额度失败的文件会在充值后重开 Run，非额度失败需加 `--retry-failed`。
