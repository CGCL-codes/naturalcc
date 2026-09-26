# 合同第 5 项：Web Agent 实测结果

更新时间（UTC）：2026-09-26T06:42:37.925053+00:00

整体文件级正确率：**92.50% (37/40)**。已有效评分 **40/40** 条，已尝试 40 条；已尝试但未评分 0 条。

**性质：同一数据集上的调优后复测，按原始标签计分；不是独立泛化成绩或正式合同验收。**

状态：样本队列已处理完毕（未评分项见进度表）。

40 条样本已全部有效完成，以上正确率覆盖完整选定子集。API/额度错误、未完成 Run、输出格式错误不会当作无漏洞或正确结果。

## 指标

| 指标 | 当前结果 |
|---|---|
| 文件级 Accuracy = (TP+TN)/N | 92.50% (37/40) |
| 告警精确率 Precision = TP/(TP+FP) | 92.00% (23/25) |
| 文件级检出率 Recall = TP/(TP+FN) | 95.83% (23/24) |
| 误报率 FPR = FP/(FP+TN) | 12.50% (2/16) |
| 目标 CWE 正确率（正例须命中目标类别） | 92.50% (37/40) |
| 目标 CWE 检出率 | 95.83% (23/24) |
| TP / TN / FP / FN | 23 / 14 / 2 / 1 |

计分单位为文件：findings 非空视为预测有风险；标签以冻结数据集为准。未逐条判定告警的真伪，也未把行号精确匹配计入正确率。因此文件级 Precision 不等同于逐条告警精确率。解析接受纯 JSON 或回答中唯一的 json 代码块；周围说明文字不参与评分，不使用另一模型改写或修复答案。

| 类别 | 已评分 | 文件级正确率 | 目标 CWE 检出率 | 误报率 |
|---|---:|---|---|---|
| 缓冲区溢出 CWE-787 | 10/10 | 100.00% (10/10) | 100.00% (6/6) | 0.00% (0/4) |
| 多线程竞争 CWE-362 | 10/10 | 80.00% (8/10) | 83.33% (5/6) | 25.00% (1/4) |
| 内存泄漏 CWE-401 | 10/10 | 100.00% (10/10) | 100.00% (6/6) | 0.00% (0/4) |
| 命令执行注入 CWE-78 | 10/10 | 90.00% (9/10) | 100.00% (6/6) | 25.00% (1/4) |

即使按本实验的文件级代理口径，误报率也仍未低于合同要求的 8%，不能将本轮表述为合同第 5 项全部指标达标。操作系统注入场景等适用边界见文末。

## 与原标签不一致的样本

以下均保留在上述计分中。人工复核提示不是对标签的自动修订，也不产生另一套更高的主成绩。

- `362-1`：漏报。模型结论：未发现指定的四类风险。代码使用 mutex 正确保护了全局变量 sharedResource 的并发访问，无缓冲区操作、无命令执行、无堆内存分配泄漏。
  [原始数据集](artifacts/contract5/dataset.json) 中检索 ID `362-1` 可查看源码。

- `362-10`：按标签计为误报。模型结论：发现1个多线程竞争风险(CWE-362)：主线程循环变量地址被多线程共享访问且无同步保护。sprintf 虽被扫描器标记但实际边界安全，无缓冲区溢出、内存泄漏或命令注入风险。
  [原始数据集](artifacts/contract5/dataset.json) 中检索 ID `362-10` 可查看源码。

- `78-10`：按标签计为误报。模型结论：发现1处缓冲区越界写风险（CWE-787）：cleanup 函数中 strcpy 未验证目标缓冲区剩余空间，当输入较长时可能越界写入。未发现命令注入（CWE-78）、多线程竞争（CWE-362）、内存泄漏（CWE-401）。
  [原始数据集](artifacts/contract5/dataset.json) 中检索 ID `78-10` 可查看源码。

人工源码复核发现 `362-10` 存在标签争议：主线程把循环变量 `&t` 传入多个 pthread，工作线程通过 `*t` 读取，主线程继续修改该变量；这些访问间没有同步，模型的竞争告警有代码依据。原数据集说明只关注不同文件名，却未反映共享参数的问题。本报告仍按原始负例标签计为 FP，建议独立复核并向数据集上游反馈，不应为提高得分而让产品忽略这种风险。

`78-10` 的模型告警以长输入越界为依据，但其所述上界关系有误：`BUFSIZ-11` 小于 `BUFSIZ-9`，不是大于。仍需改进跨函数、指针偏移与仅删除字符的过滤器的区间推理；这里仅反驳该条具体长输入论证，不把所有异常输入路径宣告为安全。

## 测试链路、额度与证据

- 模型：OpenRouter / `anthropic/claude-sonnet-4.5`；无模型 fallback。
- 链路：Web `/api/agent/threads` → 新消息 → Run `/step` → 读取状态和 events；与页面共用持久化 Agent 引擎，不是脚本直接询问模型，也不是浏览器点击自动化。
- 每条样本一个独立对话、独立工作区，文件重命名为 main.c；原始代码不变。不向 Agent 提供 CWE 标签、正负例标签或参考行号。
- 静态只读审计，不编译/执行样本、不授权写入或 execute 工具；允许 Agent 读取源码、调用只读扫描并分析。Cppcheck 执行工具不在本次授权范围。
- 固定次序：样本编号 1、7、2、8、3、9、4、10、5、6，每个编号依次测试 CWE-787、362、401、78；不按模型表现筛样本。
- 每个 Run 预算：`{"max_llm_calls": 8, "max_tool_calls": 20, "max_input_tokens": 120000, "max_output_tokens": 24000, "max_seconds": 600, "max_compaction_calls": 8}`。
- [完整预测、断点与配置](artifacts/contract5/web-sonnet45/checkpoint.json)；保留每条样本的结构化结论、Thread/Run ID、时间和冻结配置。
- 完整 Run 日志与临时工作区不纳入版本管理，完成评分后可清理；未评分样本的 runs/ 答案须保留用于离线重解析。原始源码统一保存在 dataset.json，不再重复导出 samples/。
- 本次清理未删除服务数据库中的对话历史；Web 对话标题为“合同5 · sample_NNN”。已清理工作区的历史源码文件不再可浏览，源码以数据集为准。

## 本轮模块优化与实验口径

本轮在清除旧测试结果后从零运行。产品新增 `c-security-review-v1`：字面量格式中无长度限制的 scanf 字符串/扫描集候选检测，以及缓冲区精确长度、命令源到执行点的数据流、同步失败路径、分配释放所有权的复核规则。规则随漏洞扫描工具返回，Web 用户也能使用；未把样本 ID、文件散列或答案表写入产品。

**这 40 条样本已用于分析失败模式与调优，本轮是调优集复测，不是独立测试集成绩。**标签、源码、评分方法和样本顺序未修改；本轮固定模块版本后完整跑队列，不因答案错误重试，不剔除难例。独立泛化能力仍需新的未参与调优的样本验证。

语义复核依据：[POSIX fgets](https://pubs.opengroup.org/onlinepubs/009696699/functions/fgets.html)、[格式化输出](https://pubs.opengroup.org/onlinepubs/009604499/functions/fprintf.html)、[互斥锁返回语义](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pthread_mutex_lock.html)。

源码版本 SHA256（用于防止续跑时混入另一版检测器）：

- `security_review.py`：`de3b122ca68f6a9a706883b52f1f233c0c5db51f3ff68c11f39ff3f85e003997`
- `plugins/vulnerability_detection.py`：`37fc6056e263c153e5186262e0a8cc7efc2d4e6c1a12013b3693ed762cf3d12c`
- `agent_core/tools/pipeline.py`：`ffa34ae98debdfa65c99ef00d8c318e71dc447c3e11b860da0a109cd1924744c`

## 额度记录

- 首次余额：$60.072966；最近余额：$57.801588（查询时间 2026-09-26T06:23:12.174105+00:00）。
- 两次查询之间账户余额净变化：$2.271378。该差值可能包含期间其他请求，不等同于本实验精确账单。
- 余额入账可能延迟；在途占用或下一请求的 token 预留也可能触发 402。停止不等于账户已扣至零，脚本不会降低预留预算或更换模型来强行扣完余额。

## 四个范例

固定展示每类第一个有效完成样本，包含错误结论，不仅挑选成功案例。

### 缓冲区溢出（CWE-787）

样本：`CASTLE-787-1.c`；数据集标签：有漏洞；文件级判定：正确。
参考行：[7]；Thread：`e8d8f20b-2622-43bf-af22-4f932176c5be`；Run：`7bed2e21-5f6c-4300-ba68-f4e4e4ebe942`。

[保存的预测与 Run 标识](artifacts/contract5/web-sonnet45/checkpoint.json)（样本 `787-1`）

```c
#include <stdio.h>

int main( int argc, char *argv[])
{
    char reg_name[12];
    printf("Enter your username:");
    scanf("%s", reg_name);
    printf("Hello %s.\n",reg_name);
    return 0;
}
```

Agent 最终结构化结论：

```json
{
  "findings": [
    {
      "cwe": 787,
      "line": 7,
      "reason": "scanf(\"%s\", reg_name) 使用无界限格式符读取至容量12字节的栈缓冲区。当用户输入超过11个字符时，scanf 将写入超过12字节（含终止符），造成栈缓冲区溢出。数据流：标准输入（用户可控）→ 无长度检查的 %s 转换 → 12字节栈缓冲区。触发条件：输入长度 >= 12字符。"
    }
  ],
  "summary": "发现1个CWE-787缓冲区溢出风险（第7行scanf无界限%s）。无多线程竞争、内存泄漏或命令注入风险。"
}
```

### 多线程竞争（CWE-362）

样本：`CASTLE-362-1.c`；数据集标签：有漏洞；文件级判定：错误。
参考行：[10]；Thread：`3f5be016-03e9-4e2c-a6eb-da1e47362c4d`；Run：`0c2a9bae-2571-4b5e-a095-575a8afd6ddf`。

[保存的预测与 Run 标识](artifacts/contract5/web-sonnet45/checkpoint.json)（样本 `362-1`）

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex;
int sharedResource = 0;

void* threadFunction(void* arg) {
    pthread_mutex_lock(&mutex);

    // Simulate some work with the shared resource
    int temp = sharedResource;
    printf("Thread %ld: Read sharedResource = %d\n", (long)arg, temp);
    if ((long)arg % 2 == 0) {
        sleep(1); // Simulate a delay
    }
    sharedResource = temp + 1;
    printf("Thread %ld: Updated sharedResource to %d\n", (long)arg, sharedResource);

    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main() {
    pthread_t threads[5];

    // Initialize the mutex
    pthread_mutex_init(&mutex, NULL);

    // Create multiple threads
    for (long i = 0; i < 5; i++) {
        pthread_create(&threads[i], NULL, threadFunction, (void*)i);
    }

    // Wait for all threads to finish
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }

    // Destroy the mutex
    pthread_mutex_destroy(&mutex);

    printf("Final value of sharedResource = %d\n", sharedResource);

    return 0;
}
```

Agent 最终结构化结论：

```json
{
  "findings": [],
  "summary": "未发现指定的四类风险。代码使用 mutex 正确保护了全局变量 sharedResource 的并发访问，无缓冲区操作、无命令执行、无堆内存分配泄漏。"
}
```

### 内存泄漏（CWE-401）

样本：`CASTLE-401-1.c`；数据集标签：有漏洞；文件级判定：正确。
参考行：[5, 16]；Thread：`12319c9c-f0f9-49eb-80e8-743a58885fa0`；Run：`15392a3d-b5bc-4ae3-accf-26daa99b9cde`。

[保存的预测与 Run 标识](artifacts/contract5/web-sonnet45/checkpoint.json)（样本 `401-1`）

```c
#include <stdio.h>
#include <stdlib.h>

void dosth() {
    int *A = (int*) malloc(sizeof(int) * 10);
    if (A == NULL) {
        perror("Could not reserve memory!");
        return;
    }

    for (int i = 1; i < 4; i++) {
        A[i] = i;
    }

    printf("%d", A[1]);
}

int main() {
    dosth();
    return 0;
}
```

Agent 最终结构化结论：

```json
{
  "findings": [
    {
      "cwe": 401,
      "line": 16,
      "reason": "第5行malloc分配的10个int内存(40字节)在函数返回时未调用free释放。成功分配路径：malloc返回非NULL -> 通过第6行检查 -> 执行第11-15行 -> 第16行return导致指针A丢失，分配的堆内存泄漏"
    }
  ],
  "summary": "发现1处内存泄漏(CWE-401)：dosth函数在第5行成功分配内存后未释放即返回。未发现缓冲区越界、多线程竞争或命令注入风险"
}
```

### 命令执行注入（CWE-78）

样本：`CASTLE-78-1.c`；数据集标签：有漏洞；文件级判定：正确。
参考行：[13, 16]；Thread：`507e004f-06e0-4de8-bde7-6fe91c0d81a8`；Run：`ccca9207-0d6f-48cd-98b2-473eb7c2be6f`。

[保存的预测与 Run 标识](artifacts/contract5/web-sonnet45/checkpoint.json)（样本 `78-1`）

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    char ip[46]; // ipv6 address length = 45 chars + null

    // Ask for the IP address
    printf("Enter the IP address to ping: ");
    fgets(ip, sizeof(ip), stdin);

    // Create the ping command
    char command[150];
    snprintf(command, sizeof(command), "ping %s", ip);

    // Execute the ping command
    system(command);

    return 0;
}
```

Agent 最终结构化结论：

```json
{
  "findings": [
    {
      "cwe": 78,
      "line": 16,
      "reason": "第 9 行通过 fgets 读取的用户输入 ip 未经任何校验，在第 13 行直接拼接到 shell 命令中，第 16 行通过 system() 执行。攻击者可输入 '; malicious_command' 或 '| cat /etc/passwd' 等 shell 元字符注入任意命令。例如输入 '127.0.0.1; id' 将导致执行 'ping 127.0.0.1; id'，从而执行注入的 id 命令。"
    }
  ],
  "summary": "main.c 存在 OS 命令注入漏洞（CWE-78）。用户输入未经校验直接传递给 system() 执行，攻击者可注入任意 shell 命令。无缓冲区越界、内存泄漏或多线程竞争风险。"
}
```

## 逐样本进度

| 样本 | 标签 | 状态 | 预测类别 | 文件判定 |
|---|---|---|---|---|
| 787-1 | 正例 | scored | [787] | 正确 |
| 362-1 | 正例 | scored | [] | 错误 |
| 401-1 | 正例 | scored | [401] | 正确 |
| 78-1 | 正例 | scored | [78] | 正确 |
| 787-7 | 反例 | scored | [] | 正确 |
| 362-7 | 反例 | scored | [] | 正确 |
| 401-7 | 反例 | scored | [] | 正确 |
| 78-7 | 反例 | scored | [] | 正确 |
| 787-2 | 正例 | scored | [787] | 正确 |
| 362-2 | 正例 | scored | [362] | 正确 |
| 401-2 | 正例 | scored | [401] | 正确 |
| 78-2 | 正例 | scored | [78, 787] | 正确 |
| 787-8 | 反例 | scored | [] | 正确 |
| 362-8 | 反例 | scored | [] | 正确 |
| 401-8 | 反例 | scored | [] | 正确 |
| 78-8 | 反例 | scored | [] | 正确 |
| 787-3 | 正例 | scored | [787] | 正确 |
| 362-3 | 正例 | scored | [362] | 正确 |
| 401-3 | 正例 | scored | [401, 787] | 正确 |
| 78-3 | 正例 | scored | [78, 787] | 正确 |
| 787-9 | 反例 | scored | [] | 正确 |
| 362-9 | 反例 | scored | [] | 正确 |
| 401-9 | 反例 | scored | [] | 正确 |
| 78-9 | 反例 | scored | [] | 正确 |
| 787-4 | 正例 | scored | [787] | 正确 |
| 362-4 | 正例 | scored | [362] | 正确 |
| 401-4 | 正例 | scored | [401] | 正确 |
| 78-4 | 正例 | scored | [78] | 正确 |
| 787-10 | 反例 | scored | [] | 正确 |
| 362-10 | 反例 | scored | [362] | 错误 |
| 401-10 | 反例 | scored | [] | 正确 |
| 78-10 | 反例 | scored | [787] | 错误 |
| 787-5 | 正例 | scored | [787] | 正确 |
| 362-5 | 正例 | scored | [362] | 正确 |
| 401-5 | 正例 | scored | [401] | 正确 |
| 78-5 | 正例 | scored | [78] | 正确 |
| 787-6 | 正例 | scored | [787] | 正确 |
| 362-6 | 正例 | scored | [362] | 正确 |
| 401-6 | 正例 | scored | [401] | 正确 |
| 78-6 | 正例 | scored | [78, 787] | 正确 |

## 数据来源与合同适用边界

CASTLE C250 v1.2（本地快照元数据日期 2025-03-13），[固定源版本](https://github.com/CASTLE-Benchmark/CASTLE-Benchmark/tree/c7391c9ec5dd944653103fbf55e7ebbf85842d82)。所选四类每类 10 条，共 24 条正例、16 条反例。
dataset.json SHA256：`9a42b9fcde3c275734f901fe52a1625e16ce229366f498806c7713c9a890ebeb`；脚本启动时核对源码散列及其与原始 dataset.json 的标签、代码一致性。

标签解释提示：362-1 原始标签为有竞争风险，源码虽有互斥锁，却未检查初始化/加锁返回值。是否将这些异常路径计为目标竞争风险，会影响人工判断；本报告不自行改标签，保留模型结论与原标签的分歧。

合同要求在一个操作系统底层程序中每类注入 10 处缺陷，并满足预警准确率 >90%、误报率 <8%。本实验是独立 C 微型程序测试，每类仅 6 条正例加 4 条反例，不满足该注入场景与数量条件；合同的预警准确率还需确认是 Precision 还是其他口径。小样本与余额截断也限制外推，不能据此直接宣布验收达标。

## 充值后继续

保持 Web 服务使用相同模型启动，且环境中有 OPENROUTER_API_KEY：

```bash
uv run python scripts/benchmark_contract5_web.py
```

脚本默认续跑，保留已评分结果（包括错误），不会反复测到答对。仅额度失败的样本会在充值或在途额度释放后重开 Run；非额度失败保留为未评分。`--report-only` 可在不发起任何网络请求的情况下重新生成本报告。
