# 合同 3：RTOS 测试集与 Ground Truth 清单

> 本文由 [scripts/export_contract3_ground_truth.py](scripts/export_contract3_ground_truth.py) 自动生成，请勿手改；重新生成：`uv run python scripts/export_contract3_ground_truth.py`。

## 1. 概览

- **工程**：3 套固定版本的公开 RTOS 源码。每套工程的“检测文件”合计不少于 10,000 行，见第 3 节。
- **缺陷**：每套工程按类别（数组越界、字符串溢出、空指针调用）各注入一个测试文件，每个文件 40 个函数：20 个正例（真实缺陷）+ 20 个同类安全对照。上游源码本身不改动。
- **测试集**：`balanced`（平衡验收集）、`independent`（独立压力集）、`balanced-hard`（平衡验收集（加难版））、`independent-hard`（独立压力集（加难版））；每套 3 工程 × 3 类 × 40 = 360 个函数。各套的区别只在注入的写法（第 2 节）和函数顺序（加难版打乱）。
- **计分**：只有注入函数参与计分。正例的告警落在函数行范围内（加难版按函数名）记 TP，否则 FN；对照被告警记 FP，否则 TN。上游源码文件里的告警不计分。
- **机器可读 GT**：`artifacts/contract3/ground_truth/<测试集>.json`，字段见第 5 节。

| 测试集 | GT 文件 |
|---|---|
| 平衡验收集 | [ground_truth/balanced.json](artifacts/contract3/ground_truth/balanced.json) |
| 独立压力集 | [ground_truth/independent.json](artifacts/contract3/ground_truth/independent.json) |
| 平衡验收集（加难版） | [ground_truth/balanced-hard.json](artifacts/contract3/ground_truth/balanced-hard.json) |
| 独立压力集（加难版） | [ground_truth/independent-hard.json](artifacts/contract3/ground_truth/independent-hard.json) |

## 2. 缺陷写法（正例与对照）

“正例代码”与“对照代码”是注入函数的函数体；对照与正例写法相近，但实际安全。`base-N` 用于平衡集、压力集，加难版另加 `hard-N`（差一错误与看似危险的安全写法）。

### 数组越界（`array_oob`）

| 写法 | 测试点 | 正例代码 | 对照代码 |
|---|---|---|---|
| `base-0` | 常量下标越界：`values[3]` 访问 3 元素数组 | `int values[3] = {1, 2, 3};`<br>`return values[3];` | `int values[3] = {1, 2, 3};`<br>`return values[2];` |
| `base-1` | 下标经局部变量传入后越界 | `int values[3] = {1, 2, 3};`<br>`int index = 3;`<br>`return values[index];` | `int values[3] = {1, 2, 3};`<br>`int index = 2;`<br>`return values[index];` |
| `base-2` | 循环边界写成 `<=`，最后一轮越界 | `int values[3] = {1, 2, 3};`<br>`int total = 0;`<br>`for (int index = 0; index <= 3; ++index) total += values[index];`<br>`return total;` | `int values[3] = {1, 2, 3};`<br>`int total = 0;`<br>`for (int index = 0; index < 3; ++index) total += values[index];`<br>`return total;` |
| `base-3` | 下标经辅助函数 `contract3_identity` 返回后越界（跨函数） | `int values[3] = {1, 2, 3};`<br>`int index = contract3_identity(3);`<br>`return values[index];` | `int values[3] = {1, 2, 3};`<br>`int index = contract3_identity(2);`<br>`return values[index];` |
| `base-4` | 指针算术 `values + 3` 后解引用越界 | `int values[3] = {1, 2, 3};`<br>`int *cursor = values + 3;`<br>`return *cursor;` | `int values[3] = {1, 2, 3};`<br>`int *cursor = values + 2;`<br>`return *cursor;` |
| `hard-0` | 守卫写成 `> 3`（应为 `>= 3`），差一越界 | `int values[3] = {1, 2, 3};`<br>`int index = contract3_identity(3);`<br>`if (index > 3) return 0;`<br>`return values[index];` | `int values[3] = {1, 2, 3};`<br>`int index = contract3_identity(3);`<br>`if (index >= 3) return 0;`<br>`return values[index];` |
| `hard-1` | 倒序循环从下标 3 开始，差一越界 | `int values[3] = {1, 2, 3};`<br>`int total = 0;`<br>`for (int index = 3; index >= 0; --index) total += values[index];`<br>`return total;` | `int values[3] = {1, 2, 3};`<br>`int total = 0;`<br>`for (int index = 3; index-- > 0;) total += values[index];`<br>`return total;` |
| `hard-2` | 用 `sizeof` 算出的元素个数直接当下标 | `int values[3] = {1, 2, 3};`<br>`return values[sizeof(values) / sizeof(values[0])];` | `int values[3] = {1, 2, 3};`<br>`return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `hard-3` | 尾指针 `end[0]` 越界 | `int values[3] = {1, 2, 3};`<br>`int *end = values + 3;`<br>`return end[0];` | `int values[3] = {1, 2, 3};`<br>`int *end = values + 3;`<br>`return end[-1];` |
| `hard-4` | 4 元素数组访问 `values[4]` | `int values[4] = {1, 2, 3, 4};`<br>`return values[4];` | `int values[4] = {1, 2, 3, 4};`<br>`return values[3];` |

### 字符串溢出（`string_overflow`）

| 写法 | 测试点 | 正例代码 | 对照代码 |
|---|---|---|---|
| `base-0` | `strcpy` 写入超过目标容量的字面量 | `char target[4] = {0};`<br>`strcpy(target, "ABCDE");`<br>`return target[0];` | `char target[4] = {0};`<br>`strcpy(target, "ABC");`<br>`return target[0];` |
| `base-1` | `sprintf` 写入超过目标容量的字符串 | `char target[4] = {0};`<br>`sprintf(target, "%s", "ABCDE");`<br>`return target[0];` | `char target[4] = {0};`<br>`sprintf(target, "%s", "ABC");`<br>`return target[0];` |
| `base-2` | `strcat` 拼接后超过目标容量 | `char target[8] = "A";`<br>`strcat(target, "BCDEFGH");`<br>`return target[0];` | `char target[8] = "A";`<br>`strcat(target, "BC");`<br>`return target[0];` |
| `base-3` | `memcpy` 拷贝长度超过目标容量 | `char target[4] = {0};`<br>`memcpy(target, "ABCDE", 6);`<br>`return target[0];` | `char target[4] = {0};`<br>`memcpy(target, "ABCDE", 3);`<br>`return target[0];` |
| `base-4` | 经辅助函数 `contract3_copy`（内部 `strcpy`）写入超长字符串（跨函数） | `char target[4] = {0};`<br>`contract3_copy(target, "ABCDE");`<br>`return target[0];` | `char target[4] = {0};`<br>`contract3_copy(target, "ABC");`<br>`return target[0];` |
| `hard-0` | 目标数组恰好少 1 字节，放不下结尾 `\0` | `char target[5] = {0};`<br>`strcpy(target, "ABCDE");`<br>`return target[0];` | `char target[6] = {0};`<br>`strcpy(target, "ABCDE");`<br>`return target[0];` |
| `hard-1` | `snprintf` 长度参数大于目标容量 | `char target[4] = {0};`<br>`snprintf(target, 16, "%s", "ABCDEFGHIJ");`<br>`return target[0];` | `char target[4] = {0};`<br>`snprintf(target, sizeof(target), "%s", "ABCDEFGHIJ");`<br>`return target[0];` |
| `hard-2` | `strncpy` 长度参数大于目标容量 | `char target[4];`<br>`strncpy(target, "ABCDEFGH", 8);`<br>`return target[0];` | `char target[4];`<br>`strncpy(target, "ABCDEFGH", sizeof(target) - 1);`<br>`target[sizeof(target) - 1] = '\0';`<br>`return target[0];` |
| `hard-3` | `strcat` 拼接后恰好多 1 字节 | `char target[8] = "ABC";`<br>`strcat(target, "DEFGH");`<br>`return target[0];` | `char target[8] = "ABC";`<br>`strcat(target, "DEFG");`<br>`return target[0];` |
| `hard-4` | `memcpy` 长度取源串 `sizeof`，大于目标容量 | `char target[4];`<br>`memcpy(target, "ABCDE", sizeof("ABCDE"));`<br>`return target[0];` | `char target[4];`<br>`memcpy(target, "ABCDE", sizeof(target));`<br>`return target[0];` |

### 空指针调用（`null_pointer`）

| 写法 | 测试点 | 正例代码 | 对照代码 |
|---|---|---|---|
| `base-0` | 指针初始化为 0 后直接解引用 | `int *pointer = 0;`<br>`return *pointer;` | `int value = 7;`<br>`return value;` |
| `base-1` | 指针赋值为 0 后解引用 | `int value = 7;`<br>`int *pointer = 0;`<br>`return *pointer;` | `int value = 7;`<br>`int *pointer = &value;`<br>`return *pointer;` |
| `base-2` | 把 0 作为实参传给会解引用的辅助函数 `contract3_read`（跨函数） | `int value = 7;`<br>`return contract3_read(0);` | `int value = 7;`<br>`return contract3_read(&value);` |
| `base-3` | 结构体成员指针为 0 后解引用 | `int value = 7;`<br>`struct contract3_box box = {0};`<br>`return *box.pointer;` | `int value = 7;`<br>`struct contract3_box box = {&value};`<br>`return *box.pointer;` |
| `base-4` | 分支条件不成立，指针保持 0 后解引用 | `int value = 7;`<br>`int *pointer = 0;`<br>`if (value < 0) pointer = &value;`<br>`return *pointer;` | `int value = 7;`<br>`int *pointer = 0;`<br>`if (value > 0) pointer = &value;`<br>`return *pointer;` |
| `hard-0` | 判空后未返回，继续解引用 | `int value = 7;`<br>`int *pointer = 0;`<br>`if (pointer == 0) value = -1;`<br>`return *pointer + value;` | `int value = 7;`<br>`int *pointer = 0;`<br>`if (pointer == 0) return -1;`<br>`return *pointer + value;` |
| `hard-1` | 条件表达式选出空指针后直接解引用 | `int value = 7;`<br>`int *pointer = value < 0 ? &value : 0;`<br>`return *pointer;` | `int value = 7;`<br>`int *pointer = value < 0 ? &value : 0;`<br>`return pointer ? *pointer : 0;` |
| `hard-2` | 指针先指向有效对象，又被重新赋值为 0 后解引用 | `int value = 7;`<br>`int *pointer = &value;`<br>`pointer = 0;`<br>`return *pointer;` | `int value = 7;`<br>`int *pointer = 0;`<br>`pointer = &value;`<br>`return *pointer;` |
| `hard-3` | 辅助函数 `contract3_select` 按参数返回空指针后解引用（跨函数） | `int value = 7;`<br>`int *pointer = contract3_select(&value, 0);`<br>`return *pointer;` | `int value = 7;`<br>`int *pointer = contract3_select(&value, 1);`<br>`return *pointer;` |
| `hard-4` | 判断条件写反，指针保持 0 后解引用 | `int value = 7;`<br>`struct contract3_box box = {0};`<br>`if (box.pointer != 0) box.pointer = &value;`<br>`return *box.pointer;` | `int value = 7;`<br>`struct contract3_box box = {0};`<br>`if (box.pointer == 0) box.pointer = &value;`<br>`return *box.pointer;` |

## 3. 工程与检测文件

检测文件 = `scan_roots` 下的 C/C++ 源码与头文件（不含注入的测试文件）。完整文件列表与逐文件行数见下方折叠项及 GT JSON 的 `detection_files`。

| 工程 | 版本 | 提交 | 扫描范围 | 检测文件数 | 检测文件合计行数 |
|---|---|---|---|---:|---:|
| FreeRTOS Kernel | V11.1.0 | `dbf70559b27d` | `.` | 255 | 186,067 |
| Zephyr RTOS | v3.7.0 | `36940db938a8` | `kernel` | 52 | 17,050 |
| RT-Thread | v5.2.1 | `97893c004c65` | `src` | 32 | 20,545 |

<details><summary>FreeRTOS Kernel 检测文件（255 个）</summary>

| 文件 | 行数 |
|---|---:|
| `croutine.c` | 405 |
| `event_groups.c` | 884 |
| `list.c` | 246 |
| `portable/ARMv8M/non_secure/port.c` | 2,144 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM23/mpu_wrappers_v2_asm.c` | 2,125 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM23/portasm.c` | 604 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM23_NTZ/mpu_wrappers_v2_asm.c` | 2,125 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM23_NTZ/portasm.c` | 516 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM33/mpu_wrappers_v2_asm.c` | 2,056 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM33/portasm.c` | 577 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM33_NTZ/mpu_wrappers_v2_asm.c` | 2,055 |
| `portable/ARMv8M/non_secure/portable/GCC/ARM_CM33_NTZ/portasm.c` | 471 |
| `portable/ARMv8M/secure/context/portable/GCC/ARM_CM23/secure_context_port.c` | 99 |
| `portable/ARMv8M/secure/context/portable/GCC/ARM_CM33/secure_context_port.c` | 97 |
| `portable/ARMv8M/secure/context/secure_context.c` | 354 |
| `portable/ARMv8M/secure/heap/secure_heap.c` | 476 |
| `portable/ARMv8M/secure/init/secure_init.c` | 106 |
| `portable/BCC/16BitDOS/Flsh186/port.c` | 244 |
| `portable/BCC/16BitDOS/PC/port.c` | 288 |
| `portable/BCC/16BitDOS/common/portcomn.c` | 120 |
| `portable/CCS/ARM_CM3/port.c` | 692 |
| `portable/CCS/ARM_CM4F/port.c` | 717 |
| `portable/CCS/ARM_Cortex-R4/port.c` | 312 |
| `portable/CCS/MSP430X/port.c` | 188 |
| `portable/CodeWarrior/ColdFire_V1/port.c` | 183 |
| `portable/CodeWarrior/ColdFire_V2/port.c` | 147 |
| `portable/CodeWarrior/HCS12/port.c` | 236 |
| `portable/Common/mpu_wrappers.c` | 2,551 |
| `portable/Common/mpu_wrappers_v2.c` | 5,267 |
| `portable/GCC/ARM7_AT91FR40008/port.c` | 240 |
| `portable/GCC/ARM7_AT91FR40008/portISR.c` | 233 |
| `portable/GCC/ARM7_AT91SAM7S/lib_AT91SAM7X256.c` | 50 |
| `portable/GCC/ARM7_AT91SAM7S/port.c` | 209 |
| `portable/GCC/ARM7_AT91SAM7S/portISR.c` | 227 |
| `portable/GCC/ARM7_LPC2000/port.c` | 222 |
| `portable/GCC/ARM7_LPC2000/portISR.c` | 216 |
| `portable/GCC/ARM7_LPC23xx/port.c` | 233 |
| `portable/GCC/ARM7_LPC23xx/portISR.c` | 219 |
| `portable/GCC/ARM_AARCH64/port.c` | 524 |
| `portable/GCC/ARM_AARCH64_SRE/port.c` | 465 |
| `portable/GCC/ARM_CA9/port.c` | 577 |
| `portable/GCC/ARM_CM0/mpu_wrappers_v2_asm.c` | 2,217 |
| `portable/GCC/ARM_CM0/port.c` | 1,665 |
| `portable/GCC/ARM_CM0/portasm.c` | 523 |
| `portable/GCC/ARM_CM23/non_secure/mpu_wrappers_v2_asm.c` | 2,125 |
| `portable/GCC/ARM_CM23/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM23/non_secure/portasm.c` | 604 |
| `portable/GCC/ARM_CM23/secure/secure_context.c` | 354 |
| `portable/GCC/ARM_CM23/secure/secure_context_port.c` | 99 |
| `portable/GCC/ARM_CM23/secure/secure_heap.c` | 476 |
| `portable/GCC/ARM_CM23/secure/secure_init.c` | 106 |
| `portable/GCC/ARM_CM23_NTZ/non_secure/mpu_wrappers_v2_asm.c` | 2,125 |
| `portable/GCC/ARM_CM23_NTZ/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM23_NTZ/non_secure/portasm.c` | 516 |
| `portable/GCC/ARM_CM3/port.c` | 824 |
| `portable/GCC/ARM_CM33/non_secure/mpu_wrappers_v2_asm.c` | 2,056 |
| `portable/GCC/ARM_CM33/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM33/non_secure/portasm.c` | 577 |
| `portable/GCC/ARM_CM33/secure/secure_context.c` | 354 |
| `portable/GCC/ARM_CM33/secure/secure_context_port.c` | 97 |
| `portable/GCC/ARM_CM33/secure/secure_heap.c` | 476 |
| `portable/GCC/ARM_CM33/secure/secure_init.c` | 106 |
| `portable/GCC/ARM_CM33_NTZ/non_secure/mpu_wrappers_v2_asm.c` | 2,055 |
| `portable/GCC/ARM_CM33_NTZ/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c` | 471 |
| `portable/GCC/ARM_CM35P/non_secure/mpu_wrappers_v2_asm.c` | 2,056 |
| `portable/GCC/ARM_CM35P/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM35P/non_secure/portasm.c` | 577 |
| `portable/GCC/ARM_CM35P/secure/secure_context.c` | 354 |
| `portable/GCC/ARM_CM35P/secure/secure_context_port.c` | 97 |
| `portable/GCC/ARM_CM35P/secure/secure_heap.c` | 476 |
| `portable/GCC/ARM_CM35P/secure/secure_init.c` | 106 |
| `portable/GCC/ARM_CM35P_NTZ/non_secure/mpu_wrappers_v2_asm.c` | 2,055 |
| `portable/GCC/ARM_CM35P_NTZ/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM35P_NTZ/non_secure/portasm.c` | 471 |
| `portable/GCC/ARM_CM3_MPU/mpu_wrappers_v2_asm.c` | 2,067 |
| `portable/GCC/ARM_CM3_MPU/port.c` | 1,581 |
| `portable/GCC/ARM_CM4F/port.c` | 909 |
| `portable/GCC/ARM_CM4_MPU/mpu_wrappers_v2_asm.c` | 2,067 |
| `portable/GCC/ARM_CM4_MPU/port.c` | 1,724 |
| `portable/GCC/ARM_CM55/non_secure/mpu_wrappers_v2_asm.c` | 2,056 |
| `portable/GCC/ARM_CM55/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM55/non_secure/portasm.c` | 577 |
| `portable/GCC/ARM_CM55/secure/secure_context.c` | 354 |
| `portable/GCC/ARM_CM55/secure/secure_context_port.c` | 97 |
| `portable/GCC/ARM_CM55/secure/secure_heap.c` | 476 |
| `portable/GCC/ARM_CM55/secure/secure_init.c` | 106 |
| `portable/GCC/ARM_CM55_NTZ/non_secure/mpu_wrappers_v2_asm.c` | 2,055 |
| `portable/GCC/ARM_CM55_NTZ/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM55_NTZ/non_secure/portasm.c` | 471 |
| `portable/GCC/ARM_CM7/r0p1/port.c` | 899 |
| `portable/GCC/ARM_CM85/non_secure/mpu_wrappers_v2_asm.c` | 2,056 |
| `portable/GCC/ARM_CM85/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM85/non_secure/portasm.c` | 577 |
| `portable/GCC/ARM_CM85/secure/secure_context.c` | 354 |
| `portable/GCC/ARM_CM85/secure/secure_context_port.c` | 97 |
| `portable/GCC/ARM_CM85/secure/secure_heap.c` | 476 |
| `portable/GCC/ARM_CM85/secure/secure_init.c` | 106 |
| `portable/GCC/ARM_CM85_NTZ/non_secure/mpu_wrappers_v2_asm.c` | 2,055 |
| `portable/GCC/ARM_CM85_NTZ/non_secure/port.c` | 2,144 |
| `portable/GCC/ARM_CM85_NTZ/non_secure/portasm.c` | 471 |
| `portable/GCC/ARM_CR5/port.c` | 691 |
| `portable/GCC/ARM_CRx_MPU/port.c` | 845 |
| `portable/GCC/ARM_CRx_No_GIC/port.c` | 324 |
| `portable/GCC/ATMega323/port.c` | 428 |
| `portable/GCC/AVR32_UC3/port.c` | 473 |
| `portable/GCC/CORTUS_APS3/port.c` | 148 |
| `portable/GCC/ColdFire_V2/port.c` | 133 |
| `portable/GCC/H8S2329/port.c` | 304 |
| `portable/GCC/HCS12/port.c` | 239 |
| `portable/GCC/IA32_flat/port.c` | 686 |
| `portable/GCC/MSP430F449/port.c` | 335 |
| `portable/GCC/MicroBlaze/port.c` | 334 |
| `portable/GCC/MicroBlazeV8/port.c` | 457 |
| `portable/GCC/MicroBlazeV8/port_exceptions.c` | 278 |
| `portable/GCC/MicroBlazeV9/port.c` | 545 |
| `portable/GCC/MicroBlazeV9/port_exceptions.c` | 277 |
| `portable/GCC/NiosII/port.c` | 216 |
| `portable/GCC/PPC405_Xilinx/port.c` | 267 |
| `portable/GCC/PPC440_Xilinx/port.c` | 267 |
| `portable/GCC/RISC-V/port.c` | 208 |
| `portable/GCC/RL78/port.c` | 213 |
| `portable/GCC/RX100/port.c` | 714 |
| `portable/GCC/RX200/port.c` | 445 |
| `portable/GCC/RX600/port.c` | 399 |
| `portable/GCC/RX600v2/port.c` | 442 |
| `portable/GCC/RX700v3_DPFPU/port.c` | 619 |
| `portable/GCC/STR75x/port.c` | 193 |
| `portable/GCC/STR75x/portISR.c` | 178 |
| `portable/GCC/TriCore_1782/port.c` | 547 |
| `portable/GCC/TriCore_1782/porttrap.c` | 282 |
| `portable/IAR/78K0R/port.c` | 227 |
| `portable/IAR/ARM_CA5_No_GIC/port.c` | 304 |
| `portable/IAR/ARM_CA9/port.c` | 446 |
| `portable/IAR/ARM_CM0/port.c` | 543 |
| `portable/IAR/ARM_CM23/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM23/secure/secure_context.c` | 354 |
| `portable/IAR/ARM_CM23/secure/secure_heap.c` | 476 |
| `portable/IAR/ARM_CM23/secure/secure_init.c` | 106 |
| `portable/IAR/ARM_CM23_NTZ/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM3/port.c` | 732 |
| `portable/IAR/ARM_CM33/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM33/secure/secure_context.c` | 354 |
| `portable/IAR/ARM_CM33/secure/secure_heap.c` | 476 |
| `portable/IAR/ARM_CM33/secure/secure_init.c` | 106 |
| `portable/IAR/ARM_CM33_NTZ/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM35P/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM35P/secure/secure_context.c` | 354 |
| `portable/IAR/ARM_CM35P/secure/secure_heap.c` | 476 |
| `portable/IAR/ARM_CM35P/secure/secure_init.c` | 106 |
| `portable/IAR/ARM_CM35P_NTZ/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM4F/port.c` | 776 |
| `portable/IAR/ARM_CM4F_MPU/port.c` | 1,455 |
| `portable/IAR/ARM_CM55/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM55/secure/secure_context.c` | 354 |
| `portable/IAR/ARM_CM55/secure/secure_heap.c` | 476 |
| `portable/IAR/ARM_CM55/secure/secure_init.c` | 106 |
| `portable/IAR/ARM_CM55_NTZ/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM7/r0p1/port.c` | 764 |
| `portable/IAR/ARM_CM85/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CM85/secure/secure_context.c` | 354 |
| `portable/IAR/ARM_CM85/secure/secure_heap.c` | 476 |
| `portable/IAR/ARM_CM85/secure/secure_init.c` | 106 |
| `portable/IAR/ARM_CM85_NTZ/non_secure/port.c` | 2,144 |
| `portable/IAR/ARM_CRx_No_GIC/port.c` | 321 |
| `portable/IAR/ATMega323/port.c` | 340 |
| `portable/IAR/AVR32_UC3/port.c` | 446 |
| `portable/IAR/AVR32_UC3/read.c` | 129 |
| `portable/IAR/AVR32_UC3/write.c` | 136 |
| `portable/IAR/AVR_AVRDx/port.c` | 301 |
| `portable/IAR/AVR_Mega0/port.c` | 299 |
| `portable/IAR/AtmelSAM7S64/port.c` | 252 |
| `portable/IAR/AtmelSAM9XE/port.c` | 253 |
| `portable/IAR/LPC2000/port.c` | 314 |
| `portable/IAR/MSP430/port.c` | 173 |
| `portable/IAR/MSP430X/port.c` | 184 |
| `portable/IAR/RISC-V/port.c` | 241 |
| `portable/IAR/RL78/port.c` | 215 |
| `portable/IAR/RX100/port.c` | 523 |
| `portable/IAR/RX600/port.c` | 193 |
| `portable/IAR/RX700v3_DPFPU/port.c` | 565 |
| `portable/IAR/RXv2/port.c` | 201 |
| `portable/IAR/STR71x/port.c` | 255 |
| `portable/IAR/STR75x/port.c` | 234 |
| `portable/IAR/STR91x/port.c` | 435 |
| `portable/IAR/V850ES/port.c` | 184 |
| `portable/MPLAB/PIC18F/port.c` | 615 |
| `portable/MPLAB/PIC24_dsPIC/port.c` | 334 |
| `portable/MPLAB/PIC32MEC14xx/port.c` | 341 |
| `portable/MPLAB/PIC32MX/port.c` | 330 |
| `portable/MPLAB/PIC32MZ/port.c` | 369 |
| `portable/MSVC-MingW/port.c` | 728 |
| `portable/MemMang/heap_1.c` | 165 |
| `portable/MemMang/heap_2.c` | 401 |
| `portable/MemMang/heap_3.c` | 106 |
| `portable/MemMang/heap_4.c` | 626 |
| `portable/MemMang/heap_5.c` | 730 |
| `portable/MikroC/ARM_CM4F/port.c` | 915 |
| `portable/Paradigm/Tern_EE/large_untested/port.c` | 233 |
| `portable/Paradigm/Tern_EE/small/port.c` | 213 |
| `portable/RVDS/ARM7_LPC21xx/port.c` | 292 |
| `portable/RVDS/ARM_CA9/port.c` | 484 |
| `portable/RVDS/ARM_CM0/port.c` | 622 |
| `portable/RVDS/ARM_CM3/port.c` | 793 |
| `portable/RVDS/ARM_CM4F/port.c` | 889 |
| `portable/RVDS/ARM_CM4_MPU/mpu_wrappers_v2_asm.c` | 1,705 |
| `portable/RVDS/ARM_CM4_MPU/port.c` | 1,721 |
| `portable/RVDS/ARM_CM7/r0p1/port.c` | 875 |
| `portable/Renesas/RX100/port.c` | 646 |
| `portable/Renesas/RX200/port.c` | 322 |
| `portable/Renesas/RX600/port.c` | 327 |
| `portable/Renesas/RX600v2/port.c` | 362 |
| `portable/Renesas/RX700v3_DPFPU/port.c` | 591 |
| `portable/Renesas/SH2A_FPU/port.c` | 272 |
| `portable/Rowley/MSP430F449/port.c` | 173 |
| `portable/SDCC/Cygnal/port.c` | 421 |
| `portable/Softune/MB91460/__STD_LIB_sbrk.c` | 55 |
| `portable/Softune/MB91460/port.c` | 321 |
| `portable/Softune/MB96340/__STD_LIB_sbrk.c` | 55 |
| `portable/Softune/MB96340/port.c` | 509 |
| `portable/Tasking/ARM_CM4F/port.c` | 275 |
| `portable/ThirdParty/CDK/T-HEAD_CK802/port.c` | 168 |
| `portable/ThirdParty/GCC/ARC_EM_HS/arc_freertos_exceptions.c` | 52 |
| `portable/ThirdParty/GCC/ARC_EM_HS/freertos_tls.c` | 240 |
| `portable/ThirdParty/GCC/ARC_EM_HS/port.c` | 293 |
| `portable/ThirdParty/GCC/ARC_v1/arc_freertos_exceptions.c` | 52 |
| `portable/ThirdParty/GCC/ARC_v1/port.c` | 288 |
| `portable/ThirdParty/GCC/ARM_TFM/os_wrapper_freertos.c` | 112 |
| `portable/ThirdParty/GCC/ATmega/port.c` | 779 |
| `portable/ThirdParty/GCC/Posix/port.c` | 618 |
| `portable/ThirdParty/GCC/Posix/utils/wait_for_event.c` | 104 |
| `portable/ThirdParty/GCC/RP2040/port.c` | 1,184 |
| `portable/ThirdParty/GCC/Xtensa_ESP32/FreeRTOS-openocd.c` | 27 |
| `portable/ThirdParty/GCC/Xtensa_ESP32/port.c` | 572 |
| `portable/ThirdParty/GCC/Xtensa_ESP32/port_common.c` | 167 |
| `portable/ThirdParty/GCC/Xtensa_ESP32/port_systick.c` | 197 |
| `portable/ThirdParty/GCC/Xtensa_ESP32/xtensa_init.c` | 76 |
| `portable/ThirdParty/GCC/Xtensa_ESP32/xtensa_overlay_os_hook.c` | 76 |
| `portable/ThirdParty/XCC/Xtensa/port.c` | 209 |
| `portable/ThirdParty/XCC/Xtensa/portclib.c` | 230 |
| `portable/ThirdParty/XCC/Xtensa/xtensa_init.c` | 70 |
| `portable/ThirdParty/XCC/Xtensa/xtensa_intr.c` | 137 |
| `portable/ThirdParty/XCC/Xtensa/xtensa_overlay_os_hook.c` | 76 |
| `portable/ThirdParty/xClang/XCOREAI/port.c` | 262 |
| `portable/WizC/PIC18/Drivers/Tick/Tick.c` | 139 |
| `portable/WizC/PIC18/Drivers/Tick/isrTick.c` | 79 |
| `portable/WizC/PIC18/port.c` | 318 |
| `portable/oWatcom/16BitDOS/Flsh186/port.c` | 246 |
| `portable/oWatcom/16BitDOS/PC/port.c` | 302 |
| `portable/oWatcom/16BitDOS/common/portcomn.c` | 141 |
| `portable/template/port.c` | 82 |
| `queue.c` | 3,364 |
| `stream_buffer.c` | 1,718 |
| `tasks.c` | 8,696 |
| `timers.c` | 1,340 |

</details>

<details><summary>Zephyr RTOS 检测文件（52 个）</summary>

| 文件 | 行数 |
|---|---:|
| `kernel/atomic_c.c` | 414 |
| `kernel/banner.c` | 48 |
| `kernel/busy_wait.c` | 56 |
| `kernel/compiler_stack_protect.c` | 64 |
| `kernel/condvar.c` | 158 |
| `kernel/cpu_mask.c` | 76 |
| `kernel/device.c` | 196 |
| `kernel/dynamic.c` | 181 |
| `kernel/dynamic_disabled.c` | 25 |
| `kernel/errno.c` | 57 |
| `kernel/events.c` | 367 |
| `kernel/fatal.c` | 178 |
| `kernel/float.c` | 47 |
| `kernel/futex.c` | 101 |
| `kernel/idle.c` | 102 |
| `kernel/init.c` | 755 |
| `kernel/init_static.c` | 87 |
| `kernel/ipi.c` | 108 |
| `kernel/irq_offload.c` | 23 |
| `kernel/kheap.c` | 161 |
| `kernel/mailbox.c` | 461 |
| `kernel/main_weak.c` | 30 |
| `kernel/mem_domain.c` | 368 |
| `kernel/mem_slab.c` | 334 |
| `kernel/mempool.c` | 178 |
| `kernel/mmu.c` | 1,680 |
| `kernel/msg_q.c` | 442 |
| `kernel/mutex.c` | 314 |
| `kernel/nothread.c` | 52 |
| `kernel/obj_core.c` | 327 |
| `kernel/paging/statistics.c` | 251 |
| `kernel/pipes.c` | 832 |
| `kernel/poll.c` | 815 |
| `kernel/priority_queues.c` | 30 |
| `kernel/queue.c` | 480 |
| `kernel/sched.c` | 1,602 |
| `kernel/sem.c` | 231 |
| `kernel/smp.c` | 250 |
| `kernel/spinlock_validate.c` | 47 |
| `kernel/stack.c` | 222 |
| `kernel/system_work_q.c` | 37 |
| `kernel/thread.c` | 1,109 |
| `kernel/thread_monitor.c` | 85 |
| `kernel/timeout.c` | 340 |
| `kernel/timer.c` | 365 |
| `kernel/timeslicing.c` | 131 |
| `kernel/usage.c` | 485 |
| `kernel/userspace.c` | 1,017 |
| `kernel/userspace_handler.c` | 89 |
| `kernel/version.c` | 21 |
| `kernel/work.c` | 1,146 |
| `kernel/xip.c` | 75 |

</details>

<details><summary>RT-Thread 检测文件（32 个）</summary>

| 文件 | 行数 |
|---|---:|
| `src/clock.c` | 261 |
| `src/components.c` | 286 |
| `src/cpu_mp.c` | 236 |
| `src/cpu_up.c` | 108 |
| `src/defunct.c` | 178 |
| `src/idle.c` | 219 |
| `src/ipc.c` | 4,033 |
| `src/irq.c` | 157 |
| `src/klibc/kerrno.c` | 161 |
| `src/klibc/kstdio.c` | 116 |
| `src/klibc/kstring.c` | 559 |
| `src/klibc/rt_vsnprintf_std.c` | 1,351 |
| `src/klibc/rt_vsnprintf_tiny.c` | 611 |
| `src/klibc/rt_vsscanf.c` | 700 |
| `src/klibc/utest/TC_rt_memcmp.c` | 161 |
| `src/klibc/utest/TC_rt_memcpy.c` | 108 |
| `src/klibc/utest/TC_rt_memmove.c` | 108 |
| `src/klibc/utest/TC_rt_memset.c` | 102 |
| `src/klibc/utest/TC_rt_sprintf.c` | 1,065 |
| `src/klibc/utest/TC_rt_sscanf.c` | 250 |
| `src/kservice.c` | 1,162 |
| `src/mem.c` | 667 |
| `src/memheap.c` | 998 |
| `src/mempool.c` | 411 |
| `src/object.c` | 774 |
| `src/scheduler_comm.c` | 309 |
| `src/scheduler_mp.c` | 1,332 |
| `src/scheduler_up.c` | 571 |
| `src/signal.c` | 679 |
| `src/slab.c` | 856 |
| `src/thread.c` | 1,145 |
| `src/timer.c` | 871 |

</details>

## 4. 逐例 Ground Truth

测试文件放在各工程根目录。“缺陷行”是正例中触发缺陷的语句；对照列出的是同一位置的安全语句。

### 平衡验收集（`balanced`）

<details><summary>FreeRTOS Kernel · 数组越界 · <code>contract3_balanced_web_array_oob.c</code>（288 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_01` | 11–15 | 14 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_02` | 17–22 | 21 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 24–30 | 29 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_04` | 32–37 | 36 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_05` | 39–43 | 42 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_06` | 45–50 | 49 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_07` | 52–58 | 57 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_08` | 60–65 | 64 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_09` | 67–71 | 70 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_10` | 73–78 | 77 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 80–86 | 85 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 88–93 | 92 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_13` | 95–99 | 98 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 101–106 | 105 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_15` | 108–114 | 113 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_16` | 116–121 | 120 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_17` | 123–127 | 126 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_18` | 129–134 | 133 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_19` | 136–141 | 140 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_20` | 143–148 | 147 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_21` | 150–154 | 153 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 156–161 | 160 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_23` | 163–169 | 168 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 171–176 | 175 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_25` | 178–182 | 181 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_26` | 184–189 | 188 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_27` | 191–197 | 196 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_28` | 199–204 | 203 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_29` | 206–210 | 209 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_30` | 212–217 | 216 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_31` | 219–225 | 224 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_32` | 227–232 | 231 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_33` | 234–238 | 237 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_34` | 240–245 | 244 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_35` | 247–253 | 252 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_36` | 255–260 | 259 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_37` | 262–266 | 265 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_38` | 268–273 | 272 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_39` | 275–280 | 279 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_40` | 282–287 | 286 | `base-4` | `return *cursor;` |

</details>

<details><summary>FreeRTOS Kernel · 字符串溢出 · <code>contract3_balanced_web_string_overflow.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_01` | 11–16 | 15 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_02` | 18–23 | 22 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_03` | 25–30 | 29 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_04` | 32–37 | 36 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_05` | 39–44 | 43 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_06` | 46–51 | 50 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_07` | 53–58 | 57 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_08` | 60–65 | 64 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_09` | 67–72 | 71 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_10` | 74–79 | 78 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_11` | 81–86 | 85 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_12` | 88–93 | 92 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 95–100 | 99 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_14` | 102–107 | 106 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_15` | 109–114 | 113 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_16` | 116–121 | 120 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_17` | 123–128 | 127 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_18` | 130–135 | 134 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_19` | 137–142 | 141 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_20` | 144–149 | 148 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_21` | 151–156 | 155 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_22` | 158–163 | 162 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_23` | 165–170 | 169 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_24` | 172–177 | 176 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_25` | 179–184 | 183 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_26` | 186–191 | 190 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_27` | 193–198 | 197 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_28` | 200–205 | 204 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 207–212 | 211 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_30` | 214–219 | 218 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_31` | 221–226 | 225 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_32` | 228–233 | 232 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_33` | 235–240 | 239 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_34` | 242–247 | 246 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_35` | 249–254 | 253 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_36` | 256–261 | 260 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_37` | 263–268 | 267 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_38` | 270–275 | 274 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_39` | 277–282 | 281 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_40` | 284–289 | 288 | `base-4` | `return target[0];` |

</details>

<details><summary>FreeRTOS Kernel · 空指针调用 · <code>contract3_balanced_web_null_pointer.c</code>（288 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 11–15 | 14 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 17–22 | 21 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 24–30 | 29 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_04` | 32–36 | 35 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_05` | 38–43 | 42 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 45–51 | 50 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 53–57 | 56 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_08` | 59–64 | 63 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_09` | 66–72 | 71 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_10` | 74–78 | 77 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_11` | 80–85 | 84 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 87–93 | 92 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_13` | 95–99 | 98 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 101–106 | 105 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_15` | 108–114 | 113 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_16` | 116–120 | 119 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_17` | 122–127 | 126 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_18` | 129–135 | 134 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_19` | 137–141 | 140 | `base-2` | `return contract3_read(0);` |
| `freertos_kernel-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_20` | 143–148 | 147 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_21` | 150–154 | 153 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_22` | 156–161 | 160 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_23` | 163–169 | 168 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_24` | 171–175 | 174 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_25` | 177–182 | 181 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 184–190 | 189 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 192–196 | 195 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_28` | 198–203 | 202 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 205–211 | 210 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_30` | 213–217 | 216 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_31` | 219–224 | 223 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_32` | 226–232 | 231 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_33` | 234–238 | 237 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_34` | 240–245 | 244 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_35` | 247–253 | 252 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_36` | 255–259 | 258 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_37` | 261–266 | 265 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_38` | 268–274 | 273 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_39` | 276–280 | 279 | `base-2` | `return contract3_read(&value);` |
| `freertos_kernel-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 282–287 | 286 | `base-3` | `return *box.pointer;` |

</details>

<details><summary>Zephyr RTOS · 数组越界 · <code>contract3_balanced_web_array_oob.c</code>（288 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_01` | 11–15 | 14 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_02` | 17–22 | 21 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 24–30 | 29 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_04` | 32–37 | 36 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_05` | 39–43 | 42 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_06` | 45–50 | 49 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_07` | 52–58 | 57 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_08` | 60–65 | 64 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_09` | 67–71 | 70 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_10` | 73–78 | 77 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 80–86 | 85 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 88–93 | 92 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_13` | 95–99 | 98 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 101–106 | 105 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_15` | 108–114 | 113 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_16` | 116–121 | 120 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_17` | 123–127 | 126 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_18` | 129–134 | 133 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_19` | 136–141 | 140 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_20` | 143–148 | 147 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_21` | 150–154 | 153 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 156–161 | 160 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_23` | 163–169 | 168 | `base-2` | `return total;` |
| `zephyr-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 171–176 | 175 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_25` | 178–182 | 181 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_26` | 184–189 | 188 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_27` | 191–197 | 196 | `base-2` | `return total;` |
| `zephyr-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_28` | 199–204 | 203 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_29` | 206–210 | 209 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_30` | 212–217 | 216 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_31` | 219–225 | 224 | `base-2` | `return total;` |
| `zephyr-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_32` | 227–232 | 231 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_33` | 234–238 | 237 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_34` | 240–245 | 244 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_35` | 247–253 | 252 | `base-2` | `return total;` |
| `zephyr-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_36` | 255–260 | 259 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_37` | 262–266 | 265 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_38` | 268–273 | 272 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_39` | 275–280 | 279 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_40` | 282–287 | 286 | `base-4` | `return *cursor;` |

</details>

<details><summary>Zephyr RTOS · 字符串溢出 · <code>contract3_balanced_web_string_overflow.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_01` | 11–16 | 15 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_02` | 18–23 | 22 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_03` | 25–30 | 29 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_04` | 32–37 | 36 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_05` | 39–44 | 43 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_06` | 46–51 | 50 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_07` | 53–58 | 57 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_08` | 60–65 | 64 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_09` | 67–72 | 71 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_10` | 74–79 | 78 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_11` | 81–86 | 85 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_12` | 88–93 | 92 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 95–100 | 99 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_14` | 102–107 | 106 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_15` | 109–114 | 113 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_16` | 116–121 | 120 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_17` | 123–128 | 127 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_18` | 130–135 | 134 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_19` | 137–142 | 141 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_20` | 144–149 | 148 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_21` | 151–156 | 155 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_22` | 158–163 | 162 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_23` | 165–170 | 169 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_24` | 172–177 | 176 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_25` | 179–184 | 183 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_26` | 186–191 | 190 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_27` | 193–198 | 197 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_28` | 200–205 | 204 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 207–212 | 211 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_30` | 214–219 | 218 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_31` | 221–226 | 225 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_32` | 228–233 | 232 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_33` | 235–240 | 239 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_34` | 242–247 | 246 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_35` | 249–254 | 253 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_36` | 256–261 | 260 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_37` | 263–268 | 267 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_38` | 270–275 | 274 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_39` | 277–282 | 281 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_40` | 284–289 | 288 | `base-4` | `return target[0];` |

</details>

<details><summary>Zephyr RTOS · 空指针调用 · <code>contract3_balanced_web_null_pointer.c</code>（288 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 11–15 | 14 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 17–22 | 21 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 24–30 | 29 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_04` | 32–36 | 35 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_05` | 38–43 | 42 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 45–51 | 50 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 53–57 | 56 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_08` | 59–64 | 63 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_09` | 66–72 | 71 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_10` | 74–78 | 77 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_11` | 80–85 | 84 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 87–93 | 92 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_13` | 95–99 | 98 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 101–106 | 105 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_15` | 108–114 | 113 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_16` | 116–120 | 119 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_17` | 122–127 | 126 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_18` | 129–135 | 134 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_19` | 137–141 | 140 | `base-2` | `return contract3_read(0);` |
| `zephyr-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_20` | 143–148 | 147 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_21` | 150–154 | 153 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_22` | 156–161 | 160 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_23` | 163–169 | 168 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_24` | 171–175 | 174 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_25` | 177–182 | 181 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 184–190 | 189 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 192–196 | 195 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_28` | 198–203 | 202 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 205–211 | 210 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_30` | 213–217 | 216 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_31` | 219–224 | 223 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_32` | 226–232 | 231 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_33` | 234–238 | 237 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_34` | 240–245 | 244 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_35` | 247–253 | 252 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_36` | 255–259 | 258 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_37` | 261–266 | 265 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_38` | 268–274 | 273 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_39` | 276–280 | 279 | `base-2` | `return contract3_read(&value);` |
| `zephyr-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 282–287 | 286 | `base-3` | `return *box.pointer;` |

</details>

<details><summary>RT-Thread · 数组越界 · <code>contract3_balanced_web_array_oob.c</code>（288 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_01` | 11–15 | 14 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_02` | 17–22 | 21 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 24–30 | 29 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_04` | 32–37 | 36 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_05` | 39–43 | 42 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_06` | 45–50 | 49 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_07` | 52–58 | 57 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_08` | 60–65 | 64 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_09` | 67–71 | 70 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_10` | 73–78 | 77 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 80–86 | 85 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 88–93 | 92 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_13` | 95–99 | 98 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 101–106 | 105 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_15` | 108–114 | 113 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_16` | 116–121 | 120 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_17` | 123–127 | 126 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_18` | 129–134 | 133 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_19` | 136–141 | 140 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_20` | 143–148 | 147 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_21` | 150–154 | 153 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 156–161 | 160 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_23` | 163–169 | 168 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 171–176 | 175 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_25` | 178–182 | 181 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_26` | 184–189 | 188 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_27` | 191–197 | 196 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_28` | 199–204 | 203 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_29` | 206–210 | 209 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_30` | 212–217 | 216 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_31` | 219–225 | 224 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_32` | 227–232 | 231 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_33` | 234–238 | 237 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_34` | 240–245 | 244 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_35` | 247–253 | 252 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_36` | 255–260 | 259 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_37` | 262–266 | 265 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_38` | 268–273 | 272 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_39` | 275–280 | 279 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_40` | 282–287 | 286 | `base-4` | `return *cursor;` |

</details>

<details><summary>RT-Thread · 字符串溢出 · <code>contract3_balanced_web_string_overflow.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_01` | 11–16 | 15 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_02` | 18–23 | 22 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_03` | 25–30 | 29 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_04` | 32–37 | 36 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_05` | 39–44 | 43 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_06` | 46–51 | 50 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_07` | 53–58 | 57 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_08` | 60–65 | 64 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_09` | 67–72 | 71 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_10` | 74–79 | 78 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_11` | 81–86 | 85 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_12` | 88–93 | 92 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 95–100 | 99 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_14` | 102–107 | 106 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_15` | 109–114 | 113 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_16` | 116–121 | 120 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_17` | 123–128 | 127 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_18` | 130–135 | 134 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_19` | 137–142 | 141 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_20` | 144–149 | 148 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_21` | 151–156 | 155 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_22` | 158–163 | 162 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_23` | 165–170 | 169 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_24` | 172–177 | 176 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_25` | 179–184 | 183 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_26` | 186–191 | 190 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_27` | 193–198 | 197 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_28` | 200–205 | 204 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 207–212 | 211 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_30` | 214–219 | 218 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_31` | 221–226 | 225 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_32` | 228–233 | 232 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_33` | 235–240 | 239 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_34` | 242–247 | 246 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_35` | 249–254 | 253 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_36` | 256–261 | 260 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_37` | 263–268 | 267 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_38` | 270–275 | 274 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_39` | 277–282 | 281 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_40` | 284–289 | 288 | `base-4` | `return target[0];` |

</details>

<details><summary>RT-Thread · 空指针调用 · <code>contract3_balanced_web_null_pointer.c</code>（288 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 11–15 | 14 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 17–22 | 21 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 24–30 | 29 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_04` | 32–36 | 35 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_05` | 38–43 | 42 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 45–51 | 50 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 53–57 | 56 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_08` | 59–64 | 63 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_09` | 66–72 | 71 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_10` | 74–78 | 77 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_11` | 80–85 | 84 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 87–93 | 92 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_13` | 95–99 | 98 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 101–106 | 105 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_15` | 108–114 | 113 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_16` | 116–120 | 119 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_17` | 122–127 | 126 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_18` | 129–135 | 134 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_19` | 137–141 | 140 | `base-2` | `return contract3_read(0);` |
| `rt_thread-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_20` | 143–148 | 147 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_21` | 150–154 | 153 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_22` | 156–161 | 160 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_23` | 163–169 | 168 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_24` | 171–175 | 174 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_25` | 177–182 | 181 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 184–190 | 189 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 192–196 | 195 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_28` | 198–203 | 202 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 205–211 | 210 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_30` | 213–217 | 216 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_31` | 219–224 | 223 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_32` | 226–232 | 231 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_33` | 234–238 | 237 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_34` | 240–245 | 244 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_35` | 247–253 | 252 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_36` | 255–259 | 258 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_37` | 261–266 | 265 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_38` | 268–274 | 273 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_39` | 276–280 | 279 | `base-2` | `return contract3_read(&value);` |
| `rt_thread-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 282–287 | 286 | `base-3` | `return *box.pointer;` |

</details>

### 独立压力集（`independent`）

<details><summary>FreeRTOS Kernel · 数组越界 · <code>contract3_independent_web_array_oob.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_01` | 11–15 | 14 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_02` | 17–22 | 21 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 24–30 | 29 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_04` | 32–37 | 36 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_05` | 39–44 | 43 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_06` | 46–50 | 49 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_07` | 52–57 | 56 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_08` | 59–65 | 64 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_09` | 67–72 | 71 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_10` | 74–79 | 78 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 81–85 | 84 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 87–92 | 91 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_13` | 94–100 | 99 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 102–107 | 106 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_15` | 109–114 | 113 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_16` | 116–120 | 119 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_17` | 122–127 | 126 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_18` | 129–135 | 134 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_19` | 137–142 | 141 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_20` | 144–149 | 148 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_21` | 151–155 | 154 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 157–162 | 161 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_23` | 164–170 | 169 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 172–177 | 176 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_25` | 179–184 | 183 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_26` | 186–190 | 189 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_27` | 192–197 | 196 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_28` | 199–205 | 204 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_29` | 207–212 | 211 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_30` | 214–219 | 218 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_31` | 221–225 | 224 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_32` | 227–232 | 231 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_33` | 234–240 | 239 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_34` | 242–247 | 246 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_35` | 249–254 | 253 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_36` | 256–260 | 259 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_37` | 262–267 | 266 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_38` | 269–275 | 274 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_39` | 277–282 | 281 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_40` | 284–289 | 288 | `base-4` | `return *cursor;` |

</details>

<details><summary>FreeRTOS Kernel · 字符串溢出 · <code>contract3_independent_web_string_overflow.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_01` | 11–16 | 15 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_02` | 18–23 | 22 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_03` | 25–30 | 29 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_04` | 32–37 | 36 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_05` | 39–44 | 43 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_06` | 46–51 | 50 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_07` | 53–58 | 57 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_08` | 60–65 | 64 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_09` | 67–72 | 71 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_10` | 74–79 | 78 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_11` | 81–86 | 85 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_12` | 88–93 | 92 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 95–100 | 99 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_14` | 102–107 | 106 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_15` | 109–114 | 113 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_16` | 116–121 | 120 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_17` | 123–128 | 127 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_18` | 130–135 | 134 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_19` | 137–142 | 141 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_20` | 144–149 | 148 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_21` | 151–156 | 155 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_22` | 158–163 | 162 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_23` | 165–170 | 169 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_24` | 172–177 | 176 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_25` | 179–184 | 183 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_26` | 186–191 | 190 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_27` | 193–198 | 197 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_28` | 200–205 | 204 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 207–212 | 211 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_30` | 214–219 | 218 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_31` | 221–226 | 225 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_32` | 228–233 | 232 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_33` | 235–240 | 239 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_34` | 242–247 | 246 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_35` | 249–254 | 253 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_36` | 256–261 | 260 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_37` | 263–268 | 267 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_38` | 270–275 | 274 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_39` | 277–282 | 281 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_40` | 284–289 | 288 | `base-4` | `return target[0];` |

</details>

<details><summary>FreeRTOS Kernel · 空指针调用 · <code>contract3_independent_web_null_pointer.c</code>（282 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 11–15 | 14 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 17–22 | 21 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 24–28 | 27 | `base-2` | `return contract3_read(0);` |
| `freertos_kernel-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_04` | 30–35 | 34 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_05` | 37–43 | 42 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 45–49 | 48 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 51–56 | 55 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_08` | 58–62 | 61 | `base-2` | `return contract3_read(0);` |
| `freertos_kernel-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_09` | 64–69 | 68 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_10` | 71–77 | 76 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_11` | 79–83 | 82 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 85–90 | 89 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_13` | 92–96 | 95 | `base-2` | `return contract3_read(0);` |
| `freertos_kernel-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 98–103 | 102 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_15` | 105–111 | 110 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_16` | 113–117 | 116 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_17` | 119–124 | 123 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_18` | 126–130 | 129 | `base-2` | `return contract3_read(0);` |
| `freertos_kernel-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_19` | 132–137 | 136 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_20` | 139–145 | 144 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_21` | 147–151 | 150 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_22` | 153–158 | 157 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_23` | 160–164 | 163 | `base-2` | `return contract3_read(&value);` |
| `freertos_kernel-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_24` | 166–171 | 170 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_25` | 173–179 | 178 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 181–185 | 184 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 187–192 | 191 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_28` | 194–198 | 197 | `base-2` | `return contract3_read(&value);` |
| `freertos_kernel-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 200–205 | 204 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_30` | 207–213 | 212 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_31` | 215–219 | 218 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_32` | 221–226 | 225 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_33` | 228–232 | 231 | `base-2` | `return contract3_read(&value);` |
| `freertos_kernel-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_34` | 234–239 | 238 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_35` | 241–247 | 246 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_36` | 249–253 | 252 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_37` | 255–260 | 259 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_38` | 262–266 | 265 | `base-2` | `return contract3_read(&value);` |
| `freertos_kernel-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_39` | 268–273 | 272 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 275–281 | 280 | `base-4` | `return *pointer;` |

</details>

<details><summary>Zephyr RTOS · 数组越界 · <code>contract3_independent_web_array_oob.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_01` | 11–15 | 14 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_02` | 17–22 | 21 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 24–30 | 29 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_04` | 32–37 | 36 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_05` | 39–44 | 43 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_06` | 46–50 | 49 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_07` | 52–57 | 56 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_08` | 59–65 | 64 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_09` | 67–72 | 71 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_10` | 74–79 | 78 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 81–85 | 84 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 87–92 | 91 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_13` | 94–100 | 99 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 102–107 | 106 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_15` | 109–114 | 113 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_16` | 116–120 | 119 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_17` | 122–127 | 126 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_18` | 129–135 | 134 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_19` | 137–142 | 141 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_20` | 144–149 | 148 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_21` | 151–155 | 154 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 157–162 | 161 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_23` | 164–170 | 169 | `base-2` | `return total;` |
| `zephyr-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 172–177 | 176 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_25` | 179–184 | 183 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_26` | 186–190 | 189 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_27` | 192–197 | 196 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_28` | 199–205 | 204 | `base-2` | `return total;` |
| `zephyr-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_29` | 207–212 | 211 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_30` | 214–219 | 218 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_31` | 221–225 | 224 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_32` | 227–232 | 231 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_33` | 234–240 | 239 | `base-2` | `return total;` |
| `zephyr-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_34` | 242–247 | 246 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_35` | 249–254 | 253 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_36` | 256–260 | 259 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_37` | 262–267 | 266 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_38` | 269–275 | 274 | `base-2` | `return total;` |
| `zephyr-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_39` | 277–282 | 281 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_40` | 284–289 | 288 | `base-4` | `return *cursor;` |

</details>

<details><summary>Zephyr RTOS · 字符串溢出 · <code>contract3_independent_web_string_overflow.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_01` | 11–16 | 15 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_02` | 18–23 | 22 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_03` | 25–30 | 29 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_04` | 32–37 | 36 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_05` | 39–44 | 43 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_06` | 46–51 | 50 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_07` | 53–58 | 57 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_08` | 60–65 | 64 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_09` | 67–72 | 71 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_10` | 74–79 | 78 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_11` | 81–86 | 85 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_12` | 88–93 | 92 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 95–100 | 99 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_14` | 102–107 | 106 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_15` | 109–114 | 113 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_16` | 116–121 | 120 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_17` | 123–128 | 127 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_18` | 130–135 | 134 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_19` | 137–142 | 141 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_20` | 144–149 | 148 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_21` | 151–156 | 155 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_22` | 158–163 | 162 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_23` | 165–170 | 169 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_24` | 172–177 | 176 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_25` | 179–184 | 183 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_26` | 186–191 | 190 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_27` | 193–198 | 197 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_28` | 200–205 | 204 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 207–212 | 211 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_30` | 214–219 | 218 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_31` | 221–226 | 225 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_32` | 228–233 | 232 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_33` | 235–240 | 239 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_34` | 242–247 | 246 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_35` | 249–254 | 253 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_36` | 256–261 | 260 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_37` | 263–268 | 267 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_38` | 270–275 | 274 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_39` | 277–282 | 281 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_40` | 284–289 | 288 | `base-4` | `return target[0];` |

</details>

<details><summary>Zephyr RTOS · 空指针调用 · <code>contract3_independent_web_null_pointer.c</code>（282 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 11–15 | 14 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 17–22 | 21 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 24–28 | 27 | `base-2` | `return contract3_read(0);` |
| `zephyr-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_04` | 30–35 | 34 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_05` | 37–43 | 42 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 45–49 | 48 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 51–56 | 55 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_08` | 58–62 | 61 | `base-2` | `return contract3_read(0);` |
| `zephyr-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_09` | 64–69 | 68 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_10` | 71–77 | 76 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_11` | 79–83 | 82 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 85–90 | 89 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_13` | 92–96 | 95 | `base-2` | `return contract3_read(0);` |
| `zephyr-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 98–103 | 102 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_15` | 105–111 | 110 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_16` | 113–117 | 116 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_17` | 119–124 | 123 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_18` | 126–130 | 129 | `base-2` | `return contract3_read(0);` |
| `zephyr-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_19` | 132–137 | 136 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_20` | 139–145 | 144 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_21` | 147–151 | 150 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_22` | 153–158 | 157 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_23` | 160–164 | 163 | `base-2` | `return contract3_read(&value);` |
| `zephyr-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_24` | 166–171 | 170 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_25` | 173–179 | 178 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 181–185 | 184 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 187–192 | 191 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_28` | 194–198 | 197 | `base-2` | `return contract3_read(&value);` |
| `zephyr-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 200–205 | 204 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_30` | 207–213 | 212 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_31` | 215–219 | 218 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_32` | 221–226 | 225 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_33` | 228–232 | 231 | `base-2` | `return contract3_read(&value);` |
| `zephyr-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_34` | 234–239 | 238 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_35` | 241–247 | 246 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_36` | 249–253 | 252 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_37` | 255–260 | 259 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_38` | 262–266 | 265 | `base-2` | `return contract3_read(&value);` |
| `zephyr-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_39` | 268–273 | 272 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 275–281 | 280 | `base-4` | `return *pointer;` |

</details>

<details><summary>RT-Thread · 数组越界 · <code>contract3_independent_web_array_oob.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_01` | 11–15 | 14 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_02` | 17–22 | 21 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 24–30 | 29 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_04` | 32–37 | 36 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_05` | 39–44 | 43 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_06` | 46–50 | 49 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_07` | 52–57 | 56 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_08` | 59–65 | 64 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_09` | 67–72 | 71 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_10` | 74–79 | 78 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 81–85 | 84 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 87–92 | 91 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_13` | 94–100 | 99 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 102–107 | 106 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_15` | 109–114 | 113 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_16` | 116–120 | 119 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_17` | 122–127 | 126 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_18` | 129–135 | 134 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_19` | 137–142 | 141 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_20` | 144–149 | 148 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_21` | 151–155 | 154 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 157–162 | 161 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_23` | 164–170 | 169 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 172–177 | 176 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_25` | 179–184 | 183 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_26` | 186–190 | 189 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_27` | 192–197 | 196 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_28` | 199–205 | 204 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_29` | 207–212 | 211 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_30` | 214–219 | 218 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_31` | 221–225 | 224 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_32` | 227–232 | 231 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_33` | 234–240 | 239 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_34` | 242–247 | 246 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_35` | 249–254 | 253 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_36` | 256–260 | 259 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_37` | 262–267 | 266 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_38` | 269–275 | 274 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_39` | 277–282 | 281 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_40` | 284–289 | 288 | `base-4` | `return *cursor;` |

</details>

<details><summary>RT-Thread · 字符串溢出 · <code>contract3_independent_web_string_overflow.c</code>（290 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_01` | 11–16 | 15 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_02` | 18–23 | 22 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_03` | 25–30 | 29 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_04` | 32–37 | 36 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_05` | 39–44 | 43 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_06` | 46–51 | 50 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_07` | 53–58 | 57 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_08` | 60–65 | 64 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_09` | 67–72 | 71 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_10` | 74–79 | 78 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_11` | 81–86 | 85 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_12` | 88–93 | 92 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 95–100 | 99 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_14` | 102–107 | 106 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_15` | 109–114 | 113 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_16` | 116–121 | 120 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_17` | 123–128 | 127 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_18` | 130–135 | 134 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_19` | 137–142 | 141 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_20` | 144–149 | 148 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_21` | 151–156 | 155 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_22` | 158–163 | 162 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_23` | 165–170 | 169 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_24` | 172–177 | 176 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_25` | 179–184 | 183 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_26` | 186–191 | 190 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_27` | 193–198 | 197 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_28` | 200–205 | 204 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 207–212 | 211 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_30` | 214–219 | 218 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_31` | 221–226 | 225 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_32` | 228–233 | 232 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_33` | 235–240 | 239 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_34` | 242–247 | 246 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_35` | 249–254 | 253 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_36` | 256–261 | 260 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_37` | 263–268 | 267 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_38` | 270–275 | 274 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_39` | 277–282 | 281 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_40` | 284–289 | 288 | `base-4` | `return target[0];` |

</details>

<details><summary>RT-Thread · 空指针调用 · <code>contract3_independent_web_null_pointer.c</code>（282 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 11–15 | 14 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 17–22 | 21 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 24–28 | 27 | `base-2` | `return contract3_read(0);` |
| `rt_thread-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_04` | 30–35 | 34 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_05` | 37–43 | 42 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 45–49 | 48 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 51–56 | 55 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_08` | 58–62 | 61 | `base-2` | `return contract3_read(0);` |
| `rt_thread-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_09` | 64–69 | 68 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_10` | 71–77 | 76 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_11` | 79–83 | 82 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 85–90 | 89 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_13` | 92–96 | 95 | `base-2` | `return contract3_read(0);` |
| `rt_thread-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 98–103 | 102 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_15` | 105–111 | 110 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_16` | 113–117 | 116 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_17` | 119–124 | 123 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_18` | 126–130 | 129 | `base-2` | `return contract3_read(0);` |
| `rt_thread-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_19` | 132–137 | 136 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_20` | 139–145 | 144 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_21` | 147–151 | 150 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_22` | 153–158 | 157 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_23` | 160–164 | 163 | `base-2` | `return contract3_read(&value);` |
| `rt_thread-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_24` | 166–171 | 170 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_25` | 173–179 | 178 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 181–185 | 184 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 187–192 | 191 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_28` | 194–198 | 197 | `base-2` | `return contract3_read(&value);` |
| `rt_thread-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 200–205 | 204 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_30` | 207–213 | 212 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_31` | 215–219 | 218 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_32` | 221–226 | 225 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_33` | 228–232 | 231 | `base-2` | `return contract3_read(&value);` |
| `rt_thread-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_34` | 234–239 | 238 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_35` | 241–247 | 246 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_36` | 249–253 | 252 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_37` | 255–260 | 259 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_38` | 262–266 | 265 | `base-2` | `return contract3_read(&value);` |
| `rt_thread-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_39` | 268–273 | 272 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 275–281 | 280 | `base-4` | `return *pointer;` |

</details>

### 平衡验收集（加难版）（`balanced-hard`）

<details><summary>FreeRTOS Kernel · 数组越界 · <code>contract3_balanced_hard_web_array_oob.c</code>（289 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_01` | 12–16 | 15 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_02` | 18–23 | 22 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_03` | 25–30 | 29 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_04` | 32–38 | 37 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_05` | 40–45 | 44 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_06` | 47–53 | 52 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_07` | 55–59 | 58 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_08` | 61–67 | 66 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_09` | 69–75 | 74 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_10` | 77–81 | 80 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `freertos_kernel-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_11` | 83–87 | 86 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_12` | 89–93 | 92 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_13` | 95–99 | 98 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `freertos_kernel-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 101–106 | 105 | `hard-3` | `return end[0];` |
| `freertos_kernel-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_15` | 108–113 | 112 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_16` | 115–119 | 118 | `hard-4` | `return values[4];` |
| `freertos_kernel-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_17` | 121–125 | 124 | `hard-4` | `return values[3];` |
| `freertos_kernel-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_18` | 127–132 | 131 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_19` | 134–140 | 139 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_20` | 142–148 | 147 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_21` | 150–156 | 155 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_22` | 158–163 | 162 | `hard-3` | `return end[0];` |
| `freertos_kernel-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_23` | 165–171 | 170 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_24` | 173–179 | 178 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_25` | 181–186 | 185 | `hard-3` | `return end[-1];` |
| `freertos_kernel-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_26` | 188–194 | 193 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_27` | 196–200 | 199 | `hard-4` | `return values[4];` |
| `freertos_kernel-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_28` | 202–207 | 206 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_29` | 209–214 | 213 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_30` | 216–222 | 221 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_31` | 224–228 | 227 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `freertos_kernel-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_32` | 230–234 | 233 | `hard-4` | `return values[3];` |
| `freertos_kernel-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_33` | 236–241 | 240 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_34` | 243–247 | 246 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_35` | 249–253 | 252 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `freertos_kernel-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_36` | 255–260 | 259 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_37` | 262–267 | 266 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_38` | 269–274 | 273 | `hard-3` | `return end[-1];` |
| `freertos_kernel-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_39` | 276–282 | 281 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_40` | 284–288 | 287 | `base-0` | `return values[2];` |

</details>

<details><summary>FreeRTOS Kernel · 字符串溢出 · <code>contract3_balanced_hard_web_string_overflow.c</code>（293 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_01` | 12–17 | 16 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_02` | 19–24 | 23 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_03` | 26–31 | 30 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_04` | 33–38 | 37 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_05` | 40–45 | 44 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_06` | 47–52 | 51 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_07` | 54–59 | 58 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_08` | 61–66 | 65 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_09` | 68–73 | 72 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_10` | 75–80 | 79 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_11` | 82–87 | 86 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_12` | 89–94 | 93 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_13` | 96–102 | 101 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_14` | 104–109 | 108 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_15` | 111–116 | 115 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_16` | 118–123 | 122 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_17` | 125–130 | 129 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_18` | 132–137 | 136 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_19` | 139–144 | 143 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_20` | 146–151 | 150 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_21` | 153–158 | 157 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_22` | 160–165 | 164 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_23` | 167–172 | 171 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_24` | 174–179 | 178 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_25` | 181–186 | 185 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_26` | 188–193 | 192 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_27` | 195–200 | 199 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_28` | 202–207 | 206 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_29` | 209–214 | 213 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_30` | 216–221 | 220 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_31` | 223–228 | 227 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_32` | 230–235 | 234 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_33` | 237–242 | 241 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_34` | 244–249 | 248 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_35` | 251–256 | 255 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_36` | 258–264 | 263 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_37` | 266–271 | 270 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_38` | 273–278 | 277 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_39` | 280–285 | 284 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_40` | 287–292 | 291 | `base-0` | `return target[0];` |

</details>

<details><summary>FreeRTOS Kernel · 空指针调用 · <code>contract3_balanced_hard_web_null_pointer.c</code>（301 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_01` | 12–16 | 15 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_02` | 18–22 | 21 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_03` | 24–30 | 29 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_04` | 32–37 | 36 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_05` | 39–45 | 44 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_06` | 47–53 | 52 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_07` | 55–61 | 60 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_08` | 63–69 | 68 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_09` | 71–77 | 76 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_10` | 79–84 | 83 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_11` | 86–90 | 89 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_12` | 92–97 | 96 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_13` | 99–104 | 103 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_14` | 106–111 | 110 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_15` | 113–119 | 118 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_16` | 121–125 | 124 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_17` | 127–133 | 132 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_18` | 135–140 | 139 | `hard-1` | `return pointer ? *pointer : 0;` |
| `freertos_kernel-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_19` | 142–147 | 146 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_20` | 149–155 | 154 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_21` | 157–162 | 161 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_22` | 164–168 | 167 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_23` | 170–176 | 175 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_24` | 178–184 | 183 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_25` | 186–191 | 190 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_26` | 193–199 | 198 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_27` | 201–205 | 204 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_28` | 207–213 | 212 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_29` | 215–219 | 218 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_30` | 221–225 | 224 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_31` | 227–233 | 232 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_32` | 235–240 | 239 | `hard-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_33` | 242–248 | 247 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_34` | 250–255 | 254 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_35` | 257–263 | 262 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_36` | 265–270 | 269 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_37` | 272–278 | 277 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_38` | 280–286 | 285 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_39` | 288–293 | 292 | `hard-1` | `return pointer ? *pointer : 0;` |
| `freertos_kernel-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_40` | 295–300 | 299 | `hard-1` | `return *pointer;` |

</details>

<details><summary>Zephyr RTOS · 数组越界 · <code>contract3_balanced_hard_web_array_oob.c</code>（289 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_01` | 12–18 | 17 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_02` | 20–24 | 23 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `zephyr-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_03` | 26–31 | 30 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_04` | 33–37 | 36 | `hard-4` | `return values[3];` |
| `zephyr-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_05` | 39–43 | 42 | `hard-4` | `return values[3];` |
| `zephyr-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_06` | 45–50 | 49 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_07` | 52–58 | 57 | `hard-1` | `return total;` |
| `zephyr-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_08` | 60–66 | 65 | `hard-1` | `return total;` |
| `zephyr-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_09` | 68–72 | 71 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_10` | 74–79 | 78 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_11` | 81–87 | 86 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_12` | 89–94 | 93 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_13` | 96–102 | 101 | `hard-1` | `return total;` |
| `zephyr-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_14` | 104–109 | 108 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_15` | 111–116 | 115 | `base-1` | `return values[index];` |
| `zephyr-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_16` | 118–122 | 121 | `base-0` | `return values[2];` |
| `zephyr-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_17` | 124–128 | 127 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `zephyr-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_18` | 130–136 | 135 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_19` | 138–144 | 143 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_20` | 146–151 | 150 | `hard-3` | `return end[-1];` |
| `zephyr-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_21` | 153–158 | 157 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_22` | 160–165 | 164 | `base-3` | `return values[index];` |
| `zephyr-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_23` | 167–171 | 170 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_24` | 173–177 | 176 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_25` | 179–184 | 183 | `hard-3` | `return end[0];` |
| `zephyr-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_26` | 186–190 | 189 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `zephyr-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_27` | 192–197 | 196 | `hard-3` | `return end[-1];` |
| `zephyr-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_28` | 199–205 | 204 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_29` | 207–212 | 211 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_30` | 214–218 | 217 | `base-0` | `return values[3];` |
| `zephyr-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_31` | 220–224 | 223 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `zephyr-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_32` | 226–232 | 231 | `hard-1` | `return total;` |
| `zephyr-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_33` | 234–239 | 238 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_34` | 241–247 | 246 | `base-2` | `return total;` |
| `zephyr-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_35` | 249–253 | 252 | `base-0` | `return values[2];` |
| `zephyr-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_36` | 255–261 | 260 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_37` | 263–267 | 266 | `hard-4` | `return values[4];` |
| `zephyr-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_38` | 269–273 | 272 | `hard-4` | `return values[4];` |
| `zephyr-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_39` | 275–280 | 279 | `hard-3` | `return end[0];` |
| `zephyr-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_40` | 282–288 | 287 | `base-2` | `return total;` |

</details>

<details><summary>Zephyr RTOS · 字符串溢出 · <code>contract3_balanced_hard_web_string_overflow.c</code>（293 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_01` | 12–17 | 16 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_02` | 19–24 | 23 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_03` | 26–31 | 30 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_04` | 33–38 | 37 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_05` | 40–45 | 44 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_06` | 47–52 | 51 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_07` | 54–59 | 58 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_08` | 61–66 | 65 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_09` | 68–73 | 72 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_10` | 75–80 | 79 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_11` | 82–87 | 86 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_12` | 89–94 | 93 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_13` | 96–101 | 100 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_14` | 103–108 | 107 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_15` | 110–116 | 115 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_16` | 118–123 | 122 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_17` | 125–130 | 129 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_18` | 132–137 | 136 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_19` | 139–144 | 143 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_20` | 146–151 | 150 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_21` | 153–159 | 158 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_22` | 161–166 | 165 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_23` | 168–173 | 172 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_24` | 175–180 | 179 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_25` | 182–187 | 186 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_26` | 189–194 | 193 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_27` | 196–201 | 200 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_28` | 203–208 | 207 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_29` | 210–215 | 214 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_30` | 217–222 | 221 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_31` | 224–229 | 228 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_32` | 231–236 | 235 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_33` | 238–243 | 242 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_34` | 245–250 | 249 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_35` | 252–257 | 256 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_36` | 259–264 | 263 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_37` | 266–271 | 270 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_38` | 273–278 | 277 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_39` | 280–285 | 284 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_40` | 287–292 | 291 | `hard-0` | `return target[0];` |

</details>

<details><summary>Zephyr RTOS · 空指针调用 · <code>contract3_balanced_hard_web_null_pointer.c</code>（301 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_01` | 12–18 | 17 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_02` | 20–24 | 23 | `base-0` | `return value;` |
| `zephyr-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_03` | 26–31 | 30 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_04` | 33–38 | 37 | `hard-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_05` | 40–45 | 44 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_06` | 47–53 | 52 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_07` | 55–59 | 58 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_08` | 61–67 | 66 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_09` | 69–73 | 72 | `base-0` | `return value;` |
| `zephyr-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_10` | 75–81 | 80 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_11` | 83–87 | 86 | `base-0` | `return value;` |
| `zephyr-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_12` | 89–93 | 92 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_13` | 95–100 | 99 | `hard-1` | `return pointer ? *pointer : 0;` |
| `zephyr-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_14` | 102–108 | 107 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_15` | 110–116 | 115 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_16` | 118–123 | 122 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_17` | 125–130 | 129 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_18` | 132–138 | 137 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_19` | 140–146 | 145 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_20` | 148–153 | 152 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_21` | 155–160 | 159 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_22` | 162–168 | 167 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_23` | 170–176 | 175 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_24` | 178–184 | 183 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_25` | 186–191 | 190 | `hard-1` | `return pointer ? *pointer : 0;` |
| `zephyr-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_26` | 193–198 | 197 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_27` | 200–205 | 204 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_28` | 207–211 | 210 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_29` | 213–219 | 218 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_30` | 221–226 | 225 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_31` | 228–233 | 232 | `hard-1` | `return *pointer;` |
| `zephyr-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_32` | 235–241 | 240 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_33` | 243–247 | 246 | `base-0` | `return value;` |
| `zephyr-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_34` | 249–255 | 254 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_35` | 257–263 | 262 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_36` | 265–271 | 270 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_37` | 273–279 | 278 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_38` | 281–285 | 284 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_39` | 287–292 | 291 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_40` | 294–300 | 299 | `hard-4` | `return *box.pointer;` |

</details>

<details><summary>RT-Thread · 数组越界 · <code>contract3_balanced_hard_web_array_oob.c</code>（289 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_01` | 12–17 | 16 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_02` | 19–23 | 22 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_03` | 25–31 | 30 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_04` | 33–39 | 38 | `hard-1` | `return total;` |
| `rt_thread-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_05` | 41–46 | 45 | `hard-3` | `return end[-1];` |
| `rt_thread-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_06` | 48–54 | 53 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_07` | 56–62 | 61 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_08` | 64–68 | 67 | `hard-4` | `return values[3];` |
| `rt_thread-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_09` | 70–74 | 73 | `hard-4` | `return values[4];` |
| `rt_thread-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_10` | 76–81 | 80 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_11` | 83–89 | 88 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_12` | 91–97 | 96 | `hard-1` | `return total;` |
| `rt_thread-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_13` | 99–103 | 102 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_14` | 105–109 | 108 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_15` | 111–116 | 115 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_16` | 118–123 | 122 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_17` | 125–129 | 128 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_18` | 131–135 | 134 | `hard-4` | `return values[3];` |
| `rt_thread-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_19` | 137–142 | 141 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_20` | 144–148 | 147 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_21` | 150–156 | 155 | `hard-1` | `return total;` |
| `rt_thread-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_22` | 158–163 | 162 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_23` | 165–170 | 169 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_24` | 172–177 | 176 | `hard-3` | `return end[-1];` |
| `rt_thread-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_25` | 179–184 | 183 | `hard-3` | `return end[0];` |
| `rt_thread-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_26` | 186–190 | 189 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `rt_thread-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_27` | 192–196 | 195 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_28` | 198–203 | 202 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_29` | 205–211 | 210 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_30` | 213–218 | 217 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_31` | 220–226 | 225 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_32` | 228–232 | 231 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `rt_thread-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_33` | 234–240 | 239 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_34` | 242–246 | 245 | `hard-4` | `return values[4];` |
| `rt_thread-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_35` | 248–252 | 251 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `rt_thread-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_36` | 254–259 | 258 | `hard-3` | `return end[0];` |
| `rt_thread-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_37` | 261–267 | 266 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_38` | 269–273 | 272 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `rt_thread-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_39` | 275–280 | 279 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_40` | 282–288 | 287 | `hard-1` | `return total;` |

</details>

<details><summary>RT-Thread · 字符串溢出 · <code>contract3_balanced_hard_web_string_overflow.c</code>（293 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_01` | 12–18 | 17 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_02` | 20–25 | 24 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_03` | 27–32 | 31 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_04` | 34–39 | 38 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_05` | 41–46 | 45 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_06` | 48–53 | 52 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_07` | 55–60 | 59 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_08` | 62–67 | 66 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_09` | 69–74 | 73 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_10` | 76–81 | 80 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_11` | 83–88 | 87 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_12` | 90–95 | 94 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_13` | 97–102 | 101 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_14` | 104–110 | 109 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_15` | 112–117 | 116 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_16` | 119–124 | 123 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_17` | 126–131 | 130 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_18` | 133–138 | 137 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_19` | 140–145 | 144 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_20` | 147–152 | 151 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_21` | 154–159 | 158 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_22` | 161–166 | 165 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_23` | 168–173 | 172 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_24` | 175–180 | 179 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_25` | 182–187 | 186 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_26` | 189–194 | 193 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_27` | 196–201 | 200 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_28` | 203–208 | 207 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_29` | 210–215 | 214 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_30` | 217–222 | 221 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_31` | 224–229 | 228 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_32` | 231–236 | 235 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_33` | 238–243 | 242 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_34` | 245–250 | 249 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_35` | 252–257 | 256 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_36` | 259–264 | 263 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_37` | 266–271 | 270 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_38` | 273–278 | 277 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_39` | 280–285 | 284 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_40` | 287–292 | 291 | `hard-0` | `return target[0];` |

</details>

<details><summary>RT-Thread · 空指针调用 · <code>contract3_balanced_hard_web_null_pointer.c</code>（301 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 12–16 | 15 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_02` | 18–24 | 23 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_03` | 26–31 | 30 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_04` | 33–38 | 37 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_05` | 40–46 | 45 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_06` | 48–53 | 52 | `hard-1` | `return pointer ? *pointer : 0;` |
| `rt_thread-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_07` | 55–59 | 58 | `base-0` | `return value;` |
| `rt_thread-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_08` | 61–66 | 65 | `hard-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_09` | 68–74 | 73 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_10` | 76–81 | 80 | `hard-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_11` | 83–87 | 86 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_12` | 89–94 | 93 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_13` | 96–102 | 101 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_14` | 104–108 | 107 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_15` | 110–114 | 113 | `base-0` | `return value;` |
| `rt_thread-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_16` | 116–121 | 120 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_17` | 123–129 | 128 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_18` | 131–137 | 136 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_19` | 139–145 | 144 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_20` | 147–153 | 152 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_21` | 155–160 | 159 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_22` | 162–167 | 166 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_23` | 169–173 | 172 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_24` | 175–181 | 180 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_25` | 183–189 | 188 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_26` | 191–197 | 196 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_27` | 199–205 | 204 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_28` | 207–211 | 210 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_29` | 213–219 | 218 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_30` | 221–226 | 225 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_31` | 228–233 | 232 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_32` | 235–239 | 238 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_33` | 241–247 | 246 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_34` | 249–255 | 254 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_35` | 257–263 | 262 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_36` | 265–271 | 270 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_37` | 273–279 | 278 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_38` | 281–286 | 285 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_39` | 288–293 | 292 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_40` | 295–300 | 299 | `hard-1` | `return pointer ? *pointer : 0;` |

</details>

### 独立压力集（加难版）（`independent-hard`）

<details><summary>FreeRTOS Kernel · 数组越界 · <code>contract3_independent_hard_web_array_oob.c</code>（291 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_01` | 12–18 | 17 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_02` | 20–24 | 23 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `freertos_kernel-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_03` | 26–31 | 30 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_04` | 33–39 | 38 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_05` | 41–47 | 46 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_06` | 49–55 | 54 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_07` | 57–62 | 61 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_08` | 64–68 | 67 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `freertos_kernel-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_09` | 70–74 | 73 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `freertos_kernel-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_10` | 76–82 | 81 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_11` | 84–90 | 89 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_12` | 92–96 | 95 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `freertos_kernel-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_13` | 98–102 | 101 | `base-0` | `return values[2];` |
| `freertos_kernel-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_14` | 104–110 | 109 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_15` | 112–117 | 116 | `hard-3` | `return end[-1];` |
| `freertos_kernel-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_16` | 119–124 | 123 | `base-3` | `return values[index];` |
| `freertos_kernel-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_17` | 126–130 | 129 | `hard-4` | `return values[4];` |
| `freertos_kernel-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_18` | 132–138 | 137 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_19` | 140–146 | 145 | `base-2` | `return total;` |
| `freertos_kernel-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_20` | 148–152 | 151 | `hard-4` | `return values[3];` |
| `freertos_kernel-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_21` | 154–158 | 157 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `freertos_kernel-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_22` | 160–165 | 164 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_23` | 167–173 | 172 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_24` | 175–179 | 178 | `hard-4` | `return values[4];` |
| `freertos_kernel-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_25` | 181–186 | 185 | `hard-3` | `return end[-1];` |
| `freertos_kernel-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_26` | 188–192 | 191 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `freertos_kernel-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_27` | 194–199 | 198 | `hard-3` | `return end[0];` |
| `freertos_kernel-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_28` | 201–205 | 204 | `hard-4` | `return values[3];` |
| `freertos_kernel-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_29` | 207–213 | 212 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_30` | 215–220 | 219 | `hard-3` | `return end[-1];` |
| `freertos_kernel-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_31` | 222–228 | 227 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_32` | 230–235 | 234 | `base-1` | `return values[index];` |
| `freertos_kernel-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_33` | 237–243 | 242 | `hard-0` | `return values[index];` |
| `freertos_kernel-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_34` | 245–250 | 249 | `hard-3` | `return end[0];` |
| `freertos_kernel-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_35` | 252–257 | 256 | `hard-3` | `return end[0];` |
| `freertos_kernel-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_36` | 259–264 | 263 | `base-4` | `return *cursor;` |
| `freertos_kernel-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_37` | 266–270 | 269 | `hard-4` | `return values[3];` |
| `freertos_kernel-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_38` | 272–276 | 275 | `base-0` | `return values[3];` |
| `freertos_kernel-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_39` | 278–284 | 283 | `hard-1` | `return total;` |
| `freertos_kernel-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_40` | 286–290 | 289 | `hard-4` | `return values[4];` |

</details>

<details><summary>FreeRTOS Kernel · 字符串溢出 · <code>contract3_independent_hard_web_string_overflow.c</code>（294 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_01` | 12–17 | 16 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_02` | 19–24 | 23 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_03` | 26–31 | 30 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_04` | 33–38 | 37 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_05` | 40–45 | 44 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_06` | 47–52 | 51 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_07` | 54–59 | 58 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_08` | 61–66 | 65 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_09` | 68–73 | 72 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_10` | 75–80 | 79 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_11` | 82–87 | 86 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_12` | 89–94 | 93 | `base-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_13` | 96–102 | 101 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_14` | 104–109 | 108 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_15` | 111–116 | 115 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_16` | 118–123 | 122 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_17` | 125–130 | 129 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_18` | 132–137 | 136 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_19` | 139–144 | 143 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_20` | 146–151 | 150 | `base-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_21` | 153–158 | 157 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_22` | 160–165 | 164 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_23` | 167–172 | 171 | `base-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_24` | 174–179 | 178 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_25` | 181–186 | 185 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_26` | 188–194 | 193 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_27` | 196–201 | 200 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_28` | 203–208 | 207 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_29` | 210–215 | 214 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_30` | 217–222 | 221 | `hard-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_31` | 224–229 | 228 | `hard-4` | `return target[0];` |
| `freertos_kernel-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_32` | 231–237 | 236 | `hard-2` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_33` | 239–244 | 243 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_34` | 246–251 | 250 | `base-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_35` | 253–258 | 257 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_36` | 260–265 | 264 | `hard-3` | `return target[0];` |
| `freertos_kernel-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_37` | 267–272 | 271 | `base-1` | `return target[0];` |
| `freertos_kernel-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_38` | 274–279 | 278 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_39` | 281–286 | 285 | `hard-0` | `return target[0];` |
| `freertos_kernel-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_40` | 288–293 | 292 | `base-0` | `return target[0];` |

</details>

<details><summary>FreeRTOS Kernel · 空指针调用 · <code>contract3_independent_hard_web_null_pointer.c</code>（307 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `freertos_kernel-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_01` | 12–17 | 16 | `hard-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_02` | 19–24 | 23 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_03` | 26–32 | 31 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_04` | 34–40 | 39 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_05` | 42–48 | 47 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_06` | 50–56 | 55 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_07` | 58–63 | 62 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_08` | 65–70 | 69 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_09` | 72–78 | 77 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_10` | 80–85 | 84 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_11` | 87–92 | 91 | `hard-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_12` | 94–100 | 99 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_13` | 102–108 | 107 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_14` | 110–116 | 115 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_15` | 118–124 | 123 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_16` | 126–132 | 131 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_17` | 134–138 | 137 | `base-2` | `return contract3_read(0);` |
| `freertos_kernel-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_18` | 140–145 | 144 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_19` | 147–151 | 150 | `base-0` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_20` | 153–159 | 158 | `hard-2` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_21` | 161–166 | 165 | `hard-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_22` | 168–173 | 172 | `hard-1` | `return pointer ? *pointer : 0;` |
| `freertos_kernel-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_23` | 175–181 | 180 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_24` | 183–189 | 188 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_25` | 191–196 | 195 | `base-1` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_26` | 198–203 | 202 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_27` | 205–211 | 210 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_28` | 213–219 | 218 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_29` | 221–226 | 225 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_30` | 228–234 | 233 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_31` | 236–242 | 241 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_32` | 244–250 | 249 | `base-4` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_33` | 252–257 | 256 | `hard-1` | `return pointer ? *pointer : 0;` |
| `freertos_kernel-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_34` | 259–265 | 264 | `hard-4` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_35` | 267–273 | 272 | `hard-0` | `return *pointer + value;` |
| `freertos_kernel-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_36` | 275–280 | 279 | `hard-3` | `return *pointer;` |
| `freertos_kernel-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_37` | 282–286 | 285 | `base-2` | `return contract3_read(&value);` |
| `freertos_kernel-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_38` | 288–293 | 292 | `base-3` | `return *box.pointer;` |
| `freertos_kernel-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_39` | 295–299 | 298 | `base-0` | `return value;` |
| `freertos_kernel-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_40` | 301–306 | 305 | `hard-1` | `return pointer ? *pointer : 0;` |

</details>

<details><summary>Zephyr RTOS · 数组越界 · <code>contract3_independent_hard_web_array_oob.c</code>（291 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_01` | 12–18 | 17 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_02` | 20–24 | 23 | `hard-4` | `return values[4];` |
| `zephyr-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_03` | 26–32 | 31 | `base-2` | `return total;` |
| `zephyr-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_04` | 34–38 | 37 | `hard-4` | `return values[3];` |
| `zephyr-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_05` | 40–46 | 45 | `base-2` | `return total;` |
| `zephyr-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_06` | 48–53 | 52 | `hard-3` | `return end[0];` |
| `zephyr-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_07` | 55–59 | 58 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `zephyr-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_08` | 61–67 | 66 | `hard-1` | `return total;` |
| `zephyr-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_09` | 69–75 | 74 | `hard-1` | `return total;` |
| `zephyr-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_10` | 77–83 | 82 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_11` | 85–89 | 88 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `zephyr-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_12` | 91–95 | 94 | `hard-4` | `return values[3];` |
| `zephyr-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_13` | 97–102 | 101 | `hard-3` | `return end[-1];` |
| `zephyr-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_14` | 104–109 | 108 | `hard-3` | `return end[0];` |
| `zephyr-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_15` | 111–117 | 116 | `hard-1` | `return total;` |
| `zephyr-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_16` | 119–123 | 122 | `hard-4` | `return values[3];` |
| `zephyr-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_17` | 125–131 | 130 | `hard-1` | `return total;` |
| `zephyr-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_18` | 133–138 | 137 | `hard-3` | `return end[-1];` |
| `zephyr-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_19` | 140–146 | 145 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_20` | 148–154 | 153 | `hard-1` | `return total;` |
| `zephyr-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_21` | 156–160 | 159 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `zephyr-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_22` | 162–166 | 165 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `zephyr-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_23` | 168–172 | 171 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `zephyr-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_24` | 174–179 | 178 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_25` | 181–187 | 186 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_26` | 189–195 | 194 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_27` | 197–202 | 201 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_28` | 204–209 | 208 | `hard-3` | `return end[-1];` |
| `zephyr-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_29` | 211–216 | 215 | `base-3` | `return values[index];` |
| `zephyr-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_30` | 218–224 | 223 | `hard-1` | `return total;` |
| `zephyr-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_31` | 226–231 | 230 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_32` | 233–237 | 236 | `base-0` | `return values[3];` |
| `zephyr-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_33` | 239–243 | 242 | `hard-4` | `return values[4];` |
| `zephyr-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_34` | 245–251 | 250 | `hard-0` | `return values[index];` |
| `zephyr-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_35` | 253–257 | 256 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `zephyr-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_36` | 259–264 | 263 | `base-4` | `return *cursor;` |
| `zephyr-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_37` | 266–271 | 270 | `base-1` | `return values[index];` |
| `zephyr-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_38` | 273–277 | 276 | `hard-4` | `return values[4];` |
| `zephyr-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_39` | 279–284 | 283 | `hard-3` | `return end[0];` |
| `zephyr-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_40` | 286–290 | 289 | `base-0` | `return values[2];` |

</details>

<details><summary>Zephyr RTOS · 字符串溢出 · <code>contract3_independent_hard_web_string_overflow.c</code>（294 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_01` | 12–17 | 16 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_02` | 19–24 | 23 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_03` | 26–31 | 30 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_04` | 33–38 | 37 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_05` | 40–45 | 44 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_06` | 47–52 | 51 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_07` | 54–59 | 58 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_08` | 61–66 | 65 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_09` | 68–73 | 72 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_10` | 75–80 | 79 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_11` | 82–87 | 86 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_12` | 89–94 | 93 | `base-4` | `return target[0];` |
| `zephyr-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_13` | 96–101 | 100 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_14` | 103–108 | 107 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_15` | 110–115 | 114 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_16` | 117–122 | 121 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_17` | 124–129 | 128 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_18` | 131–136 | 135 | `base-2` | `return target[0];` |
| `zephyr-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_19` | 138–143 | 142 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_20` | 145–150 | 149 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_21` | 152–158 | 157 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_22` | 160–166 | 165 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_23` | 168–173 | 172 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_24` | 175–180 | 179 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_25` | 182–188 | 187 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_26` | 190–195 | 194 | `base-1` | `return target[0];` |
| `zephyr-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_27` | 197–202 | 201 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_28` | 204–209 | 208 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_29` | 211–216 | 215 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_30` | 218–223 | 222 | `hard-2` | `return target[0];` |
| `zephyr-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_31` | 225–230 | 229 | `hard-3` | `return target[0];` |
| `zephyr-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_32` | 232–237 | 236 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_33` | 239–244 | 243 | `hard-1` | `return target[0];` |
| `zephyr-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_34` | 246–251 | 250 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_35` | 253–258 | 257 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_36` | 260–265 | 264 | `hard-4` | `return target[0];` |
| `zephyr-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_37` | 267–272 | 271 | `base-3` | `return target[0];` |
| `zephyr-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_38` | 274–279 | 278 | `hard-0` | `return target[0];` |
| `zephyr-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_39` | 281–286 | 285 | `base-0` | `return target[0];` |
| `zephyr-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_40` | 288–293 | 292 | `hard-1` | `return target[0];` |

</details>

<details><summary>Zephyr RTOS · 空指针调用 · <code>contract3_independent_hard_web_null_pointer.c</code>（307 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `zephyr-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_01` | 12–16 | 15 | `base-0` | `return *pointer;` |
| `zephyr-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_02` | 18–24 | 23 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_03` | 26–32 | 31 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_04` | 34–40 | 39 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_05` | 42–47 | 46 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_06` | 49–55 | 54 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_07` | 57–63 | 62 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_08` | 65–71 | 70 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_09` | 73–79 | 78 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_10` | 81–87 | 86 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_11` | 89–95 | 94 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_12` | 97–102 | 101 | `hard-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_13` | 104–110 | 109 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_14` | 112–118 | 117 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_15` | 120–126 | 125 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_16` | 128–134 | 133 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_17` | 136–142 | 141 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_18` | 144–149 | 148 | `hard-1` | `return *pointer;` |
| `zephyr-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_19` | 151–156 | 155 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_20` | 158–164 | 163 | `base-4` | `return *pointer;` |
| `zephyr-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_21` | 166–171 | 170 | `base-3` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_22` | 173–179 | 178 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_23` | 181–187 | 186 | `hard-2` | `return *pointer;` |
| `zephyr-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_24` | 189–194 | 193 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_25` | 196–201 | 200 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_26` | 203–209 | 208 | `hard-0` | `return *pointer + value;` |
| `zephyr-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_27` | 211–216 | 215 | `hard-1` | `return pointer ? *pointer : 0;` |
| `zephyr-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_28` | 218–223 | 222 | `hard-1` | `return *pointer;` |
| `zephyr-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_29` | 225–231 | 230 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_30` | 233–239 | 238 | `hard-4` | `return *box.pointer;` |
| `zephyr-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_31` | 241–245 | 244 | `base-2` | `return contract3_read(0);` |
| `zephyr-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_32` | 247–252 | 251 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_33` | 254–259 | 258 | `hard-1` | `return pointer ? *pointer : 0;` |
| `zephyr-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_34` | 261–266 | 265 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_35` | 268–272 | 271 | `base-0` | `return value;` |
| `zephyr-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_36` | 274–279 | 278 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_37` | 281–286 | 285 | `hard-3` | `return *pointer;` |
| `zephyr-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_38` | 288–293 | 292 | `base-1` | `return *pointer;` |
| `zephyr-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_39` | 295–300 | 299 | `hard-1` | `return pointer ? *pointer : 0;` |
| `zephyr-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_40` | 302–306 | 305 | `base-2` | `return contract3_read(&value);` |

</details>

<details><summary>RT-Thread · 数组越界 · <code>contract3_independent_hard_web_array_oob.c</code>（291 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-array_oob-positive-11` | **正例** | `contract3_independent_array_oob_case_01` | 12–18 | 17 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-positive-20` | **正例** | `contract3_independent_array_oob_case_02` | 20–24 | 23 | `hard-4` | `return values[4];` |
| `rt_thread-array_oob-positive-14` | **正例** | `contract3_independent_array_oob_case_03` | 26–31 | 30 | `hard-3` | `return end[0];` |
| `rt_thread-array_oob-positive-03` | **正例** | `contract3_independent_array_oob_case_04` | 33–39 | 38 | `base-2` | `return total;` |
| `rt_thread-array_oob-control-09` | 对照 | `contract3_independent_array_oob_case_05` | 41–46 | 45 | `hard-3` | `return end[-1];` |
| `rt_thread-array_oob-positive-12` | **正例** | `contract3_independent_array_oob_case_06` | 48–54 | 53 | `hard-1` | `return total;` |
| `rt_thread-array_oob-positive-18` | **正例** | `contract3_independent_array_oob_case_07` | 56–60 | 59 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `rt_thread-array_oob-control-11` | 对照 | `contract3_independent_array_oob_case_08` | 62–68 | 67 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-control-19` | 对照 | `contract3_independent_array_oob_case_09` | 70–75 | 74 | `hard-3` | `return end[-1];` |
| `rt_thread-array_oob-control-20` | 对照 | `contract3_independent_array_oob_case_10` | 77–81 | 80 | `hard-4` | `return values[3];` |
| `rt_thread-array_oob-positive-06` | **正例** | `contract3_independent_array_oob_case_11` | 83–89 | 88 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-positive-07` | **正例** | `contract3_independent_array_oob_case_12` | 91–97 | 96 | `hard-1` | `return total;` |
| `rt_thread-array_oob-positive-09` | **正例** | `contract3_independent_array_oob_case_13` | 99–104 | 103 | `hard-3` | `return end[0];` |
| `rt_thread-array_oob-control-07` | 对照 | `contract3_independent_array_oob_case_14` | 106–112 | 111 | `hard-1` | `return total;` |
| `rt_thread-array_oob-control-03` | 对照 | `contract3_independent_array_oob_case_15` | 114–120 | 119 | `base-2` | `return total;` |
| `rt_thread-array_oob-positive-05` | **正例** | `contract3_independent_array_oob_case_16` | 122–127 | 126 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-control-08` | 对照 | `contract3_independent_array_oob_case_17` | 129–133 | 132 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `rt_thread-array_oob-positive-01` | **正例** | `contract3_independent_array_oob_case_18` | 135–139 | 138 | `base-0` | `return values[3];` |
| `rt_thread-array_oob-positive-15` | **正例** | `contract3_independent_array_oob_case_19` | 141–145 | 144 | `hard-4` | `return values[4];` |
| `rt_thread-array_oob-positive-16` | **正例** | `contract3_independent_array_oob_case_20` | 147–153 | 152 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-positive-17` | **正例** | `contract3_independent_array_oob_case_21` | 155–161 | 160 | `hard-1` | `return total;` |
| `rt_thread-array_oob-control-12` | 对照 | `contract3_independent_array_oob_case_22` | 163–169 | 168 | `hard-1` | `return total;` |
| `rt_thread-array_oob-control-15` | 对照 | `contract3_independent_array_oob_case_23` | 171–175 | 174 | `hard-4` | `return values[3];` |
| `rt_thread-array_oob-control-04` | 对照 | `contract3_independent_array_oob_case_24` | 177–182 | 181 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-06` | 对照 | `contract3_independent_array_oob_case_25` | 184–190 | 189 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-control-02` | 对照 | `contract3_independent_array_oob_case_26` | 192–197 | 196 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-positive-19` | **正例** | `contract3_independent_array_oob_case_27` | 199–204 | 203 | `hard-3` | `return end[0];` |
| `rt_thread-array_oob-control-13` | 对照 | `contract3_independent_array_oob_case_28` | 206–210 | 209 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `rt_thread-array_oob-positive-10` | **正例** | `contract3_independent_array_oob_case_29` | 212–216 | 215 | `hard-4` | `return values[4];` |
| `rt_thread-array_oob-control-17` | 对照 | `contract3_independent_array_oob_case_30` | 218–224 | 223 | `hard-1` | `return total;` |
| `rt_thread-array_oob-positive-02` | **正例** | `contract3_independent_array_oob_case_31` | 226–231 | 230 | `base-1` | `return values[index];` |
| `rt_thread-array_oob-control-16` | 对照 | `contract3_independent_array_oob_case_32` | 233–239 | 238 | `hard-0` | `return values[index];` |
| `rt_thread-array_oob-control-14` | 对照 | `contract3_independent_array_oob_case_33` | 241–246 | 245 | `hard-3` | `return end[-1];` |
| `rt_thread-array_oob-control-10` | 对照 | `contract3_independent_array_oob_case_34` | 248–252 | 251 | `hard-4` | `return values[3];` |
| `rt_thread-array_oob-control-18` | 对照 | `contract3_independent_array_oob_case_35` | 254–258 | 257 | `hard-2` | `return values[sizeof(values) / sizeof(values[0]) - 1];` |
| `rt_thread-array_oob-control-05` | 对照 | `contract3_independent_array_oob_case_36` | 260–265 | 264 | `base-4` | `return *cursor;` |
| `rt_thread-array_oob-positive-04` | **正例** | `contract3_independent_array_oob_case_37` | 267–272 | 271 | `base-3` | `return values[index];` |
| `rt_thread-array_oob-control-01` | 对照 | `contract3_independent_array_oob_case_38` | 274–278 | 277 | `base-0` | `return values[2];` |
| `rt_thread-array_oob-positive-13` | **正例** | `contract3_independent_array_oob_case_39` | 280–284 | 283 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |
| `rt_thread-array_oob-positive-08` | **正例** | `contract3_independent_array_oob_case_40` | 286–290 | 289 | `hard-2` | `return values[sizeof(values) / sizeof(values[0])];` |

</details>

<details><summary>RT-Thread · 字符串溢出 · <code>contract3_independent_hard_web_string_overflow.c</code>（294 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-string_overflow-positive-12` | **正例** | `contract3_independent_string_overflow_case_01` | 12–17 | 16 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-control-08` | 对照 | `contract3_independent_string_overflow_case_02` | 19–25 | 24 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-control-05` | 对照 | `contract3_independent_string_overflow_case_03` | 27–32 | 31 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-10` | **正例** | `contract3_independent_string_overflow_case_04` | 34–39 | 38 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-19` | **正例** | `contract3_independent_string_overflow_case_05` | 41–46 | 45 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-control-06` | 对照 | `contract3_independent_string_overflow_case_06` | 48–53 | 52 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-control-03` | 对照 | `contract3_independent_string_overflow_case_07` | 55–60 | 59 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-16` | 对照 | `contract3_independent_string_overflow_case_08` | 62–67 | 66 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-control-13` | 对照 | `contract3_independent_string_overflow_case_09` | 69–75 | 74 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-control-17` | 对照 | `contract3_independent_string_overflow_case_10` | 77–82 | 81 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-control-09` | 对照 | `contract3_independent_string_overflow_case_11` | 84–89 | 88 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-07` | **正例** | `contract3_independent_string_overflow_case_12` | 91–96 | 95 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-11` | **正例** | `contract3_independent_string_overflow_case_13` | 98–103 | 102 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-02` | **正例** | `contract3_independent_string_overflow_case_14` | 105–110 | 109 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-04` | **正例** | `contract3_independent_string_overflow_case_15` | 112–117 | 116 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-17` | **正例** | `contract3_independent_string_overflow_case_16` | 119–124 | 123 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-control-15` | 对照 | `contract3_independent_string_overflow_case_17` | 126–131 | 130 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-02` | 对照 | `contract3_independent_string_overflow_case_18` | 133–138 | 137 | `base-1` | `return target[0];` |
| `rt_thread-string_overflow-control-04` | 对照 | `contract3_independent_string_overflow_case_19` | 140–145 | 144 | `base-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-06` | **正例** | `contract3_independent_string_overflow_case_20` | 147–152 | 151 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-14` | **正例** | `contract3_independent_string_overflow_case_21` | 154–159 | 158 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-13` | **正例** | `contract3_independent_string_overflow_case_22` | 161–166 | 165 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-20` | **正例** | `contract3_independent_string_overflow_case_23` | 168–173 | 172 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-19` | 对照 | `contract3_independent_string_overflow_case_24` | 175–180 | 179 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-positive-09` | **正例** | `contract3_independent_string_overflow_case_25` | 182–187 | 186 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-control-01` | 对照 | `contract3_independent_string_overflow_case_26` | 189–194 | 193 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-18` | **正例** | `contract3_independent_string_overflow_case_27` | 196–201 | 200 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-control-20` | 对照 | `contract3_independent_string_overflow_case_28` | 203–208 | 207 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-01` | **正例** | `contract3_independent_string_overflow_case_29` | 210–215 | 214 | `base-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-16` | **正例** | `contract3_independent_string_overflow_case_30` | 217–222 | 221 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-positive-08` | **正例** | `contract3_independent_string_overflow_case_31` | 224–229 | 228 | `hard-2` | `return target[0];` |
| `rt_thread-string_overflow-positive-05` | **正例** | `contract3_independent_string_overflow_case_32` | 231–236 | 235 | `base-4` | `return target[0];` |
| `rt_thread-string_overflow-positive-15` | **正例** | `contract3_independent_string_overflow_case_33` | 238–243 | 242 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-10` | 对照 | `contract3_independent_string_overflow_case_34` | 245–250 | 249 | `hard-4` | `return target[0];` |
| `rt_thread-string_overflow-control-11` | 对照 | `contract3_independent_string_overflow_case_35` | 252–257 | 256 | `hard-0` | `return target[0];` |
| `rt_thread-string_overflow-control-14` | 对照 | `contract3_independent_string_overflow_case_36` | 259–264 | 263 | `hard-3` | `return target[0];` |
| `rt_thread-string_overflow-control-12` | 对照 | `contract3_independent_string_overflow_case_37` | 266–271 | 270 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-control-07` | 对照 | `contract3_independent_string_overflow_case_38` | 273–278 | 277 | `hard-1` | `return target[0];` |
| `rt_thread-string_overflow-positive-03` | **正例** | `contract3_independent_string_overflow_case_39` | 280–285 | 284 | `base-2` | `return target[0];` |
| `rt_thread-string_overflow-control-18` | 对照 | `contract3_independent_string_overflow_case_40` | 287–293 | 292 | `hard-2` | `return target[0];` |

</details>

<details><summary>RT-Thread · 空指针调用 · <code>contract3_independent_hard_web_null_pointer.c</code>（307 行，正例 20 / 对照 20）</summary>

| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |
|---|---|---|---|---:|---|---|
| `rt_thread-null_pointer-control-04` | 对照 | `contract3_independent_null_pointer_case_01` | 12–17 | 16 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-09` | **正例** | `contract3_independent_null_pointer_case_02` | 19–24 | 23 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-positive-03` | **正例** | `contract3_independent_null_pointer_case_03` | 26–30 | 29 | `base-2` | `return contract3_read(0);` |
| `rt_thread-null_pointer-control-20` | 对照 | `contract3_independent_null_pointer_case_04` | 32–38 | 37 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-02` | **正例** | `contract3_independent_null_pointer_case_05` | 40–45 | 44 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-01` | **正例** | `contract3_independent_null_pointer_case_06` | 47–51 | 50 | `base-0` | `return *pointer;` |
| `rt_thread-null_pointer-control-13` | 对照 | `contract3_independent_null_pointer_case_07` | 53–59 | 58 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-control-05` | 对照 | `contract3_independent_null_pointer_case_08` | 61–67 | 66 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-03` | 对照 | `contract3_independent_null_pointer_case_09` | 69–73 | 72 | `base-2` | `return contract3_read(&value);` |
| `rt_thread-null_pointer-positive-18` | **正例** | `contract3_independent_null_pointer_case_10` | 75–81 | 80 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-control-17` | 对照 | `contract3_independent_null_pointer_case_11` | 83–88 | 87 | `hard-1` | `return pointer ? *pointer : 0;` |
| `rt_thread-null_pointer-control-18` | 对照 | `contract3_independent_null_pointer_case_12` | 90–96 | 95 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-control-15` | 对照 | `contract3_independent_null_pointer_case_13` | 98–104 | 103 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-01` | 对照 | `contract3_independent_null_pointer_case_14` | 106–110 | 109 | `base-0` | `return value;` |
| `rt_thread-null_pointer-control-19` | 对照 | `contract3_independent_null_pointer_case_15` | 112–117 | 116 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-positive-08` | **正例** | `contract3_independent_null_pointer_case_16` | 119–125 | 124 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-control-06` | 对照 | `contract3_independent_null_pointer_case_17` | 127–133 | 132 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-control-12` | 对照 | `contract3_independent_null_pointer_case_18` | 135–140 | 139 | `hard-1` | `return pointer ? *pointer : 0;` |
| `rt_thread-null_pointer-positive-07` | **正例** | `contract3_independent_null_pointer_case_19` | 142–147 | 146 | `hard-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-04` | **正例** | `contract3_independent_null_pointer_case_20` | 149–154 | 153 | `base-3` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-19` | **正例** | `contract3_independent_null_pointer_case_21` | 156–161 | 160 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-positive-14` | **正例** | `contract3_independent_null_pointer_case_22` | 163–168 | 167 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-positive-17` | **正例** | `contract3_independent_null_pointer_case_23` | 170–175 | 174 | `hard-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-05` | **正例** | `contract3_independent_null_pointer_case_24` | 177–183 | 182 | `base-4` | `return *pointer;` |
| `rt_thread-null_pointer-control-10` | 对照 | `contract3_independent_null_pointer_case_25` | 185–191 | 190 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-02` | 对照 | `contract3_independent_null_pointer_case_26` | 193–198 | 197 | `base-1` | `return *pointer;` |
| `rt_thread-null_pointer-positive-10` | **正例** | `contract3_independent_null_pointer_case_27` | 200–206 | 205 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-control-14` | 对照 | `contract3_independent_null_pointer_case_28` | 208–213 | 212 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-control-09` | 对照 | `contract3_independent_null_pointer_case_29` | 215–220 | 219 | `hard-3` | `return *pointer;` |
| `rt_thread-null_pointer-positive-13` | **正例** | `contract3_independent_null_pointer_case_30` | 222–228 | 227 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-positive-12` | **正例** | `contract3_independent_null_pointer_case_31` | 230–235 | 234 | `hard-1` | `return *pointer;` |
| `rt_thread-null_pointer-control-11` | 对照 | `contract3_independent_null_pointer_case_32` | 237–243 | 242 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-positive-16` | **正例** | `contract3_independent_null_pointer_case_33` | 245–251 | 250 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-control-08` | 对照 | `contract3_independent_null_pointer_case_34` | 253–259 | 258 | `hard-2` | `return *pointer;` |
| `rt_thread-null_pointer-control-16` | 对照 | `contract3_independent_null_pointer_case_35` | 261–267 | 266 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-control-07` | 对照 | `contract3_independent_null_pointer_case_36` | 269–274 | 273 | `hard-1` | `return pointer ? *pointer : 0;` |
| `rt_thread-null_pointer-positive-20` | **正例** | `contract3_independent_null_pointer_case_37` | 276–282 | 281 | `hard-4` | `return *box.pointer;` |
| `rt_thread-null_pointer-positive-11` | **正例** | `contract3_independent_null_pointer_case_38` | 284–290 | 289 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-positive-06` | **正例** | `contract3_independent_null_pointer_case_39` | 292–298 | 297 | `hard-0` | `return *pointer + value;` |
| `rt_thread-null_pointer-positive-15` | **正例** | `contract3_independent_null_pointer_case_40` | 300–306 | 305 | `hard-4` | `return *box.pointer;` |

</details>

## 5. GT JSON 字段

| 字段 | 含义 |
|---|---|
| `projects[].detection_files[]` | 要检测的上游源码文件：`path`（相对工程根目录）、`lines` |
| `projects[].line_totals` | `upstream_detection_files` 检测文件合计行数；`fixtures` 测试文件行数；`all_detection_files` 两者之和 |
| `projects[].fixtures[]` | 注入的测试文件：`path`、`category`、`lines`、`sha256` |
| `projects[].cases[]` | 逐函数标签：`case_id`、`category`、`expected`（`defect`/`control`）、`function`、`file`、`line_start`、`line_end`、`defect_line`、`defect_code`；加难版另有 `variant` |

