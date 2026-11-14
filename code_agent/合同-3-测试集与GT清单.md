# 合同三：工程教学 v5 测试集与 GT 清单

版本：`engineering-v5-20261114`。三工程、四套独立源码快照；缺陷修改原有函数中的操作，不新增独立测试函数。GT、教师配方均在 project/ 外。

每工程每套每类 7 个缺陷、7 个安全对照，共 42 个操作。四套是有关联的教师校准题集，不能称为独立泛化测试。实际统计见 [总报告](合同-3-result.md)。

| 题集 | 工程 | 完整 C 文件数 | 送检行数 | 数组 正/负 | 字符串 正/负 | 空指针 正/负 |
|---|---|---:|---:|---:|---:|---:|
| balanced | freertos_kernel | 7 | 16642 | 7/7 | 7/7 | 7/7 |
| balanced | zephyr | 124 | 31870 | 7/7 | 7/7 | 7/7 |
| balanced | rt_thread | 77 | 66133 | 7/7 | 7/7 | 7/7 |
| independent | freertos_kernel | 7 | 16637 | 7/7 | 7/7 | 7/7 |
| independent | zephyr | 124 | 31871 | 7/7 | 7/7 | 7/7 |
| independent | rt_thread | 77 | 66133 | 7/7 | 7/7 | 7/7 |
| balanced-hard | freertos_kernel | 7 | 16643 | 7/7 | 7/7 | 7/7 |
| balanced-hard | zephyr | 124 | 31871 | 7/7 | 7/7 | 7/7 |
| balanced-hard | rt_thread | 77 | 66133 | 7/7 | 7/7 | 7/7 |
| independent-hard | freertos_kernel | 7 | 16636 | 7/7 | 7/7 | 7/7 |
| independent-hard | zephyr | 124 | 31870 | 7/7 | 7/7 | 7/7 |
| independent-hard | rt_thread | 77 | 66133 | 7/7 | 7/7 | 7/7 |

行数只累计送检的完整 C 文件；头文件不计入此表。源码审计覆盖不等于所有板级配置编译覆盖。安全标签只针对当前操作的边界/非空前提，不保证函数其他操作或整个工程没有其他问题。

旧版 v3 小样例和初始 v4 工程题及原始结果保留在各自版本目录，不能混用源码、GT 或成绩。

## balanced / FreeRTOS Kernel

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/balanced/freertos_kernel/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | defect | `vCoRoutineSchedule` | [`croutine.c:313`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| array_oob-02 | array_oob | defect | `prvInitialiseCoRoutineLists` | [`croutine.c:342`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| array_oob-03 | array_oob | defect | `vQueueAddToRegistry` | [`queue.c:3026`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/queue.c) |
| array_oob-04 | array_oob | defect | `pcQueueGetName` | [`queue.c:3074`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/queue.c) |
| array_oob-05 | array_oob | defect | `vQueueUnregisterQueue` | [`queue.c:3107`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/queue.c) |
| array_oob-06 | array_oob | defect | `prvCheckForRunStateChange` | [`tasks.c:815`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-07 | array_oob | defect | `prvYieldForTask` | [`tasks.c:901`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-08 | array_oob | control | `prvSelectHighestPriorityTask` | [`tasks.c:1010`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-09 | array_oob | control | `vTaskDelete` | [`tasks.c:2267`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-10 | array_oob | control | `eTaskGetState` | [`tasks.c:2551`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-11 | array_oob | control | `vTaskPrioritySet` | [`tasks.c:2907`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-12 | array_oob | control | `vTaskSuspend` | [`tasks.c:3164`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-13 | array_oob | control | `prvTaskIsTaskSuspended` | [`tasks.c:3317`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| array_oob-14 | array_oob | control | `xTaskResumeFromISR` | [`tasks.c:3465`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-01 | string_overflow | defect | `prvInitialiseNewTask` | [`tasks.c:1867`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-02 | string_overflow | defect | `prvWriteNameToBuffer` | [`tasks.c:7198`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-03 | string_overflow | defect | `prvCreateIdleTasks` | [`tasks.c:3567`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-04 | string_overflow | defect | `prvWriteNameToBuffer` | [`tasks.c:7192`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-05 | string_overflow | defect | `vTaskListTasks` | [`tasks.c:7325`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-06 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7472`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-07 | string_overflow | defect | `prvCreateIdleTasks` | [`tasks.c:3572`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-08 | string_overflow | control | `prvCreateIdleTasks` | [`tasks.c:3521`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-09 | string_overflow | control | `prvInitialiseNewTask` | [`tasks.c:1884`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-10 | string_overflow | control | `prvWriteNameToBuffer` | [`tasks.c:7202`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-11 | string_overflow | control | `vTaskListTasks` | [`tasks.c:7330`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-12 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7481`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-13 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7498`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| string_overflow-14 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7510`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/tasks.c) |
| null_pointer-01 | null_pointer | defect | `xCoRoutineCreate` | [`croutine.c:137`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| null_pointer-02 | null_pointer | defect | `vCoRoutineAddToDelayedList` | [`croutine.c:186`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| null_pointer-03 | null_pointer | defect | `xEventGroupSync` | [`event_groups.c:207`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-04 | null_pointer | defect | `prvCheckDelayedList` | [`croutine.c:266`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| null_pointer-05 | null_pointer | defect | `xCoRoutineRemoveFromEventList` | [`croutine.c:367`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| null_pointer-06 | null_pointer | defect | `xEventGroupCreateStatic` | [`event_groups.c:113`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-07 | null_pointer | defect | `xEventGroupCreate` | [`event_groups.c:158`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-08 | null_pointer | control | `prvCheckPendingReadyList` | [`croutine.c:228`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/croutine.c) |
| null_pointer-09 | null_pointer | control | `xEventGroupWaitBits` | [`event_groups.c:338`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-10 | null_pointer | control | `xEventGroupClearBits` | [`event_groups.c:491`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-11 | null_pointer | control | `xEventGroupGetBitsFromISR` | [`event_groups.c:537`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-12 | null_pointer | control | `xEventGroupSetBits` | [`event_groups.c:565`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-13 | null_pointer | control | `vEventGroupDelete` | [`event_groups.c:656`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |
| null_pointer-14 | null_pointer | control | `xEventGroupGetStaticBuffer` | [`event_groups.c:712`](artifacts/contract3/engineering-v5/balanced/freertos_kernel/project/event_groups.c) |

## balanced / Zephyr RTOS

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/balanced/zephyr/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | defect | `init_mbox_module` | [`kernel/mailbox.c:74`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/mailbox.c) |
| array_oob-02 | array_oob | defect | `z_paging_histogram_inc` | [`kernel/paging/statistics.c:179`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/paging/statistics.c) |
| array_oob-03 | array_oob | defect | `slice_timeout` | [`kernel/timeslicing.c:58`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/timeslicing.c) |
| array_oob-04 | array_oob | defect | `z_reset_time_slice` | [`kernel/timeslicing.c:73`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/timeslicing.c) |
| array_oob-05 | array_oob | defect | `z_time_slice` | [`kernel/timeslicing.c:117`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/timeslicing.c) |
| array_oob-06 | array_oob | defect | `thread_idx_alloc` | [`kernel/userspace.c:275`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/userspace.c) |
| array_oob-07 | array_oob | defect | `thread_idx_free` | [`kernel/userspace.c:314`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/userspace.c) |
| array_oob-08 | array_oob | control | `pthread_barrier_pool_init` | [`lib/posix/options/barrier.c:204`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/barrier.c) |
| array_oob-09 | array_oob | control | `posix_fs_alloc_obj` | [`lib/posix/options/fs.c:45`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/fs.c) |
| array_oob-10 | array_oob | control | `acquire_mutex` | [`lib/posix/options/mutex.c:126`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-11 | array_oob | control | `pthread_mutex_init` | [`lib/posix/options/mutex.c:229`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-12 | array_oob | control | `pthread_setname_np` | [`lib/posix/options/pthread.c:1366`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-13 | array_oob | control | `pthread_getname_np` | [`lib/posix/options/pthread.c:1395`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-14 | array_oob | control | `pthread_spin_lock` | [`lib/posix/options/spinlock.c:126`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/spinlock.c) |
| string_overflow-01 | string_overflow | defect | `z_impl_k_thread_name_set` | [`kernel/thread.c:146`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/thread.c) |
| string_overflow-02 | string_overflow | defect | `readdir` | [`lib/posix/options/fs.c:313`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/fs.c) |
| string_overflow-03 | string_overflow | defect | `z_impl_k_thread_name_set` | [`kernel/thread.c:145`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/thread.c) |
| string_overflow-04 | string_overflow | defect | `z_impl_k_thread_name_copy` | [`kernel/thread.c:202`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/thread.c) |
| string_overflow-05 | string_overflow | defect | `z_setup_new_thread` | [`kernel/thread.c:607`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/thread.c) |
| string_overflow-06 | string_overflow | defect | `strerror_r` | [`lib/libc/minimal/source/string/strerror.c:56`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/libc/minimal/source/string/strerror.c) |
| string_overflow-07 | string_overflow | defect | `mq_open` | [`lib/posix/options/mqueue.c:144`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/mqueue.c) |
| string_overflow-08 | string_overflow | control | `z_setup_new_thread` | [`kernel/thread.c:609`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/thread.c) |
| string_overflow-09 | string_overflow | control | `uname` | [`lib/posix/options/uname.c:50`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-10 | string_overflow | control | `inet_ntoa` | [`lib/posix/options/net.c:82`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/net.c) |
| string_overflow-11 | string_overflow | control | `sem_open` | [`lib/posix/options/semaphore.c:289`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/semaphore.c) |
| string_overflow-12 | string_overflow | control | `strsignal` | [`lib/posix/options/signal.c:89`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/signal.c) |
| string_overflow-13 | string_overflow | control | `uname` | [`lib/posix/options/uname.c:49`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-14 | string_overflow | control | `utf8_lcpy` | [`lib/utils/utf8.c:72`](artifacts/contract3/engineering-v5/balanced/zephyr/project/lib/utils/utf8.c) |
| null_pointer-01 | null_pointer | defect | `z_impl_k_condvar_init` | [`kernel/condvar.c:23`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/condvar.c) |
| null_pointer-02 | null_pointer | defect | `z_impl_k_condvar_signal` | [`kernel/condvar.c:50`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/condvar.c) |
| null_pointer-03 | null_pointer | defect | `z_impl_k_condvar_broadcast` | [`kernel/condvar.c:87`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/condvar.c) |
| null_pointer-04 | null_pointer | defect | `z_impl_k_condvar_wait` | [`kernel/condvar.c:120`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/condvar.c) |
| null_pointer-05 | null_pointer | defect | `z_impl_device_get_binding` | [`kernel/device.c:44`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/device.c) |
| null_pointer-06 | null_pointer | defect | `z_impl_device_get_by_dt_nodelabel` | [`kernel/device.c:105`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/device.c) |
| null_pointer-07 | null_pointer | defect | `z_impl_device_is_ready` | [`kernel/device.c:152`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/device.c) |
| null_pointer-08 | null_pointer | control | `dyn_cb` | [`kernel/dynamic.c:115`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/dynamic.c) |
| null_pointer-09 | null_pointer | control | `z_impl_z_errno` | [`kernel/errno.c:39`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/errno.c) |
| null_pointer-10 | null_pointer | control | `z_impl_k_event_init` | [`kernel/events.c:54`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/events.c) |
| null_pointer-11 | null_pointer | control | `event_walk_op` | [`kernel/events.c:106`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/events.c) |
| null_pointer-12 | null_pointer | control | `k_event_post_internal` | [`kernel/events.c:140`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/events.c) |
| null_pointer-13 | null_pointer | control | `k_event_wait_internal` | [`kernel/events.c:261`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/events.c) |
| null_pointer-14 | null_pointer | control | `z_fatal_error` | [`kernel/fatal.c:99`](artifacts/contract3/engineering-v5/balanced/zephyr/project/kernel/fatal.c) |

## balanced / RT-Thread

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/balanced/rt_thread/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | defect | `rt_thread_idle_sethook` | [`src/idle.c:80`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/idle.c) |
| array_oob-02 | array_oob | defect | `rt_thread_idle_delhook` | [`src/idle.c:111`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/idle.c) |
| array_oob-03 | array_oob | defect | `idle_thread_entry` | [`src/idle.c:147`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/idle.c) |
| array_oob-04 | array_oob | defect | `get_components` | [`src/klibc/rt_vsnprintf_std.c:553`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-05 | array_oob | defect | `get_normalized_components` | [`src/klibc/rt_vsnprintf_std.c:645`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-06 | array_oob | defect | `print_exponential_number` | [`src/klibc/rt_vsnprintf_std.c:850`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-07 | array_oob | defect | `memheaptrace` | [`src/memheap.c:984`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/memheap.c) |
| array_oob-08 | array_oob | control | `rt_object_get_information` | [`src/object.c:255`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/object.c) |
| array_oob-09 | array_oob | control | `_get_global_highest_ready_prio` | [`src/scheduler_mp.c:188`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/scheduler_mp.c) |
| array_oob-10 | array_oob | control | `_get_local_highest_ready_prio` | [`src/scheduler_mp.c:205`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/scheduler_mp.c) |
| array_oob-11 | array_oob | control | `_scheduler_get_highest_priority_thread` | [`src/scheduler_mp.c:246`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/scheduler_mp.c) |
| array_oob-12 | array_oob | control | `_sched_insert_thread_locked` | [`src/scheduler_mp.c:303`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/scheduler_mp.c) |
| array_oob-13 | array_oob | control | `_sched_remove_thread_locked` | [`src/scheduler_mp.c:368`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/scheduler_mp.c) |
| array_oob-14 | array_oob | control | `_scheduler_get_highest_priority_thread` | [`src/scheduler_up.c:98`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/scheduler_up.c) |
| string_overflow-01 | string_overflow | defect | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:499`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-02 | string_overflow | defect | `finsh_get_prompt` | [`components/finsh/shell.c:102`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/components/finsh/shell.c) |
| string_overflow-03 | string_overflow | defect | `rt_thread_idle_init` | [`src/idle.c:185`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/idle.c) |
| string_overflow-04 | string_overflow | defect | `rt_snprintf` | [`src/klibc/kstdio.c:34`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-05 | string_overflow | defect | `rt_vsprintf` | [`src/klibc/kstdio.c:54`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-06 | string_overflow | defect | `rt_vsnprintf` | [`src/klibc/kstdio.c:83`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-07 | string_overflow | defect | `rt_strncpy` | [`src/klibc/kstring.c:364`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-08 | string_overflow | control | `print_floating_point` | [`src/klibc/rt_vsnprintf_std.c:975`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-09 | string_overflow | control | `finsh_wait_auth` | [`components/finsh/shell.c:357`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/components/finsh/shell.c) |
| string_overflow-10 | string_overflow | control | `rt_strcpy` | [`src/klibc/kstring.c:405`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-11 | string_overflow | control | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:514`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-12 | string_overflow | control | `rt_kprintf` | [`src/kservice.c:373`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/kservice.c) |
| string_overflow-13 | string_overflow | control | `rt_object_init` | [`src/object.c:393`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/object.c) |
| string_overflow-14 | string_overflow | control | `rt_object_allocate` | [`src/object.c:494`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/object.c) |
| null_pointer-01 | null_pointer | defect | `_update_process_times` | [`src/clock.c:110`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/clock.c) |
| null_pointer-02 | null_pointer | defect | `rt_components_board_init` | [`src/components.c:93`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/components.c) |
| null_pointer-03 | null_pointer | defect | `rt_components_init` | [`src/components.c:119`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/components.c) |
| null_pointer-04 | null_pointer | defect | `rt_spin_lock_init` | [`src/cpu_mp.c:39`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-05 | null_pointer | defect | `rt_spin_lock` | [`src/cpu_mp.c:54`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-06 | null_pointer | defect | `rt_spin_unlock` | [`src/cpu_mp.c:68`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-07 | null_pointer | defect | `rt_spin_lock_irqsave` | [`src/cpu_mp.c:89`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-08 | null_pointer | control | `rt_spin_unlock_irqrestore` | [`src/cpu_mp.c:107`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-09 | null_pointer | control | `rt_cpus_lock` | [`src/cpu_mp.c:147`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-10 | null_pointer | control | `rt_cpus_unlock` | [`src/cpu_mp.c:180`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/cpu_mp.c) |
| null_pointer-11 | null_pointer | control | `rt_thread_defunct_dequeue` | [`src/defunct.c:56`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/defunct.c) |
| null_pointer-12 | null_pointer | control | `rt_defunct_execute` | [`src/defunct.c:88`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/defunct.c) |
| null_pointer-13 | null_pointer | control | `_ipc_object_init` | [`src/ipc.c:85`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/ipc.c) |
| null_pointer-14 | null_pointer | control | `rt_susp_list_dequeue` | [`src/ipc.c:117`](artifacts/contract3/engineering-v5/balanced/rt_thread/project/src/ipc.c) |

## independent / FreeRTOS Kernel

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/independent/freertos_kernel/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | control | `vCoRoutineSchedule` | [`croutine.c:313`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| array_oob-02 | array_oob | control | `prvInitialiseCoRoutineLists` | [`croutine.c:342`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| array_oob-03 | array_oob | control | `vQueueAddToRegistry` | [`queue.c:3026`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/queue.c) |
| array_oob-04 | array_oob | control | `pcQueueGetName` | [`queue.c:3074`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/queue.c) |
| array_oob-05 | array_oob | control | `vQueueUnregisterQueue` | [`queue.c:3107`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/queue.c) |
| array_oob-06 | array_oob | control | `prvCheckForRunStateChange` | [`tasks.c:815`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-07 | array_oob | control | `prvYieldForTask` | [`tasks.c:901`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-08 | array_oob | defect | `prvSelectHighestPriorityTask` | [`tasks.c:1010`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-09 | array_oob | defect | `vTaskDelete` | [`tasks.c:2267`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-10 | array_oob | defect | `eTaskGetState` | [`tasks.c:2551`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-11 | array_oob | defect | `vTaskPrioritySet` | [`tasks.c:2907`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-12 | array_oob | defect | `vTaskSuspend` | [`tasks.c:3164`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-13 | array_oob | defect | `prvTaskIsTaskSuspended` | [`tasks.c:3317`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| array_oob-14 | array_oob | defect | `xTaskResumeFromISR` | [`tasks.c:3465`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-01 | string_overflow | control | `prvInitialiseNewTask` | [`tasks.c:1867`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-02 | string_overflow | control | `prvWriteNameToBuffer` | [`tasks.c:7198`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-03 | string_overflow | control | `prvCreateIdleTasks` | [`tasks.c:3567`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-04 | string_overflow | control | `prvWriteNameToBuffer` | [`tasks.c:7192`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-05 | string_overflow | control | `vTaskListTasks` | [`tasks.c:7325`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-06 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7473`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-07 | string_overflow | control | `prvCreateIdleTasks` | [`tasks.c:3572`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-08 | string_overflow | defect | `prvCreateIdleTasks` | [`tasks.c:3521`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-09 | string_overflow | defect | `prvInitialiseNewTask` | [`tasks.c:1884`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-10 | string_overflow | defect | `prvWriteNameToBuffer` | [`tasks.c:7202`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-11 | string_overflow | defect | `vTaskListTasks` | [`tasks.c:7337`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-12 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7486`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-13 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7499`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| string_overflow-14 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7508`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/tasks.c) |
| null_pointer-01 | null_pointer | control | `xCoRoutineCreate` | [`croutine.c:137`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| null_pointer-02 | null_pointer | control | `vCoRoutineAddToDelayedList` | [`croutine.c:186`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| null_pointer-03 | null_pointer | control | `xEventGroupSync` | [`event_groups.c:207`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-04 | null_pointer | control | `prvCheckDelayedList` | [`croutine.c:266`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| null_pointer-05 | null_pointer | control | `xCoRoutineRemoveFromEventList` | [`croutine.c:367`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| null_pointer-06 | null_pointer | control | `xEventGroupCreateStatic` | [`event_groups.c:113`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-07 | null_pointer | control | `xEventGroupCreate` | [`event_groups.c:158`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-08 | null_pointer | defect | `prvCheckPendingReadyList` | [`croutine.c:228`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/croutine.c) |
| null_pointer-09 | null_pointer | defect | `xEventGroupWaitBits` | [`event_groups.c:338`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-10 | null_pointer | defect | `xEventGroupClearBits` | [`event_groups.c:491`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-11 | null_pointer | defect | `xEventGroupGetBitsFromISR` | [`event_groups.c:537`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-12 | null_pointer | defect | `xEventGroupSetBits` | [`event_groups.c:565`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-13 | null_pointer | defect | `vEventGroupDelete` | [`event_groups.c:656`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |
| null_pointer-14 | null_pointer | defect | `xEventGroupGetStaticBuffer` | [`event_groups.c:712`](artifacts/contract3/engineering-v5/independent/freertos_kernel/project/event_groups.c) |

## independent / Zephyr RTOS

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/independent/zephyr/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | control | `init_mbox_module` | [`kernel/mailbox.c:74`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/mailbox.c) |
| array_oob-02 | array_oob | control | `z_paging_histogram_inc` | [`kernel/paging/statistics.c:179`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/paging/statistics.c) |
| array_oob-03 | array_oob | control | `slice_timeout` | [`kernel/timeslicing.c:58`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/timeslicing.c) |
| array_oob-04 | array_oob | control | `z_reset_time_slice` | [`kernel/timeslicing.c:73`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/timeslicing.c) |
| array_oob-05 | array_oob | control | `z_time_slice` | [`kernel/timeslicing.c:117`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/timeslicing.c) |
| array_oob-06 | array_oob | control | `thread_idx_alloc` | [`kernel/userspace.c:275`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/userspace.c) |
| array_oob-07 | array_oob | control | `thread_idx_free` | [`kernel/userspace.c:314`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/userspace.c) |
| array_oob-08 | array_oob | defect | `pthread_barrier_pool_init` | [`lib/posix/options/barrier.c:204`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/barrier.c) |
| array_oob-09 | array_oob | defect | `posix_fs_alloc_obj` | [`lib/posix/options/fs.c:45`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/fs.c) |
| array_oob-10 | array_oob | defect | `acquire_mutex` | [`lib/posix/options/mutex.c:126`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-11 | array_oob | defect | `pthread_mutex_init` | [`lib/posix/options/mutex.c:229`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-12 | array_oob | defect | `pthread_setname_np` | [`lib/posix/options/pthread.c:1366`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-13 | array_oob | defect | `pthread_getname_np` | [`lib/posix/options/pthread.c:1395`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-14 | array_oob | defect | `pthread_spin_lock` | [`lib/posix/options/spinlock.c:126`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/spinlock.c) |
| string_overflow-01 | string_overflow | control | `z_impl_k_thread_name_set` | [`kernel/thread.c:146`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/thread.c) |
| string_overflow-02 | string_overflow | control | `readdir` | [`lib/posix/options/fs.c:313`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/fs.c) |
| string_overflow-03 | string_overflow | control | `z_impl_k_thread_name_set` | [`kernel/thread.c:145`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/thread.c) |
| string_overflow-04 | string_overflow | control | `z_impl_k_thread_name_copy` | [`kernel/thread.c:202`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/thread.c) |
| string_overflow-05 | string_overflow | control | `z_setup_new_thread` | [`kernel/thread.c:607`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/thread.c) |
| string_overflow-06 | string_overflow | control | `strerror_r` | [`lib/libc/minimal/source/string/strerror.c:56`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/libc/minimal/source/string/strerror.c) |
| string_overflow-07 | string_overflow | control | `mq_open` | [`lib/posix/options/mqueue.c:144`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/mqueue.c) |
| string_overflow-08 | string_overflow | defect | `z_setup_new_thread` | [`kernel/thread.c:610`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/thread.c) |
| string_overflow-09 | string_overflow | defect | `uname` | [`lib/posix/options/uname.c:50`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-10 | string_overflow | defect | `inet_ntoa` | [`lib/posix/options/net.c:82`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/net.c) |
| string_overflow-11 | string_overflow | defect | `sem_open` | [`lib/posix/options/semaphore.c:289`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/semaphore.c) |
| string_overflow-12 | string_overflow | defect | `strsignal` | [`lib/posix/options/signal.c:89`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/signal.c) |
| string_overflow-13 | string_overflow | defect | `uname` | [`lib/posix/options/uname.c:49`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-14 | string_overflow | defect | `utf8_lcpy` | [`lib/utils/utf8.c:72`](artifacts/contract3/engineering-v5/independent/zephyr/project/lib/utils/utf8.c) |
| null_pointer-01 | null_pointer | control | `z_impl_k_condvar_init` | [`kernel/condvar.c:23`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/condvar.c) |
| null_pointer-02 | null_pointer | control | `z_impl_k_condvar_signal` | [`kernel/condvar.c:50`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/condvar.c) |
| null_pointer-03 | null_pointer | control | `z_impl_k_condvar_broadcast` | [`kernel/condvar.c:87`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/condvar.c) |
| null_pointer-04 | null_pointer | control | `z_impl_k_condvar_wait` | [`kernel/condvar.c:120`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/condvar.c) |
| null_pointer-05 | null_pointer | control | `z_impl_device_get_binding` | [`kernel/device.c:44`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/device.c) |
| null_pointer-06 | null_pointer | control | `z_impl_device_get_by_dt_nodelabel` | [`kernel/device.c:105`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/device.c) |
| null_pointer-07 | null_pointer | control | `z_impl_device_is_ready` | [`kernel/device.c:152`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/device.c) |
| null_pointer-08 | null_pointer | defect | `dyn_cb` | [`kernel/dynamic.c:115`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/dynamic.c) |
| null_pointer-09 | null_pointer | defect | `z_impl_z_errno` | [`kernel/errno.c:39`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/errno.c) |
| null_pointer-10 | null_pointer | defect | `z_impl_k_event_init` | [`kernel/events.c:54`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/events.c) |
| null_pointer-11 | null_pointer | defect | `event_walk_op` | [`kernel/events.c:106`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/events.c) |
| null_pointer-12 | null_pointer | defect | `k_event_post_internal` | [`kernel/events.c:140`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/events.c) |
| null_pointer-13 | null_pointer | defect | `k_event_wait_internal` | [`kernel/events.c:261`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/events.c) |
| null_pointer-14 | null_pointer | defect | `z_fatal_error` | [`kernel/fatal.c:99`](artifacts/contract3/engineering-v5/independent/zephyr/project/kernel/fatal.c) |

## independent / RT-Thread

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/independent/rt_thread/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | control | `rt_thread_idle_sethook` | [`src/idle.c:80`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/idle.c) |
| array_oob-02 | array_oob | control | `rt_thread_idle_delhook` | [`src/idle.c:111`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/idle.c) |
| array_oob-03 | array_oob | control | `idle_thread_entry` | [`src/idle.c:147`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/idle.c) |
| array_oob-04 | array_oob | control | `get_components` | [`src/klibc/rt_vsnprintf_std.c:553`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-05 | array_oob | control | `get_normalized_components` | [`src/klibc/rt_vsnprintf_std.c:645`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-06 | array_oob | control | `print_exponential_number` | [`src/klibc/rt_vsnprintf_std.c:850`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-07 | array_oob | control | `memheaptrace` | [`src/memheap.c:984`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/memheap.c) |
| array_oob-08 | array_oob | defect | `rt_object_get_information` | [`src/object.c:255`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/object.c) |
| array_oob-09 | array_oob | defect | `_get_global_highest_ready_prio` | [`src/scheduler_mp.c:188`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/scheduler_mp.c) |
| array_oob-10 | array_oob | defect | `_get_local_highest_ready_prio` | [`src/scheduler_mp.c:205`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/scheduler_mp.c) |
| array_oob-11 | array_oob | defect | `_scheduler_get_highest_priority_thread` | [`src/scheduler_mp.c:246`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/scheduler_mp.c) |
| array_oob-12 | array_oob | defect | `_sched_insert_thread_locked` | [`src/scheduler_mp.c:303`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/scheduler_mp.c) |
| array_oob-13 | array_oob | defect | `_sched_remove_thread_locked` | [`src/scheduler_mp.c:368`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/scheduler_mp.c) |
| array_oob-14 | array_oob | defect | `_scheduler_get_highest_priority_thread` | [`src/scheduler_up.c:98`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/scheduler_up.c) |
| string_overflow-01 | string_overflow | control | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:499`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-02 | string_overflow | control | `finsh_get_prompt` | [`components/finsh/shell.c:102`](artifacts/contract3/engineering-v5/independent/rt_thread/project/components/finsh/shell.c) |
| string_overflow-03 | string_overflow | control | `rt_thread_idle_init` | [`src/idle.c:185`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/idle.c) |
| string_overflow-04 | string_overflow | control | `rt_snprintf` | [`src/klibc/kstdio.c:34`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-05 | string_overflow | control | `rt_vsprintf` | [`src/klibc/kstdio.c:54`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-06 | string_overflow | control | `rt_vsnprintf` | [`src/klibc/kstdio.c:83`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-07 | string_overflow | control | `rt_strncpy` | [`src/klibc/kstring.c:364`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-08 | string_overflow | defect | `print_floating_point` | [`src/klibc/rt_vsnprintf_std.c:975`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-09 | string_overflow | defect | `finsh_wait_auth` | [`components/finsh/shell.c:357`](artifacts/contract3/engineering-v5/independent/rt_thread/project/components/finsh/shell.c) |
| string_overflow-10 | string_overflow | defect | `rt_strcpy` | [`src/klibc/kstring.c:405`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-11 | string_overflow | defect | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:514`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-12 | string_overflow | defect | `rt_kprintf` | [`src/kservice.c:373`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/kservice.c) |
| string_overflow-13 | string_overflow | defect | `rt_object_init` | [`src/object.c:393`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/object.c) |
| string_overflow-14 | string_overflow | defect | `rt_object_allocate` | [`src/object.c:494`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/object.c) |
| null_pointer-01 | null_pointer | control | `_update_process_times` | [`src/clock.c:110`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/clock.c) |
| null_pointer-02 | null_pointer | control | `rt_components_board_init` | [`src/components.c:93`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/components.c) |
| null_pointer-03 | null_pointer | control | `rt_components_init` | [`src/components.c:119`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/components.c) |
| null_pointer-04 | null_pointer | control | `rt_spin_lock_init` | [`src/cpu_mp.c:39`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-05 | null_pointer | control | `rt_spin_lock` | [`src/cpu_mp.c:54`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-06 | null_pointer | control | `rt_spin_unlock` | [`src/cpu_mp.c:68`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-07 | null_pointer | control | `rt_spin_lock_irqsave` | [`src/cpu_mp.c:89`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-08 | null_pointer | defect | `rt_spin_unlock_irqrestore` | [`src/cpu_mp.c:107`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-09 | null_pointer | defect | `rt_cpus_lock` | [`src/cpu_mp.c:147`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-10 | null_pointer | defect | `rt_cpus_unlock` | [`src/cpu_mp.c:180`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/cpu_mp.c) |
| null_pointer-11 | null_pointer | defect | `rt_thread_defunct_dequeue` | [`src/defunct.c:56`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/defunct.c) |
| null_pointer-12 | null_pointer | defect | `rt_defunct_execute` | [`src/defunct.c:88`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/defunct.c) |
| null_pointer-13 | null_pointer | defect | `_ipc_object_init` | [`src/ipc.c:85`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/ipc.c) |
| null_pointer-14 | null_pointer | defect | `rt_susp_list_dequeue` | [`src/ipc.c:117`](artifacts/contract3/engineering-v5/independent/rt_thread/project/src/ipc.c) |

## balanced-hard / FreeRTOS Kernel

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | defect | `vCoRoutineSchedule` | [`croutine.c:313`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| array_oob-02 | array_oob | defect | `prvInitialiseCoRoutineLists` | [`croutine.c:342`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| array_oob-03 | array_oob | defect | `vQueueAddToRegistry` | [`queue.c:3026`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/queue.c) |
| array_oob-04 | array_oob | defect | `pcQueueGetName` | [`queue.c:3074`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/queue.c) |
| array_oob-05 | array_oob | control | `vQueueUnregisterQueue` | [`queue.c:3107`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/queue.c) |
| array_oob-06 | array_oob | control | `prvCheckForRunStateChange` | [`tasks.c:815`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-07 | array_oob | control | `prvYieldForTask` | [`tasks.c:901`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-08 | array_oob | control | `prvSelectHighestPriorityTask` | [`tasks.c:1010`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-09 | array_oob | control | `vTaskDelete` | [`tasks.c:2267`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-10 | array_oob | control | `eTaskGetState` | [`tasks.c:2551`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-11 | array_oob | control | `vTaskPrioritySet` | [`tasks.c:2907`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-12 | array_oob | defect | `vTaskSuspend` | [`tasks.c:3164`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-13 | array_oob | defect | `prvTaskIsTaskSuspended` | [`tasks.c:3317`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| array_oob-14 | array_oob | defect | `xTaskResumeFromISR` | [`tasks.c:3465`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-01 | string_overflow | defect | `prvInitialiseNewTask` | [`tasks.c:1867`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-02 | string_overflow | defect | `prvWriteNameToBuffer` | [`tasks.c:7198`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-03 | string_overflow | defect | `prvCreateIdleTasks` | [`tasks.c:3567`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-04 | string_overflow | defect | `prvWriteNameToBuffer` | [`tasks.c:7192`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-05 | string_overflow | control | `vTaskListTasks` | [`tasks.c:7325`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-06 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7479`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-07 | string_overflow | control | `prvCreateIdleTasks` | [`tasks.c:3572`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-08 | string_overflow | control | `prvCreateIdleTasks` | [`tasks.c:3521`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-09 | string_overflow | control | `prvInitialiseNewTask` | [`tasks.c:1884`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-10 | string_overflow | control | `prvWriteNameToBuffer` | [`tasks.c:7202`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-11 | string_overflow | control | `vTaskListTasks` | [`tasks.c:7337`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-12 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7492`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-13 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7505`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| string_overflow-14 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7514`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/tasks.c) |
| null_pointer-01 | null_pointer | defect | `xCoRoutineCreate` | [`croutine.c:137`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| null_pointer-02 | null_pointer | defect | `vCoRoutineAddToDelayedList` | [`croutine.c:186`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| null_pointer-03 | null_pointer | defect | `xEventGroupSync` | [`event_groups.c:207`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-04 | null_pointer | defect | `prvCheckDelayedList` | [`croutine.c:266`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| null_pointer-05 | null_pointer | control | `xCoRoutineRemoveFromEventList` | [`croutine.c:367`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| null_pointer-06 | null_pointer | control | `xEventGroupCreateStatic` | [`event_groups.c:113`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-07 | null_pointer | control | `xEventGroupCreate` | [`event_groups.c:158`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-08 | null_pointer | control | `prvCheckPendingReadyList` | [`croutine.c:228`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/croutine.c) |
| null_pointer-09 | null_pointer | control | `xEventGroupWaitBits` | [`event_groups.c:338`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-10 | null_pointer | control | `xEventGroupClearBits` | [`event_groups.c:491`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-11 | null_pointer | control | `xEventGroupGetBitsFromISR` | [`event_groups.c:537`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-12 | null_pointer | defect | `xEventGroupSetBits` | [`event_groups.c:565`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-13 | null_pointer | defect | `vEventGroupDelete` | [`event_groups.c:656`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-14 | null_pointer | defect | `xEventGroupGetStaticBuffer` | [`event_groups.c:712`](artifacts/contract3/engineering-v5/balanced-hard/freertos_kernel/project/event_groups.c) |

## balanced-hard / Zephyr RTOS

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/balanced-hard/zephyr/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | defect | `init_mbox_module` | [`kernel/mailbox.c:74`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/mailbox.c) |
| array_oob-02 | array_oob | defect | `z_paging_histogram_inc` | [`kernel/paging/statistics.c:179`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/paging/statistics.c) |
| array_oob-03 | array_oob | defect | `slice_timeout` | [`kernel/timeslicing.c:58`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/timeslicing.c) |
| array_oob-04 | array_oob | defect | `z_reset_time_slice` | [`kernel/timeslicing.c:73`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/timeslicing.c) |
| array_oob-05 | array_oob | control | `z_time_slice` | [`kernel/timeslicing.c:117`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/timeslicing.c) |
| array_oob-06 | array_oob | control | `thread_idx_alloc` | [`kernel/userspace.c:275`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/userspace.c) |
| array_oob-07 | array_oob | control | `thread_idx_free` | [`kernel/userspace.c:314`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/userspace.c) |
| array_oob-08 | array_oob | control | `pthread_barrier_pool_init` | [`lib/posix/options/barrier.c:204`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/barrier.c) |
| array_oob-09 | array_oob | control | `posix_fs_alloc_obj` | [`lib/posix/options/fs.c:45`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/fs.c) |
| array_oob-10 | array_oob | control | `acquire_mutex` | [`lib/posix/options/mutex.c:126`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-11 | array_oob | control | `pthread_mutex_init` | [`lib/posix/options/mutex.c:229`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-12 | array_oob | defect | `pthread_setname_np` | [`lib/posix/options/pthread.c:1366`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-13 | array_oob | defect | `pthread_getname_np` | [`lib/posix/options/pthread.c:1395`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-14 | array_oob | defect | `pthread_spin_lock` | [`lib/posix/options/spinlock.c:126`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/spinlock.c) |
| string_overflow-01 | string_overflow | defect | `z_impl_k_thread_name_set` | [`kernel/thread.c:146`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/thread.c) |
| string_overflow-02 | string_overflow | defect | `readdir` | [`lib/posix/options/fs.c:313`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/fs.c) |
| string_overflow-03 | string_overflow | defect | `z_impl_k_thread_name_set` | [`kernel/thread.c:145`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/thread.c) |
| string_overflow-04 | string_overflow | defect | `z_impl_k_thread_name_copy` | [`kernel/thread.c:202`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/thread.c) |
| string_overflow-05 | string_overflow | control | `z_setup_new_thread` | [`kernel/thread.c:607`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/thread.c) |
| string_overflow-06 | string_overflow | control | `strerror_r` | [`lib/libc/minimal/source/string/strerror.c:56`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/libc/minimal/source/string/strerror.c) |
| string_overflow-07 | string_overflow | control | `mq_open` | [`lib/posix/options/mqueue.c:144`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/mqueue.c) |
| string_overflow-08 | string_overflow | control | `z_setup_new_thread` | [`kernel/thread.c:610`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/thread.c) |
| string_overflow-09 | string_overflow | control | `uname` | [`lib/posix/options/uname.c:50`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-10 | string_overflow | control | `inet_ntoa` | [`lib/posix/options/net.c:82`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/net.c) |
| string_overflow-11 | string_overflow | control | `sem_open` | [`lib/posix/options/semaphore.c:289`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/semaphore.c) |
| string_overflow-12 | string_overflow | defect | `strsignal` | [`lib/posix/options/signal.c:89`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/signal.c) |
| string_overflow-13 | string_overflow | defect | `uname` | [`lib/posix/options/uname.c:49`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-14 | string_overflow | defect | `utf8_lcpy` | [`lib/utils/utf8.c:72`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/lib/utils/utf8.c) |
| null_pointer-01 | null_pointer | defect | `z_impl_k_condvar_init` | [`kernel/condvar.c:23`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-02 | null_pointer | defect | `z_impl_k_condvar_signal` | [`kernel/condvar.c:50`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-03 | null_pointer | defect | `z_impl_k_condvar_broadcast` | [`kernel/condvar.c:87`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-04 | null_pointer | defect | `z_impl_k_condvar_wait` | [`kernel/condvar.c:120`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-05 | null_pointer | control | `z_impl_device_get_binding` | [`kernel/device.c:44`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/device.c) |
| null_pointer-06 | null_pointer | control | `z_impl_device_get_by_dt_nodelabel` | [`kernel/device.c:105`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/device.c) |
| null_pointer-07 | null_pointer | control | `z_impl_device_is_ready` | [`kernel/device.c:152`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/device.c) |
| null_pointer-08 | null_pointer | control | `dyn_cb` | [`kernel/dynamic.c:115`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/dynamic.c) |
| null_pointer-09 | null_pointer | control | `z_impl_z_errno` | [`kernel/errno.c:39`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/errno.c) |
| null_pointer-10 | null_pointer | control | `z_impl_k_event_init` | [`kernel/events.c:54`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/events.c) |
| null_pointer-11 | null_pointer | control | `event_walk_op` | [`kernel/events.c:106`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/events.c) |
| null_pointer-12 | null_pointer | defect | `k_event_post_internal` | [`kernel/events.c:140`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/events.c) |
| null_pointer-13 | null_pointer | defect | `k_event_wait_internal` | [`kernel/events.c:261`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/events.c) |
| null_pointer-14 | null_pointer | defect | `z_fatal_error` | [`kernel/fatal.c:99`](artifacts/contract3/engineering-v5/balanced-hard/zephyr/project/kernel/fatal.c) |

## balanced-hard / RT-Thread

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | defect | `rt_thread_idle_sethook` | [`src/idle.c:80`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/idle.c) |
| array_oob-02 | array_oob | defect | `rt_thread_idle_delhook` | [`src/idle.c:111`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/idle.c) |
| array_oob-03 | array_oob | defect | `idle_thread_entry` | [`src/idle.c:147`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/idle.c) |
| array_oob-04 | array_oob | defect | `get_components` | [`src/klibc/rt_vsnprintf_std.c:553`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-05 | array_oob | control | `get_normalized_components` | [`src/klibc/rt_vsnprintf_std.c:645`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-06 | array_oob | control | `print_exponential_number` | [`src/klibc/rt_vsnprintf_std.c:850`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-07 | array_oob | control | `memheaptrace` | [`src/memheap.c:984`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/memheap.c) |
| array_oob-08 | array_oob | control | `rt_object_get_information` | [`src/object.c:255`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/object.c) |
| array_oob-09 | array_oob | control | `_get_global_highest_ready_prio` | [`src/scheduler_mp.c:188`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-10 | array_oob | control | `_get_local_highest_ready_prio` | [`src/scheduler_mp.c:205`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-11 | array_oob | control | `_scheduler_get_highest_priority_thread` | [`src/scheduler_mp.c:246`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-12 | array_oob | defect | `_sched_insert_thread_locked` | [`src/scheduler_mp.c:303`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-13 | array_oob | defect | `_sched_remove_thread_locked` | [`src/scheduler_mp.c:368`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-14 | array_oob | defect | `_scheduler_get_highest_priority_thread` | [`src/scheduler_up.c:98`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/scheduler_up.c) |
| string_overflow-01 | string_overflow | defect | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:499`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-02 | string_overflow | defect | `finsh_get_prompt` | [`components/finsh/shell.c:102`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/components/finsh/shell.c) |
| string_overflow-03 | string_overflow | defect | `rt_thread_idle_init` | [`src/idle.c:185`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/idle.c) |
| string_overflow-04 | string_overflow | defect | `rt_snprintf` | [`src/klibc/kstdio.c:34`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-05 | string_overflow | control | `rt_vsprintf` | [`src/klibc/kstdio.c:54`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-06 | string_overflow | control | `rt_vsnprintf` | [`src/klibc/kstdio.c:83`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-07 | string_overflow | control | `rt_strncpy` | [`src/klibc/kstring.c:364`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-08 | string_overflow | control | `print_floating_point` | [`src/klibc/rt_vsnprintf_std.c:975`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-09 | string_overflow | control | `finsh_wait_auth` | [`components/finsh/shell.c:357`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/components/finsh/shell.c) |
| string_overflow-10 | string_overflow | control | `rt_strcpy` | [`src/klibc/kstring.c:405`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-11 | string_overflow | control | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:514`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-12 | string_overflow | defect | `rt_kprintf` | [`src/kservice.c:373`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/kservice.c) |
| string_overflow-13 | string_overflow | defect | `rt_object_init` | [`src/object.c:393`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/object.c) |
| string_overflow-14 | string_overflow | defect | `rt_object_allocate` | [`src/object.c:494`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/object.c) |
| null_pointer-01 | null_pointer | defect | `_update_process_times` | [`src/clock.c:110`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/clock.c) |
| null_pointer-02 | null_pointer | defect | `rt_components_board_init` | [`src/components.c:93`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/components.c) |
| null_pointer-03 | null_pointer | defect | `rt_components_init` | [`src/components.c:119`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/components.c) |
| null_pointer-04 | null_pointer | defect | `rt_spin_lock_init` | [`src/cpu_mp.c:39`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-05 | null_pointer | control | `rt_spin_lock` | [`src/cpu_mp.c:54`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-06 | null_pointer | control | `rt_spin_unlock` | [`src/cpu_mp.c:68`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-07 | null_pointer | control | `rt_spin_lock_irqsave` | [`src/cpu_mp.c:89`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-08 | null_pointer | control | `rt_spin_unlock_irqrestore` | [`src/cpu_mp.c:107`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-09 | null_pointer | control | `rt_cpus_lock` | [`src/cpu_mp.c:147`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-10 | null_pointer | control | `rt_cpus_unlock` | [`src/cpu_mp.c:180`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-11 | null_pointer | control | `rt_thread_defunct_dequeue` | [`src/defunct.c:56`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/defunct.c) |
| null_pointer-12 | null_pointer | defect | `rt_defunct_execute` | [`src/defunct.c:88`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/defunct.c) |
| null_pointer-13 | null_pointer | defect | `_ipc_object_init` | [`src/ipc.c:85`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/ipc.c) |
| null_pointer-14 | null_pointer | defect | `rt_susp_list_dequeue` | [`src/ipc.c:117`](artifacts/contract3/engineering-v5/balanced-hard/rt_thread/project/src/ipc.c) |

## independent-hard / FreeRTOS Kernel

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | control | `vCoRoutineSchedule` | [`croutine.c:313`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| array_oob-02 | array_oob | control | `prvInitialiseCoRoutineLists` | [`croutine.c:342`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| array_oob-03 | array_oob | control | `vQueueAddToRegistry` | [`queue.c:3026`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/queue.c) |
| array_oob-04 | array_oob | control | `pcQueueGetName` | [`queue.c:3074`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/queue.c) |
| array_oob-05 | array_oob | defect | `vQueueUnregisterQueue` | [`queue.c:3107`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/queue.c) |
| array_oob-06 | array_oob | defect | `prvCheckForRunStateChange` | [`tasks.c:815`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-07 | array_oob | defect | `prvYieldForTask` | [`tasks.c:901`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-08 | array_oob | defect | `prvSelectHighestPriorityTask` | [`tasks.c:1010`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-09 | array_oob | defect | `vTaskDelete` | [`tasks.c:2267`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-10 | array_oob | defect | `eTaskGetState` | [`tasks.c:2551`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-11 | array_oob | defect | `vTaskPrioritySet` | [`tasks.c:2907`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-12 | array_oob | control | `vTaskSuspend` | [`tasks.c:3164`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-13 | array_oob | control | `prvTaskIsTaskSuspended` | [`tasks.c:3317`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| array_oob-14 | array_oob | control | `xTaskResumeFromISR` | [`tasks.c:3465`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-01 | string_overflow | control | `prvInitialiseNewTask` | [`tasks.c:1867`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-02 | string_overflow | control | `prvWriteNameToBuffer` | [`tasks.c:7198`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-03 | string_overflow | control | `prvCreateIdleTasks` | [`tasks.c:3567`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-04 | string_overflow | control | `prvWriteNameToBuffer` | [`tasks.c:7192`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-05 | string_overflow | defect | `vTaskListTasks` | [`tasks.c:7325`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-06 | string_overflow | defect | `vTaskGetRunTimeStatistics` | [`tasks.c:7466`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-07 | string_overflow | defect | `prvCreateIdleTasks` | [`tasks.c:3572`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-08 | string_overflow | defect | `prvCreateIdleTasks` | [`tasks.c:3521`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-09 | string_overflow | defect | `prvInitialiseNewTask` | [`tasks.c:1884`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-10 | string_overflow | defect | `prvWriteNameToBuffer` | [`tasks.c:7202`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-11 | string_overflow | defect | `vTaskListTasks` | [`tasks.c:7330`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-12 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7475`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-13 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7492`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| string_overflow-14 | string_overflow | control | `vTaskGetRunTimeStatistics` | [`tasks.c:7504`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/tasks.c) |
| null_pointer-01 | null_pointer | control | `xCoRoutineCreate` | [`croutine.c:137`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| null_pointer-02 | null_pointer | control | `vCoRoutineAddToDelayedList` | [`croutine.c:186`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| null_pointer-03 | null_pointer | control | `xEventGroupSync` | [`event_groups.c:207`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-04 | null_pointer | control | `prvCheckDelayedList` | [`croutine.c:266`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| null_pointer-05 | null_pointer | defect | `xCoRoutineRemoveFromEventList` | [`croutine.c:367`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| null_pointer-06 | null_pointer | defect | `xEventGroupCreateStatic` | [`event_groups.c:113`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-07 | null_pointer | defect | `xEventGroupCreate` | [`event_groups.c:158`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-08 | null_pointer | defect | `prvCheckPendingReadyList` | [`croutine.c:228`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/croutine.c) |
| null_pointer-09 | null_pointer | defect | `xEventGroupWaitBits` | [`event_groups.c:338`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-10 | null_pointer | defect | `xEventGroupClearBits` | [`event_groups.c:491`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-11 | null_pointer | defect | `xEventGroupGetBitsFromISR` | [`event_groups.c:537`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-12 | null_pointer | control | `xEventGroupSetBits` | [`event_groups.c:565`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-13 | null_pointer | control | `vEventGroupDelete` | [`event_groups.c:656`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |
| null_pointer-14 | null_pointer | control | `xEventGroupGetStaticBuffer` | [`event_groups.c:712`](artifacts/contract3/engineering-v5/independent-hard/freertos_kernel/project/event_groups.c) |

## independent-hard / Zephyr RTOS

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/independent-hard/zephyr/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | control | `init_mbox_module` | [`kernel/mailbox.c:74`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/mailbox.c) |
| array_oob-02 | array_oob | control | `z_paging_histogram_inc` | [`kernel/paging/statistics.c:179`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/paging/statistics.c) |
| array_oob-03 | array_oob | control | `slice_timeout` | [`kernel/timeslicing.c:58`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/timeslicing.c) |
| array_oob-04 | array_oob | control | `z_reset_time_slice` | [`kernel/timeslicing.c:73`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/timeslicing.c) |
| array_oob-05 | array_oob | defect | `z_time_slice` | [`kernel/timeslicing.c:117`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/timeslicing.c) |
| array_oob-06 | array_oob | defect | `thread_idx_alloc` | [`kernel/userspace.c:275`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/userspace.c) |
| array_oob-07 | array_oob | defect | `thread_idx_free` | [`kernel/userspace.c:314`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/userspace.c) |
| array_oob-08 | array_oob | defect | `pthread_barrier_pool_init` | [`lib/posix/options/barrier.c:204`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/barrier.c) |
| array_oob-09 | array_oob | defect | `posix_fs_alloc_obj` | [`lib/posix/options/fs.c:45`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/fs.c) |
| array_oob-10 | array_oob | defect | `acquire_mutex` | [`lib/posix/options/mutex.c:126`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-11 | array_oob | defect | `pthread_mutex_init` | [`lib/posix/options/mutex.c:229`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/mutex.c) |
| array_oob-12 | array_oob | control | `pthread_setname_np` | [`lib/posix/options/pthread.c:1366`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-13 | array_oob | control | `pthread_getname_np` | [`lib/posix/options/pthread.c:1395`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/pthread.c) |
| array_oob-14 | array_oob | control | `pthread_spin_lock` | [`lib/posix/options/spinlock.c:126`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/spinlock.c) |
| string_overflow-01 | string_overflow | control | `z_impl_k_thread_name_set` | [`kernel/thread.c:146`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/thread.c) |
| string_overflow-02 | string_overflow | control | `readdir` | [`lib/posix/options/fs.c:313`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/fs.c) |
| string_overflow-03 | string_overflow | control | `z_impl_k_thread_name_set` | [`kernel/thread.c:145`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/thread.c) |
| string_overflow-04 | string_overflow | control | `z_impl_k_thread_name_copy` | [`kernel/thread.c:202`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/thread.c) |
| string_overflow-05 | string_overflow | defect | `z_setup_new_thread` | [`kernel/thread.c:607`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/thread.c) |
| string_overflow-06 | string_overflow | defect | `strerror_r` | [`lib/libc/minimal/source/string/strerror.c:56`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/libc/minimal/source/string/strerror.c) |
| string_overflow-07 | string_overflow | defect | `mq_open` | [`lib/posix/options/mqueue.c:144`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/mqueue.c) |
| string_overflow-08 | string_overflow | defect | `z_setup_new_thread` | [`kernel/thread.c:609`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/thread.c) |
| string_overflow-09 | string_overflow | defect | `uname` | [`lib/posix/options/uname.c:50`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-10 | string_overflow | defect | `inet_ntoa` | [`lib/posix/options/net.c:82`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/net.c) |
| string_overflow-11 | string_overflow | defect | `sem_open` | [`lib/posix/options/semaphore.c:289`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/semaphore.c) |
| string_overflow-12 | string_overflow | control | `strsignal` | [`lib/posix/options/signal.c:89`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/signal.c) |
| string_overflow-13 | string_overflow | control | `uname` | [`lib/posix/options/uname.c:49`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/posix/options/uname.c) |
| string_overflow-14 | string_overflow | control | `utf8_lcpy` | [`lib/utils/utf8.c:72`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/lib/utils/utf8.c) |
| null_pointer-01 | null_pointer | control | `z_impl_k_condvar_init` | [`kernel/condvar.c:23`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-02 | null_pointer | control | `z_impl_k_condvar_signal` | [`kernel/condvar.c:50`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-03 | null_pointer | control | `z_impl_k_condvar_broadcast` | [`kernel/condvar.c:87`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-04 | null_pointer | control | `z_impl_k_condvar_wait` | [`kernel/condvar.c:120`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/condvar.c) |
| null_pointer-05 | null_pointer | defect | `z_impl_device_get_binding` | [`kernel/device.c:44`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/device.c) |
| null_pointer-06 | null_pointer | defect | `z_impl_device_get_by_dt_nodelabel` | [`kernel/device.c:105`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/device.c) |
| null_pointer-07 | null_pointer | defect | `z_impl_device_is_ready` | [`kernel/device.c:152`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/device.c) |
| null_pointer-08 | null_pointer | defect | `dyn_cb` | [`kernel/dynamic.c:115`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/dynamic.c) |
| null_pointer-09 | null_pointer | defect | `z_impl_z_errno` | [`kernel/errno.c:39`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/errno.c) |
| null_pointer-10 | null_pointer | defect | `z_impl_k_event_init` | [`kernel/events.c:54`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/events.c) |
| null_pointer-11 | null_pointer | defect | `event_walk_op` | [`kernel/events.c:106`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/events.c) |
| null_pointer-12 | null_pointer | control | `k_event_post_internal` | [`kernel/events.c:140`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/events.c) |
| null_pointer-13 | null_pointer | control | `k_event_wait_internal` | [`kernel/events.c:261`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/events.c) |
| null_pointer-14 | null_pointer | control | `z_fatal_error` | [`kernel/fatal.c:99`](artifacts/contract3/engineering-v5/independent-hard/zephyr/project/kernel/fatal.c) |

## independent-hard / RT-Thread

[源码、外置 GT 和哈希清单](artifacts/contract3/engineering-v5/independent-hard/rt_thread/)

| 用例 | 类别 | 标签 | 原有函数 | 文件与操作行 |
|---|---|---|---|---|
| array_oob-01 | array_oob | control | `rt_thread_idle_sethook` | [`src/idle.c:80`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/idle.c) |
| array_oob-02 | array_oob | control | `rt_thread_idle_delhook` | [`src/idle.c:111`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/idle.c) |
| array_oob-03 | array_oob | control | `idle_thread_entry` | [`src/idle.c:147`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/idle.c) |
| array_oob-04 | array_oob | control | `get_components` | [`src/klibc/rt_vsnprintf_std.c:553`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-05 | array_oob | defect | `get_normalized_components` | [`src/klibc/rt_vsnprintf_std.c:645`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-06 | array_oob | defect | `print_exponential_number` | [`src/klibc/rt_vsnprintf_std.c:850`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| array_oob-07 | array_oob | defect | `memheaptrace` | [`src/memheap.c:984`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/memheap.c) |
| array_oob-08 | array_oob | defect | `rt_object_get_information` | [`src/object.c:255`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/object.c) |
| array_oob-09 | array_oob | defect | `_get_global_highest_ready_prio` | [`src/scheduler_mp.c:188`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-10 | array_oob | defect | `_get_local_highest_ready_prio` | [`src/scheduler_mp.c:205`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-11 | array_oob | defect | `_scheduler_get_highest_priority_thread` | [`src/scheduler_mp.c:246`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-12 | array_oob | control | `_sched_insert_thread_locked` | [`src/scheduler_mp.c:303`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-13 | array_oob | control | `_sched_remove_thread_locked` | [`src/scheduler_mp.c:368`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/scheduler_mp.c) |
| array_oob-14 | array_oob | control | `_scheduler_get_highest_priority_thread` | [`src/scheduler_up.c:98`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/scheduler_up.c) |
| string_overflow-01 | string_overflow | control | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:499`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-02 | string_overflow | control | `finsh_get_prompt` | [`components/finsh/shell.c:102`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/components/finsh/shell.c) |
| string_overflow-03 | string_overflow | control | `rt_thread_idle_init` | [`src/idle.c:185`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/idle.c) |
| string_overflow-04 | string_overflow | control | `rt_snprintf` | [`src/klibc/kstdio.c:34`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-05 | string_overflow | defect | `rt_vsprintf` | [`src/klibc/kstdio.c:54`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-06 | string_overflow | defect | `rt_vsnprintf` | [`src/klibc/kstdio.c:83`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/kstdio.c) |
| string_overflow-07 | string_overflow | defect | `rt_strncpy` | [`src/klibc/kstring.c:364`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-08 | string_overflow | defect | `print_floating_point` | [`src/klibc/rt_vsnprintf_std.c:975`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-09 | string_overflow | defect | `finsh_wait_auth` | [`components/finsh/shell.c:357`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/components/finsh/shell.c) |
| string_overflow-10 | string_overflow | defect | `rt_strcpy` | [`src/klibc/kstring.c:405`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/kstring.c) |
| string_overflow-11 | string_overflow | defect | `print_integer` | [`src/klibc/rt_vsnprintf_std.c:514`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/klibc/rt_vsnprintf_std.c) |
| string_overflow-12 | string_overflow | control | `rt_kprintf` | [`src/kservice.c:373`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/kservice.c) |
| string_overflow-13 | string_overflow | control | `rt_object_init` | [`src/object.c:393`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/object.c) |
| string_overflow-14 | string_overflow | control | `rt_object_allocate` | [`src/object.c:494`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/object.c) |
| null_pointer-01 | null_pointer | control | `_update_process_times` | [`src/clock.c:110`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/clock.c) |
| null_pointer-02 | null_pointer | control | `rt_components_board_init` | [`src/components.c:93`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/components.c) |
| null_pointer-03 | null_pointer | control | `rt_components_init` | [`src/components.c:119`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/components.c) |
| null_pointer-04 | null_pointer | control | `rt_spin_lock_init` | [`src/cpu_mp.c:39`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-05 | null_pointer | defect | `rt_spin_lock` | [`src/cpu_mp.c:54`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-06 | null_pointer | defect | `rt_spin_unlock` | [`src/cpu_mp.c:68`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-07 | null_pointer | defect | `rt_spin_lock_irqsave` | [`src/cpu_mp.c:89`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-08 | null_pointer | defect | `rt_spin_unlock_irqrestore` | [`src/cpu_mp.c:107`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-09 | null_pointer | defect | `rt_cpus_lock` | [`src/cpu_mp.c:147`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-10 | null_pointer | defect | `rt_cpus_unlock` | [`src/cpu_mp.c:180`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/cpu_mp.c) |
| null_pointer-11 | null_pointer | defect | `rt_thread_defunct_dequeue` | [`src/defunct.c:56`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/defunct.c) |
| null_pointer-12 | null_pointer | control | `rt_defunct_execute` | [`src/defunct.c:88`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/defunct.c) |
| null_pointer-13 | null_pointer | control | `_ipc_object_init` | [`src/ipc.c:85`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/ipc.c) |
| null_pointer-14 | null_pointer | control | `rt_susp_list_dequeue` | [`src/ipc.c:117`](artifacts/contract3/engineering-v5/independent-hard/rt_thread/project/src/ipc.c) |
