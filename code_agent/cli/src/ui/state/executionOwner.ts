// [maint-note:a6d201d0403f]
// 数据契约导航：OwnedExecution；字段含义以消费端校验为准。
// 成员 runId：number。
// 成员 controller：AbortController。
// [/maint-note:a6d201d0403f]
export interface OwnedExecution {
  readonly runId: number
  readonly controller: AbortController
}

/** Synchronous execution ownership shared by the UI entry point and tests. */
export class ExecutionOwner {
  private nextRunId = 0
  private active: OwnedExecution | null = null

  // [maint-note:95b74715f5cc]
  // 维护导航：acquire；下列为本函数源码中的接口与路径索引。
  // 返回契约：OwnedExecution | null。
  // 分支条件：this.active。
  // 返回路径：null。
  // 返回路径：execution。
  // 写入目标：this.active。
  // [/maint-note:95b74715f5cc]
  acquire(): OwnedExecution | null {
    if (this.active) return null
    const execution: OwnedExecution = {
      runId: ++this.nextRunId,
      controller: new AbortController(),
    }
    this.active = execution
    return execution
  }

  // [maint-note:e9bab2162100]
  // 维护导航：current；下列为本函数源码中的接口与路径索引。
  // 返回契约：OwnedExecution | null。
  // 返回路径：this.active。
  // [/maint-note:e9bab2162100]
  current(): OwnedExecution | null {
    return this.active
  }

  // [maint-note:7cb8636ec144]
  // 维护导航：release；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：boolean。
  // 分支条件：this.active?.runId !== runId。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.active。
  // [/maint-note:7cb8636ec144]
  release(runId: number): boolean {
    if (this.active?.runId !== runId) return false
    this.active = null
    return true
  }

  // [maint-note:3cfc9a2ab164]
  // 维护导航：interrupt；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：!execution。
  // 返回路径：false。
  // 返回路径：true。
  // 直接调用：execution.controller.abort。
  // [/maint-note:3cfc9a2ab164]
  interrupt(): boolean {
    const execution = this.active
    if (!execution) return false
    execution.controller.abort()
    return true
  }
}
