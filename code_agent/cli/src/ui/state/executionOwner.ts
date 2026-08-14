export interface OwnedExecution {
  readonly runId: number
  readonly controller: AbortController
}

/** Synchronous execution ownership shared by the UI entry point and tests. */
export class ExecutionOwner {
  private nextRunId = 0
  private active: OwnedExecution | null = null

  acquire(): OwnedExecution | null {
    if (this.active) return null
    const execution: OwnedExecution = {
      runId: ++this.nextRunId,
      controller: new AbortController(),
    }
    this.active = execution
    return execution
  }

  current(): OwnedExecution | null {
    return this.active
  }

  release(runId: number): boolean {
    if (this.active?.runId !== runId) return false
    this.active = null
    return true
  }

  interrupt(): boolean {
    const execution = this.active
    if (!execution) return false
    execution.controller.abort()
    return true
  }
}
