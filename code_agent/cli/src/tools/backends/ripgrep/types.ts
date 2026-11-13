import { ToolDiagnosticError } from '../../types.js'
import { toolDiagnostic } from '../../diagnostics.js'

// [maint-note:f2a104e045c7]
// 数据契约导航：RipgrepRequest；字段含义以消费端校验为准。
// 成员 args：readonly string[]。
// 成员 cwd：string。
// 成员 signal（可省略）：AbortSignal。
// 成员 timeoutMs（可省略）：number。
// 成员 maxOutputBytes：number。
// 成员 stdoutRecordDelimiter（可省略）：0 | 10。
// 成员 onStdoutRecord（可省略）：(record: Buffer) => boolean。
// 成员 stdoutRecordRetention（可省略）：'retain' | 'discard'。
// [/maint-note:f2a104e045c7]
export interface RipgrepRequest {
  args: readonly string[]
  cwd: string
  signal?: AbortSignal
  timeoutMs?: number
  maxOutputBytes: number
  /** Frame stdout before invoking the consumer; supported delimiters are NUL and LF. */
  stdoutRecordDelimiter?: 0 | 10
  /** Return false after a complete record to request a bounded early stop. */
  onStdoutRecord?: (record: Buffer) => boolean
  /**
   * Whether a complete framed record is kept in `RipgrepResult.stdout`. Defaults to
   * `"retain"` (current behavior). `"discard"` drops the record after the consumer
   * observes it, so callers that only need a count never accumulate matched paths.
   */
  stdoutRecordRetention?: 'retain' | 'discard'
}

// [maint-note:43e99f040f57]
// 数据契约导航：RipgrepResult；字段含义以消费端校验为准。
// 成员 stdout：string。
// 成员 stderr：string。
// 成员 exitCode：number。
// 成员 truncated：boolean。
// 成员 timedOut：boolean。
// 成员 interrupted：boolean。
// 成员 outputBytes：number。
// 成员 stoppedEarly（可省略）：boolean。
// 成员 terminationReason（可省略）：'output' | 'consumer' | 'timeout' | 'abort' | 'callback'。
// [/maint-note:43e99f040f57]
export interface RipgrepResult {
  stdout: string
  stderr: string
  exitCode: number
  truncated: boolean
  timedOut: boolean
  interrupted: boolean
  outputBytes: number
  stoppedEarly?: boolean
  terminationReason?: 'output' | 'consumer' | 'timeout' | 'abort' | 'callback'
}

// [maint-note:a82ce87c3cc0]
// 数据契约导航：RipgrepRunner；字段含义以消费端校验为准。
// 成员 run：Promise<RipgrepResult>。
// 成员 healthCheck：Promise<void>。
// [/maint-note:a82ce87c3cc0]
export interface RipgrepRunner {
  run(request: RipgrepRequest): Promise<RipgrepResult>
  healthCheck(signal?: AbortSignal): Promise<void>
}

export class RipgrepError extends ToolDiagnosticError {
  readonly code: 'unavailable' | 'timeout' | 'interrupted' | 'failed'

  // [maint-note:0312a0de3586]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 message：string。
  // 输入 code：'unavailable' | 'timeout' | 'interrupted' | 'failed'。
  // 写入目标：this.name。
  // 写入目标：this.code。
  // 直接调用：super。
  // 直接调用：toolDiagnostic。
  // [/maint-note:0312a0de3586]
  constructor(
    message: string,
    code: 'unavailable' | 'timeout' | 'interrupted' | 'failed',
  ) {
    super(message, toolDiagnostic(code === 'unavailable' ? 'backend_unavailable' : code === 'failed' ? 'backend_failed' : code))
    this.name = 'RipgrepError'
    this.code = code
  }
}
