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
}

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

export interface RipgrepRunner {
  run(request: RipgrepRequest): Promise<RipgrepResult>
  healthCheck(signal?: AbortSignal): Promise<void>
}

export class RipgrepError extends Error {
  readonly code: 'unavailable' | 'timeout' | 'interrupted' | 'failed'

  constructor(
    message: string,
    code: 'unavailable' | 'timeout' | 'interrupted' | 'failed',
  ) {
    super(message)
    this.name = 'RipgrepError'
    this.code = code
  }
}
