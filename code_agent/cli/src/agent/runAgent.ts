import type { AgentRunOptions } from './types/types.js'
import type { AgentEvent } from './types/event.js'
import { AgentRunner } from './AgentRunner.js'
import { errorMessage, isAbortError } from './errors.js'
import { createAgentThread } from './thread/AgentThread.js'

// [maint-note:f659e5f6a449]
// 维护导航：runAgent；下列为本函数源码中的接口与路径索引。
// 输入 input：AgentRunOptions。
// 返回契约：AsyncGenerator<AgentEvent>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：isAbortError(error, input.signal)。
// 直接调用：createAgentThread。
// 直接调用：new AgentRunner().runTurn。
// 直接调用：isAbortError。
// 直接调用：errorMessage。
// [/maint-note:f659e5f6a449]
export async function* runAgent(
  input: AgentRunOptions,
): AsyncGenerator<AgentEvent> {
  try {
    const thread = input.thread ?? createAgentThread()
    yield* new AgentRunner().runTurn({
      thread,
      prompt: input.prompt,
      runtime: input.runtime,
      signal: input.signal,
    })
  } catch (error) {
    if (isAbortError(error, input.signal)) {
      yield {
        type: 'interrupted',
        message: 'Agent turn interrupted.',
        workspaceMayHaveChanged: false,
      }
    } else {
      yield {
        type: 'error',
        code: 'config_error',
        message: errorMessage(error),
        recoverable: false,
        workspaceMayHaveChanged: false,
      }
    }
  } finally {
    // The compatibility wrapper deliberately owns no session or runtime state.
  }
}
