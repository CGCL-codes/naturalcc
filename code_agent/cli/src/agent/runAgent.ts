import type { AgentRunOptions } from './types/types.js'
import type { AgentEvent } from './types/event.js'
import { AgentRunner } from './AgentRunner.js'
import { errorMessage, isAbortError } from './errors.js'
import { createAgentThread } from './thread/AgentThread.js'

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
