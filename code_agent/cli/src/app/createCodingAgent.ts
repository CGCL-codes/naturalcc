import { AgentRunner } from '../agent/AgentRunner.js'
import type { ContextProvider } from '../agent/types/content.js'
import { getSystemInstruction } from '../agent/content/systemInstruction.js'
import { renderSystemPrompt } from '../agent/content/renderSystemPrompt.js'
import { createAgentThread } from '../agent/thread/AgentThread.js'
import type { AgentRuntime } from '../agent/types/types.js'
import type { ModelClient } from '../model/types/types.js'
import { createModelClient } from '../model/createModelClient.js'
import type { ToolRegistry } from '../tools/types.js'
import { createToolRegistry } from '../tools/registry.js'
import { createRipgrepRunner, type RipgrepRunner } from '../tools/ripgrep/index.js'
import {
  createToolPolicy,
  type ApprovalProvider,
} from '../tools/approval.js'
import {
  resolveCodingAgentConfig,
  type CodingAgentConfigInput,
  type ResolvedCodingAgentConfig,
} from './config.js'
import { createCodingAgentProfile, type CodingAgentProfile } from './profile.js'
import { CodingAgentSession } from './session/CodingAgentSession.js'
import { SessionManager } from './session/SessionManager.js'
import { AgentError, createAbortError, errorMessage } from '../agent/errors.js'
import { createCodingAgentContextProvider } from './context.js'

export interface CreateCodingAgentOptions extends CodingAgentConfigInput {
  modelClient?: ModelClient
  tools?: ToolRegistry
  runner?: AgentRunner
  approvalProvider?: ApprovalProvider
  ripgrepRunner?: RipgrepRunner
  /** App-selected context extension point; providers are composed before profile creation. */
  contextProviders?: readonly ContextProvider[]
  /** Test/locator hook; production defaults to PATH lookup for `rg`. */
  ripgrepExecutable?: string
  /** Cancels application initialization, including context providers. */
  signal?: AbortSignal
}

export interface CodingAgentApplication {
  readonly config: ResolvedCodingAgentConfig
  readonly profile: CodingAgentProfile
  readonly sessionManager: SessionManager
  dispose(): Promise<void>
}

export async function createCodingAgent(
  options: CreateCodingAgentOptions = {},
): Promise<CodingAgentApplication> {
  let sessionManager: SessionManager | undefined
  let handedOff = false
  try {
    throwIfAborted(options.signal)
    const config = await resolveCodingAgentConfig(
      options.modelClient &&
        options.apiKey === undefined &&
        !process.env.OPENAI_API_KEY
        ? { ...options, apiKey: '<injected-model-client>' }
      : options,
    )
    throwIfAborted(options.signal)
    const modelClient = options.modelClient ?? createModelClient({
      apiKey: config.model.apiKey,
      baseURL: config.model.baseURL,
    })
    const ripgrepRunner = options.ripgrepRunner ?? createRipgrepRunner(options.ripgrepExecutable)
    if (!options.tools && !options.ripgrepRunner) {
      try {
        await ripgrepRunner.healthCheck(options.signal)
      } catch (error) {
        if (options.signal?.aborted) throw createAbortError()
        throw new AgentError(
          'config_error',
          `ripgrep is required to start the coding agent. Install rg and verify that 'rg --no-config --version' works: ${errorMessage(error)}`,
        )
      }
    }
    throwIfAborted(options.signal)
    const tools = options.tools ?? createToolRegistry(undefined, ripgrepRunner)
    const runner = options.runner ?? new AgentRunner()
    const contextProvider = createCodingAgentContextProvider(options.contextProviders)
    const contribution = await contextProvider.build({
      projectDir: config.codingAgent.projectDir,
      now: new Date(),
      signal: options.signal,
    })
    throwIfAborted(options.signal)
    const context = {
      projectMemory: contribution.projectMemory ?? '',
      gitStatus: contribution.gitStatus ?? '',
      projectName: contribution.projectName ?? '',
      currentDate: contribution.currentDate ?? '',
      warnings: contribution.warnings ?? [],
      sections: contribution.sections ?? [],
    }
    const systemPrompt = renderSystemPrompt({
      instruction: getSystemInstruction(),
      context,
      limits: {
        maxProjectMemoryChars: config.codingAgent.maxProjectMemoryChars,
        maxDynamicContextChars: config.codingAgent.maxDynamicContextChars,
        maxSystemPromptChars: config.codingAgent.maxSystemPromptChars,
      },
    })
    const profile = createCodingAgentProfile(systemPrompt)
    const runtime: AgentRuntime = {
      modelClient,
      model: config.model.model,
      tools,
      loop: config.loop,
      toolContext: {
        projectDir: config.codingAgent.projectDir,
        timeoutMs: config.codingAgent.toolTimeoutMs,
        maxOutputChars: config.codingAgent.maxToolOutputChars,
      },
      showToolResults: config.codingAgent.showToolResults,
      toolPolicy: createToolPolicy(
        config.codingAgent.approvalMode,
        options.approvalProvider,
      ),
    }

    const manager = new SessionManager(() => new CodingAgentSession({
      profile,
      runtime,
      thread: createAgentThread([profile.systemMessage]),
      runner,
    }))
    sessionManager = manager
    await manager.replace()
    throwIfAborted(options.signal)
    handedOff = true
    return {
      config,
      profile,
      sessionManager: manager,
      dispose: () => manager.dispose(),
    }
  } finally {
    if (!handedOff) await sessionManager?.dispose()
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}
