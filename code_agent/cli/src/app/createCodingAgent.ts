import { AgentRunner } from '../agent/AgentRunner.js'
import type { ContextProvider, GitStatusSampler } from '../agent/types/content.js'
import { getSystemInstruction } from '../agent/content/systemInstruction.js'
import { renderSystemPrompt } from '../agent/content/renderSystemPrompt.js'
import { createAgentThread } from '../agent/thread/AgentThread.js'
import type { AgentRuntime } from '../agent/types/types.js'
import type { ModelBudgetCapabilities, ModelClient } from '../model/types/types.js'
import { createModelClient } from '../model/createModelClient.js'
import { createModelCompactionProvider } from '../agent/compaction/provider.js'
import type { ToolRegistry } from '../tools/types.js'
import { createToolRegistry } from '../tools/registry.js'
import { createRipgrepRunner, type RipgrepRunner } from '../tools/backends/ripgrep/index.js'
import type { GitChangesBackend } from '../tools/backends/git/index.js'
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
import {
  createSystemEnvironmentClock,
  observeEnvironment,
  SessionEnvironmentContext,
  type EnvironmentClock,
} from './session/SessionEnvironmentContext.js'

export interface CreateCodingAgentOptions extends CodingAgentConfigInput {
  modelClient?: ModelClient
  /** Trusted endpoint/model budget metadata; never inferred from the model name. */
  modelBudgetCapability?: ModelBudgetCapabilities & {
    readonly countInputTokens?: (body: Readonly<Record<string, unknown>>) => number
  }
  /** override the default model-backed compaction provider (tests). */
  compactionProvider?: import('../agent/compaction/contracts.js').CompactionProvider
  tools?: ToolRegistry
  runner?: AgentRunner
  approvalProvider?: ApprovalProvider
  ripgrepRunner?: RipgrepRunner
  /** Optional Git tool injection; omitted until the model-visible Tool passes admission. */
  gitChangesBackend?: GitChangesBackend
  /** App-selected context extension point; providers are composed before profile creation. */
  contextProviders?: readonly ContextProvider[]
  /** Trusted Session-scoped Git sampler override for deterministic tests. */
  gitStatusSampler?: GitStatusSampler
  /** Session environment clock; read only at initial build/request freeze. */
  environmentClock?: EnvironmentClock
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
      requestBudgetCapability: options.modelBudgetCapability,
    })
    validateRunBudgetCapabilities(config.loop.runBudget, modelClient)
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
    const tools = options.tools ?? createToolRegistry(undefined, ripgrepRunner, options.gitChangesBackend)
    const runner = options.runner ?? new AgentRunner()
    const contextProvider = createCodingAgentContextProvider(
      options.contextProviders,
      options.gitStatusSampler ? { gitStatusSampler: options.gitStatusSampler } : {},
    )
    const environmentClock = options.environmentClock ?? createSystemEnvironmentClock()
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
      // default injection: one tools-disabled, model-backed
      // maintenance provider per application (streaming accumulator with a
      // complete() fallback that only bounds returned content).
      ...(options.compactionProvider ? {} : {
        compactionProvider: createModelCompactionProvider({
          modelClient,
          model: config.model.model,
          ...(config.loop.runBudget?.maxOutputTokensPerRequest === undefined &&
          config.loop.runBudget?.safetyMarginTokens === undefined
            ? {}
            : {
                requestBudget: {
                  outputTokenLimit: config.loop.runBudget?.maxOutputTokensPerRequest,
                  safetyMarginTokens: config.loop.runBudget?.safetyMarginTokens,
                },
              }),
        }),
      }),
      ...(options.compactionProvider ? { compactionProvider: options.compactionProvider } : {}),
    }

    const manager = new SessionManager(async ({ signal }) => {
      const initialEnvironment = observeEnvironment(environmentClock)
      const contribution = await contextProvider.build({
        projectDir: config.codingAgent.projectDir,
        now: new Date(initialEnvironment.observedAt),
        timeZone: initialEnvironment.timeZone,
        signal,
      })
      throwIfAborted(signal)
      const context = {
        projectMemory: contribution.projectMemory ?? '',
        gitStatus: contribution.gitStatus ?? '',
        projectName: contribution.projectName ?? '',
        currentDate: contribution.currentDate ?? '',
        ...(contribution.environment ? { environment: contribution.environment } : {}),
        warnings: contribution.warnings ?? [],
        sections: contribution.sections ?? [],
      }
      const systemPrompt = renderSystemPrompt({
        instruction: getSystemInstruction({
          gitChangesAvailable: tools.schemas().some((schema) => schema.name === 'git_changes'),
        }),
        context,
        limits: {
          maxProjectMemoryChars: config.codingAgent.maxProjectMemoryChars,
          maxDynamicContextChars: config.codingAgent.maxDynamicContextChars,
          maxSystemPromptChars: config.codingAgent.maxSystemPromptChars,
        },
      })
      const profile = createCodingAgentProfile(systemPrompt)
      return new CodingAgentSession({
        profile,
        runtime,
        thread: createAgentThread([profile.systemMessage]),
        runner,
        frameworkContext: new SessionEnvironmentContext(
          environmentClock,
          initialEnvironment,
        ),
      })
    })
    sessionManager = manager
    await manager.replace(options.signal)
    throwIfAborted(options.signal)
    handedOff = true
    return {
      config,
      get profile() {
        const current = manager.current()
        if (!current) {
          throw new AgentError('config_error', 'The coding agent application has no active session')
        }
        return current.profile
      },
      sessionManager: manager,
      dispose: () => manager.dispose(),
    }
  } finally {
    if (!handedOff) await sessionManager?.dispose()
  }
}

function validateRunBudgetCapabilities(
  budget: import('../agent/types/types.js').RunBudgetConfig | undefined,
  client: ModelClient,
): void {
  if (!budget) return
  if (budget.maxCostMicrousd !== undefined) {
    throw new AgentError(
      'config_error',
      'maxCostMicrousd requires trusted pricing and retry accounting, which are not configured',
    )
  }
  if (budget.maxTotalTokens === undefined && budget.maxOutputTokensPerRequest === undefined) return
  const capability = client.budgetCapabilities
  if (
    !client.prepare ||
    !capability?.contextWindowTokens ||
    capability.inputTokenEstimation !== 'exact_tokenizer' ||
    capability.outputLimit?.covers !== 'completion_and_reasoning'
  ) {
    throw new AgentError(
      'config_error',
      'Hard token budgets require exact input measurement, context metadata, provider output limits, and prepared dispatch support',
    )
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}
