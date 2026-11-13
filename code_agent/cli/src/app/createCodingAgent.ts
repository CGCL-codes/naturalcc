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

// [maint-note:81292e1e2f7d]
// 数据契约导航：CreateCodingAgentOptions；字段含义以消费端校验为准。
// 成员 modelClient（可省略）：ModelClient。
// 成员 modelBudgetCapability（可省略）：ModelBudgetCapabilities & { readonly countInputTokens?: (body: Readonly<Record<string, unknown>>) => number }。
// 成员 compactionProvider（可省略）：import('../agent/compaction/contracts.js').CompactionProvider。
// 成员 tools（可省略）：ToolRegistry。
// 成员 runner（可省略）：AgentRunner。
// 成员 approvalProvider（可省略）：ApprovalProvider。
// 成员 ripgrepRunner（可省略）：RipgrepRunner。
// 成员 gitChangesBackend（可省略）：GitChangesBackend。
// 成员 contextProviders（可省略）：readonly ContextProvider[]。
// 成员 gitStatusSampler（可省略）：GitStatusSampler。
// 成员 environmentClock（可省略）：EnvironmentClock。
// 成员 ripgrepExecutable（可省略）：string。
// 成员 signal（可省略）：AbortSignal。
// [/maint-note:81292e1e2f7d]
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

// [maint-note:0a51dee89831]
// 数据契约导航：CodingAgentApplication；字段含义以消费端校验为准。
// 成员 config：ResolvedCodingAgentConfig。
// 成员 profile：CodingAgentProfile。
// 成员 sessionManager：SessionManager。
// 成员 dispose：Promise<void>。
// [/maint-note:0a51dee89831]
export interface CodingAgentApplication {
  readonly config: ResolvedCodingAgentConfig
  readonly profile: CodingAgentProfile
  readonly sessionManager: SessionManager
  dispose(): Promise<void>
}

// [maint-note:df66b55a63a6]
// 维护导航：createCodingAgent；下列为本函数源码中的接口与路径索引。
// 输入 options：CreateCodingAgentOptions；缺省表达式 {}。
// 返回契约：Promise<CodingAgentApplication>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!options.tools && !options.ripgrepRunner。
// 分支条件：options.signal?.aborted。
// 分支条件：!handedOff。
// 返回路径：{ config, get profile() { const current = manager.current() if (!current) { throw new AgentError('config_error', 'The coding agent application has no active session') } return current.profile }, sessionManager: manager, dispose: () => manag。
// 显式抛出：createAbortError()。
// 显式抛出：new AgentError( 'config_error', `ripgrep is required to start the coding agent. Install rg and verify that 'rg --no-config --version' works: ${errorMessage(error)}`, )。
// 写入目标：sessionManager。
// 写入目标：handedOff。
// 直接调用：throwIfAborted。
// 直接调用：resolveCodingAgentConfig。
// 直接调用：createModelClient。
// 直接调用：validateRunBudgetCapabilities。
// 直接调用：createRipgrepRunner。
// 直接调用：ripgrepRunner.healthCheck。
// 直接调用：createAbortError。
// 直接调用：errorMessage。
// 直接调用：createToolRegistry。
// 直接调用：createCodingAgentContextProvider。
// 直接调用：createSystemEnvironmentClock。
// 直接调用：createToolPolicy。
// 直接调用：createModelCompactionProvider。
// 直接调用：manager.replace。
// 直接调用：sessionManager?.dispose。
// [/maint-note:df66b55a63a6]
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
      // [maint-note:2d0502676cab]
      // 维护导航：profile；下列为本函数源码中的接口与路径索引。
      // 分支条件：!current。
      // 返回路径：current.profile。
      // 显式抛出：new AgentError('config_error', 'The coding agent application has no active session')。
      // 直接调用：manager.current。
      // [/maint-note:2d0502676cab]
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

// [maint-note:9ca4b28d74da]
// 维护导航：validateRunBudgetCapabilities；下列为本函数源码中的接口与路径索引。
// 输入 budget：import('../agent/types/types.js').RunBudgetConfig | undefined。
// 输入 client：ModelClient。
// 返回契约：void。
// 分支条件：!budget。
// 分支条件：budget.maxCostMicrousd !== undefined。
// 分支条件：budget.maxTotalTokens === undefined && budget.maxOutputTokensPerRequest === undefined。
// 分支条件：!client.prepare || !capability?.contextWindowTokens || capability.inputTokenEstimation !== 'exact_tokenizer' || capability.outputLimit?.covers !== 'completion_and_reasoning'。
// 返回路径：无返回值。
// 显式抛出：new AgentError( 'config_error', 'maxCostMicrousd requires trusted pricing and retry accounting, which are not configured', )。
// 显式抛出：new AgentError( 'config_error', 'Hard token budgets require exact input measurement, context metadata, provider output limits, and prepared dispatch support', )。
// [/maint-note:9ca4b28d74da]
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

// [maint-note:86dabfa0aaae]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：createAbortError()。
// 直接调用：createAbortError。
// [/maint-note:86dabfa0aaae]
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}
