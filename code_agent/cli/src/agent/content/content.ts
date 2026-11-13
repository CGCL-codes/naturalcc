import type {
  AgentPromptContext,
  ContextBuildInput,
  ContextContribution,
  ContextProvider,
  GitStatusSampler,
} from '../types/content.js'
import { createAbortError } from '../errors.js'
import { createDynamicContextProvider } from './dynamicContext.js'
import { projectMemoryProvider } from './projectMemory.js'

// [maint-note:9057aa83243e]
// 维护导航：composeContextProviders；下列为本函数源码中的接口与路径索引。
// 输入 providers：readonly ContextProvider[]。
// 返回契约：ContextProvider。
// 返回路径：{ name: providers.map((provider) => provider.name).join('+') || 'empty-context', async build(input: ContextBuildInput) { if (input.signal?.aborted) throw createAbortError() const contributions = await Promise.all( providers.map((provider) =。
// 直接调用：providers.map((provider) => provider.name).join。
// 直接调用：providers.map。
// [/maint-note:9057aa83243e]
export function composeContextProviders(
  providers: readonly ContextProvider[],
): ContextProvider {
  return {
    name: providers.map((provider) => provider.name).join('+') || 'empty-context',
    // [maint-note:163615078577]
    // 维护导航：build；下列为本函数源码中的接口与路径索引。
    // 输入 input：ContextBuildInput。
    // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
    // 分支条件：input.signal?.aborted。
    // 返回路径：mergeContributions(contributions)。
    // 显式抛出：createAbortError()。
    // 直接调用：createAbortError。
    // 直接调用：Promise.all。
    // 直接调用：providers.map。
    // 直接调用：mergeContributions。
    // [/maint-note:163615078577]
    async build(input: ContextBuildInput) {
      if (input.signal?.aborted) throw createAbortError()
      const contributions = await Promise.all(
        providers.map((provider) => withAbort(
          Promise.resolve().then(() => provider.build(input)),
          input.signal,
        )),
      )
      if (input.signal?.aborted) throw createAbortError()
      return mergeContributions(contributions)
    },
  }
}

// [maint-note:9311e030bfde]
// 维护导航：createDefaultContextProvider；下列为本函数源码中的接口与路径索引。
// 输入 extraProviders：readonly ContextProvider[]；缺省表达式 []。
// 输入 options：{ readonly gitStatusSampler?: GitStatusSampler }；缺省表达式 {}。
// 返回契约：ContextProvider。
// 返回路径：composeContextProviders([ projectMemoryProvider, createDynamicContextProvider(options.gitStatusSampler), ...extraProviders, ])。
// 直接调用：composeContextProviders。
// 直接调用：createDynamicContextProvider。
// [/maint-note:9311e030bfde]
export function createDefaultContextProvider(
  extraProviders: readonly ContextProvider[] = [],
  options: { readonly gitStatusSampler?: GitStatusSampler } = {},
): ContextProvider {
  return composeContextProviders([
    projectMemoryProvider,
    createDynamicContextProvider(options.gitStatusSampler),
    ...extraProviders,
  ])
}

// [maint-note:2a50ccad9ca1]
// 维护导航：buildAgentContext；下列为本函数源码中的接口与路径索引。
// 输入 input：ContextBuildInput。
// 返回契约：Promise<AgentPromptContext>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：{ projectMemory: contribution.projectMemory ?? '', gitStatus: contribution.gitStatus ?? '', projectName: contribution.projectName ?? '', currentDate: contribution.currentDate ?? '', ...(contribution.environment ? { environment: contribution。
// 直接调用：createDefaultContextProvider().build。
// 直接调用：createDefaultContextProvider。
// [/maint-note:2a50ccad9ca1]
export async function buildAgentContext(
  input: ContextBuildInput,
): Promise<AgentPromptContext> {
  const contribution = await createDefaultContextProvider().build(input)

  return {
    projectMemory: contribution.projectMemory ?? '',
    gitStatus: contribution.gitStatus ?? '',
    projectName: contribution.projectName ?? '',
    currentDate: contribution.currentDate ?? '',
    ...(contribution.environment ? { environment: contribution.environment } : {}),
    warnings: contribution.warnings ?? [],
    sections: contribution.sections ?? [],
  }
}

// [maint-note:5b23deda6a3a]
// 维护导航：mergeContributions；下列为本函数源码中的接口与路径索引。
// 输入 contributions：readonly ContextContribution[]。
// 返回契约：ContextContribution。
// 返回路径：{ projectMemory: joinNonEmpty(contributions.map((item) => item.projectMemory)), gitStatus: environment ? environment.git.result.kind === 'dirty' ? environment.git.result.status : '' : joinNonEmpty(contributions.map((item) => item.gitStatus)。
// 直接调用：contributions.find。
// 直接调用：joinNonEmpty。
// 直接调用：contributions.map。
// 直接调用：[...contributions].reverse().find。
// 直接调用：[...contributions].reverse。
// 直接调用：contributions.flatMap。
// [/maint-note:5b23deda6a3a]
function mergeContributions(
  contributions: readonly ContextContribution[],
): ContextContribution {
  // The built-in dynamic provider is first and is the sole trusted producer
  // of Session environment facts. Legacy extension strings remain additive
  // but cannot override those facts by reusing currentDate/gitStatus names.
  const environment = contributions.find((item) => item.environment)?.environment
  return {
    projectMemory: joinNonEmpty(contributions.map((item) => item.projectMemory)),
    gitStatus: environment
      ? environment.git.result.kind === 'dirty' ? environment.git.result.status : ''
      : joinNonEmpty(contributions.map((item) => item.gitStatus)),
    projectName: environment?.projectName ?? [...contributions].reverse().find((item) => item.projectName)?.projectName,
    currentDate: environment
      ? `Today's date is ${environment.localDate}.`
      : [...contributions].reverse().find((item) => item.currentDate)?.currentDate,
    ...(environment ? { environment } : {}),
    warnings: contributions.flatMap((item) => item.warnings ?? []),
    sections: contributions.flatMap((item) => item.sections ?? []),
  }
}

// [maint-note:8a8ad38a8e67]
// 维护导航：joinNonEmpty；下列为本函数源码中的接口与路径索引。
// 输入 values：readonly (string | undefined)[]。
// 返回契约：string | undefined。
// 返回路径：nonEmpty.length > 0 ? nonEmpty.join('\n\n') : undefined。
// 直接调用：values.filter。
// 直接调用：nonEmpty.join。
// [/maint-note:8a8ad38a8e67]
function joinNonEmpty(values: readonly (string | undefined)[]): string | undefined {
  const nonEmpty = values.filter((value): value is string => Boolean(value?.trim()))
  return nonEmpty.length > 0 ? nonEmpty.join('\n\n') : undefined
}

// [maint-note:526ee2cfa16d]
// 维护导航：withAbort；下列为本函数源码中的接口与路径索引。
// 输入 promise：Promise<T>。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：Promise<T>。
// 分支条件：!signal。
// 分支条件：signal.aborted。
// 返回路径：promise。
// 返回路径：Promise.reject(createAbortError())。
// 返回路径：new Promise<T>((resolve, reject) => { let settled = false const cleanup = () => signal.removeEventListener('abort', onAbort) const onAbort = () => { if (settled) return settled = true cleanup() reject(createAbortError()) } signal.addEventLi。
// 直接调用：Promise.reject。
// 直接调用：createAbortError。
// [/maint-note:526ee2cfa16d]
function withAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) return Promise.reject(createAbortError())

  return new Promise<T>((resolve, reject) => {
    let settled = false
    // [maint-note:60c2a06c6ffa]
    // 维护导航：cleanup；下列为本函数源码中的接口与路径索引。
    // 直接调用：signal.removeEventListener。
    // [/maint-note:60c2a06c6ffa]
    const cleanup = () => signal.removeEventListener('abort', onAbort)
    // [maint-note:8a28e6e4c306]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 分支条件：settled。
    // 返回路径：无返回值。
    // 写入目标：settled。
    // 直接调用：cleanup。
    // 直接调用：reject。
    // 直接调用：createAbortError。
    // [/maint-note:8a28e6e4c306]
    const onAbort = () => {
      if (settled) return
      settled = true
      cleanup()
      reject(createAbortError())
    }
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => {
        if (settled) return
        settled = true
        cleanup()
        resolve(value)
      },
      (error) => {
        if (settled) return
        settled = true
        cleanup()
        reject(error)
      },
    )
    if (signal.aborted) onAbort()
  })
}

// [maint-note:ad3e32391ba3]
// 维护导航：getAgentContext；下列为本函数源码中的接口与路径索引。
// 输入 input：Omit<ContextBuildInput, 'now'> & { now?: Date }。
// 返回契约：Promise<AgentPromptContext>。
// 返回路径：buildAgentContext({ ...input, now: input.now ?? new Date(), })。
// 直接调用：buildAgentContext。
// [/maint-note:ad3e32391ba3]
export function getAgentContext(
  input: Omit<ContextBuildInput, 'now'> & { now?: Date },
): Promise<AgentPromptContext> {
  return buildAgentContext({
    ...input,
    now: input.now ?? new Date(),
  })
}
