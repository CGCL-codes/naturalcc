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

export function composeContextProviders(
  providers: readonly ContextProvider[],
): ContextProvider {
  return {
    name: providers.map((provider) => provider.name).join('+') || 'empty-context',
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

function joinNonEmpty(values: readonly (string | undefined)[]): string | undefined {
  const nonEmpty = values.filter((value): value is string => Boolean(value?.trim()))
  return nonEmpty.length > 0 ? nonEmpty.join('\n\n') : undefined
}

function withAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) return Promise.reject(createAbortError())

  return new Promise<T>((resolve, reject) => {
    let settled = false
    const cleanup = () => signal.removeEventListener('abort', onAbort)
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

export function getAgentContext(
  input: Omit<ContextBuildInput, 'now'> & { now?: Date },
): Promise<AgentPromptContext> {
  return buildAgentContext({
    ...input,
    now: input.now ?? new Date(),
  })
}
