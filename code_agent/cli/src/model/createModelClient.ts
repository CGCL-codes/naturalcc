import type { ModelBudgetCapabilities, ModelClient } from './types/types.js'
import { OpenAIChatModelClient } from './openaiChatClient.js'

// [maint-note:7d69bc4d827d]
// 数据契约导航：CreateModelClientOptions；字段含义以消费端校验为准。
// 成员 apiKey（可省略）：string | null。
// 成员 baseURL（可省略）：string | null。
// 成员 requestBudgetCapability（可省略）：ModelBudgetCapabilities & { readonly countInputTokens?: (body: Readonly<Record<string, unknown>>) => number }。
// [/maint-note:7d69bc4d827d]
export interface CreateModelClientOptions {
  apiKey?: string | null
  baseURL?: string | null
  requestBudgetCapability?: ModelBudgetCapabilities & {
    readonly countInputTokens?: (body: Readonly<Record<string, unknown>>) => number
  }
}

// [maint-note:6b118ce3a5b3]
// 维护导航：createModelClient；下列为本函数源码中的接口与路径索引。
// 输入 options：CreateModelClientOptions；缺省表达式 {}。
// 返回契约：ModelClient。
// 返回路径：new OpenAIChatModelClient({ apiKey: resolveApiKey(options.apiKey), baseURL: resolveBaseURL(options.baseURL), requestBudgetCapability: options.requestBudgetCapability, })。
// 直接调用：resolveApiKey。
// 直接调用：resolveBaseURL。
// [/maint-note:6b118ce3a5b3]
export function createModelClient(
  options: CreateModelClientOptions = {},
): ModelClient {
  return new OpenAIChatModelClient({
    apiKey: resolveApiKey(options.apiKey),
    baseURL: resolveBaseURL(options.baseURL),
    requestBudgetCapability: options.requestBudgetCapability,
  })
}

// [maint-note:cbcd84b5e9f4]
// 维护导航：resolveApiKey；下列为本函数源码中的接口与路径索引。
// 输入 apiKey：string | null（可省略）。
// 返回契约：string。
// 分支条件：!resolved。
// 返回路径：resolved。
// 显式抛出：new Error('OPENAI_API_KEY is required to create an OpenAI model client')。
// 直接调用：normalizeOptionalString。
// [/maint-note:cbcd84b5e9f4]
function resolveApiKey(apiKey?: string | null): string {
  const resolved = normalizeOptionalString(apiKey ?? process.env.OPENAI_API_KEY)
  if (!resolved) {
    throw new Error('OPENAI_API_KEY is required to create an OpenAI model client')
  }
  return resolved
}

// [maint-note:41ff0dad467f]
// 维护导航：resolveBaseURL；下列为本函数源码中的接口与路径索引。
// 输入 baseURL：string | null（可省略）。
// 返回契约：string | undefined。
// 返回路径：normalizeOptionalString(baseURL ?? process.env.OPENAI_BASE_URL)。
// 直接调用：normalizeOptionalString。
// [/maint-note:41ff0dad467f]
function resolveBaseURL(baseURL?: string | null): string | undefined {
  return normalizeOptionalString(baseURL ?? process.env.OPENAI_BASE_URL)
}

// [maint-note:e8bcf5db3a12]
// 维护导航：normalizeOptionalString；下列为本函数源码中的接口与路径索引。
// 输入 value：string | null（可省略）。
// 返回契约：string | undefined。
// 返回路径：trimmed ? trimmed : undefined。
// 直接调用：value?.trim。
// [/maint-note:e8bcf5db3a12]
function normalizeOptionalString(value?: string | null): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}
