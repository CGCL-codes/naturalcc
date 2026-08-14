import type { ModelClient } from './types/types.js'
import { OpenAIChatModelClient } from './openaiChatClient.js'

export interface CreateModelClientOptions {
  apiKey?: string | null
  baseURL?: string | null
}

export function createModelClient(
  options: CreateModelClientOptions = {},
): ModelClient {
  return new OpenAIChatModelClient({
    apiKey: resolveApiKey(options.apiKey),
    baseURL: resolveBaseURL(options.baseURL),
  })
}

function resolveApiKey(apiKey?: string | null): string {
  const resolved = normalizeOptionalString(apiKey ?? process.env.OPENAI_API_KEY)
  if (!resolved) {
    throw new Error('OPENAI_API_KEY is required to create an OpenAI model client')
  }
  return resolved
}

function resolveBaseURL(baseURL?: string | null): string | undefined {
  return normalizeOptionalString(baseURL ?? process.env.OPENAI_BASE_URL)
}

function normalizeOptionalString(value?: string | null): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}
