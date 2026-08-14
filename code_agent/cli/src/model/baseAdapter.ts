import type {
  ModelAdapter,
  ModelRequest,
  ModelResponse,
} from './types/types.js'

export abstract class BaseModelAdapter<ProviderRequest, ProviderResponse>
  implements ModelAdapter<ProviderRequest, ProviderResponse> {
  abstract toProviderRequest(request: ModelRequest): ProviderRequest
  abstract fromProviderResponse(response: ProviderResponse): ModelResponse

  protected parseJsonObject(raw: string): {
    value: Record<string, unknown>
    error?: string
  } {
    try {
      const parsed = JSON.parse(raw || '{}')
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return { value: parsed as Record<string, unknown> }
      }
      return { value: {}, error: 'Tool arguments must be a JSON object.' }
    } catch {
      return { value: {}, error: 'Tool arguments must be valid JSON.' }
    }
  }
}
