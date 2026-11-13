import type {
  ModelAdapter,
  ModelRequest,
  ModelResponse,
} from './types/types.js'

export abstract class BaseModelAdapter<ProviderRequest, ProviderResponse>
  implements ModelAdapter<ProviderRequest, ProviderResponse> {
  abstract toProviderRequest(request: ModelRequest): ProviderRequest
  abstract fromProviderResponse(response: ProviderResponse): ModelResponse

  // [maint-note:15eda2e2ebc2]
  // 维护导航：parseJsonObject；下列为本函数源码中的接口与路径索引。
  // 输入 raw：string。
  // 返回契约：{ value: Record<string, unknown> error?: string }。
  // 分支条件：parsed && typeof parsed === 'object' && !Array.isArray(parsed)。
  // 返回路径：{ value: parsed as Record<string, unknown> }。
  // 返回路径：{ value: {}, error: 'Tool arguments must be a JSON object.' }。
  // 返回路径：{ value: {}, error: 'Tool arguments must be valid JSON.' }。
  // 直接调用：JSON.parse。
  // 直接调用：Array.isArray。
  // [/maint-note:15eda2e2ebc2]
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
