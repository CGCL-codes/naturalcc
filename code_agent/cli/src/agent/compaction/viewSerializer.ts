import type { ModelMessage, ModelSystemMessage } from '../../model/types/types.js'
import type { FinalRequestProjection } from '../types/frameworkContext.js'

/**
 * common final serializer. The maintenance
 * net-gain measurement and the normal Agent request share ONE serialized
 * representation: request messages exactly as they travel (JSON escaping
 * included) plus the request-only section envelope appended to the system
 * message. Neither side may substitute a constant for the real projection.
 */

/**
 * Append the request-only sections to a request's messages. This is the ONE
 * builder for the normal dispatch (AgentRunner) — the measurement below
 * serializes the same shape so the numbers are the outbound truth.
 */
// [maint-note:f8e967e342c9]
// 维护导航：appendRequestOnlySections；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 输入 sections：readonly string[]。
// 输入 trailingFrameworkMessages：readonly ModelSystemMessage[]；缺省表达式 []。
// 返回契约：ModelMessage[]。
// 分支条件：sections.length > 0。
// 分支条件：system。
// 返回路径：cloned。
// 直接调用：structuredClone。
// 直接调用：cloned.find。
// 直接调用：system.content.push。
// 直接调用：sections.map。
// 直接调用：cloned.unshift。
// 直接调用：cloned.push。
// [/maint-note:f8e967e342c9]
export function appendRequestOnlySections(
  messages: readonly ModelMessage[],
  sections: readonly string[],
  trailingFrameworkMessages: readonly ModelSystemMessage[] = [],
): ModelMessage[] {
  const cloned = structuredClone([...messages])
  if (sections.length > 0) {
    const system = cloned.find((message) => message.role === 'system')
    if (system) {
      system.content.push(...sections.map((text) => ({ type: 'text' as const, text: `\n\n[request-only]\n${text}` })))
    } else {
      cloned.unshift({
        role: 'system',
        content: sections.map((text) => ({ type: 'text' as const, text: `[request-only]\n${text}` })),
      })
    }
  }
  cloned.push(...structuredClone([...trailingFrameworkMessages]))
  return cloned
}

// [maint-note:b58835a08f53]
// 维护导航：buildFinalRequestMessages；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 输入 projection：FinalRequestProjection。
// 返回契约：ModelMessage[]。
// 返回路径：appendRequestOnlySections( messages, projection.requestOnlySections, projection.trailingFrameworkMessages, )。
// 直接调用：appendRequestOnlySections。
// [/maint-note:b58835a08f53]
export function buildFinalRequestMessages(
  messages: readonly ModelMessage[],
  projection: FinalRequestProjection,
): ModelMessage[] {
  return appendRequestOnlySections(
    messages,
    projection.requestOnlySections,
    projection.trailingFrameworkMessages,
  )
}

/** Serialized cost of the request-only envelope for the given sections. */
// [maint-note:c519fd26ee82]
// 维护导航：requestOnlyEnvelopeChars；下列为本函数源码中的接口与路径索引。
// 输入 sections：readonly string[]。
// 返回契约：number。
// 分支条件：sections.length === 0。
// 返回路径：0。
// 返回路径：JSON.stringify( sections.map((text) => ({ type: 'text', text: `\n\n[request-only]\n${text}` })), ).length。
// 直接调用：JSON.stringify。
// 直接调用：sections.map。
// [/maint-note:c519fd26ee82]
export function requestOnlyEnvelopeChars(sections: readonly string[]): number {
  if (sections.length === 0) return 0
  return JSON.stringify(
    sections.map((text) => ({ type: 'text', text: `\n\n[request-only]\n${text}` })),
  ).length
}

/**
 * The final serialized size of a request view: its messages plus the
 * request-only envelope, measured exactly as the request serializes.
 */
// [maint-note:9221af1635b0]
// 维护导航：serializedRequestChars；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 输入 sections：readonly string[]。
// 输入 trailingFrameworkMessages：readonly ModelSystemMessage[]；缺省表达式 []。
// 返回契约：number。
// 返回路径：JSON.stringify(appendRequestOnlySections( messages, sections, trailingFrameworkMessages, )).length。
// 直接调用：JSON.stringify。
// 直接调用：appendRequestOnlySections。
// [/maint-note:9221af1635b0]
export function serializedRequestChars(
  messages: readonly ModelMessage[],
  sections: readonly string[],
  trailingFrameworkMessages: readonly ModelSystemMessage[] = [],
): number {
  return JSON.stringify(appendRequestOnlySections(
    messages,
    sections,
    trailingFrameworkMessages,
  )).length
}

// [maint-note:24c017345fd5]
// 维护导航：serializedFinalRequestChars；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 输入 projection：FinalRequestProjection。
// 返回契约：number。
// 返回路径：JSON.stringify(buildFinalRequestMessages(messages, projection)).length。
// 直接调用：JSON.stringify。
// 直接调用：buildFinalRequestMessages。
// [/maint-note:24c017345fd5]
export function serializedFinalRequestChars(
  messages: readonly ModelMessage[],
  projection: FinalRequestProjection,
): number {
  return JSON.stringify(buildFinalRequestMessages(messages, projection)).length
}

/**
 * The shared omission notice (77.4): the SAME text rides the real dispatch
 * and the request-only peek, so the measurement accounts it once and
 * identically on both sides of the net-gain comparison.
 */
// [maint-note:80a815b6c330]
// 维护导航：omissionRequestNotice；下列为本函数源码中的接口与路径索引。
// 输入 omittedGroupCount：number。
// 输入 omittedCharsApprox：number。
// 返回契约：string。
// 返回路径：`Earlier conversation history was omitted from this request (${omittedGroupCount} exchange(s), approximately ${omittedCharsApprox} characters). Do not assume omitted work never happened; rely on the visible verified evidence.`。
// [/maint-note:80a815b6c330]
export function omissionRequestNotice(
  omittedGroupCount: number,
  omittedCharsApprox: number,
): string {
  return `Earlier conversation history was omitted from this request (${omittedGroupCount} exchange(s), approximately ${omittedCharsApprox} characters). Do not assume omitted work never happened; rely on the visible verified evidence.`
}
