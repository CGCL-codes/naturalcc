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
export function omissionRequestNotice(
  omittedGroupCount: number,
  omittedCharsApprox: number,
): string {
  return `Earlier conversation history was omitted from this request (${omittedGroupCount} exchange(s), approximately ${omittedCharsApprox} characters). Do not assume omitted work never happened; rely on the visible verified evidence.`
}
