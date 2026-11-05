export type ReplInputChunkAction =
  | { type: 'text'; content: string }
  | { type: 'ctrlC' }

/**
 * Reduce one Ink input boundary without losing the order of text and C0
 * control bytes. Ink normally presents one key at a time, but PTY writes can
 * contain several logical inputs in one callback.
 */
export function reduceReplInputChunk(input: string, ctrlKey = false): readonly ReplInputChunkAction[] {
  if (ctrlKey && input === 'c') return [{ type: 'ctrlC' }]
  if (!input.includes('\x03')) {
    return input ? [{ type: 'text', content: input }] : []
  }

  const actions: ReplInputChunkAction[] = []
  let textStart = 0
  for (let index = 0; index < input.length; index += 1) {
    if (input[index] !== '\x03') continue
    if (index > textStart) actions.push({ type: 'text', content: input.slice(textStart, index) })
    actions.push({ type: 'ctrlC' })
    textStart = index + 1
  }
  if (textStart < input.length) actions.push({ type: 'text', content: input.slice(textStart) })
  return actions
}
