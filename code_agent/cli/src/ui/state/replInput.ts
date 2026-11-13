// [maint-note:8be9a0759a84]
// 数据契约导航：ReplInputChunkAction；字段含义以消费端校验为准。
// 类型组合：| { type: 'text'; content: string } | { type: 'ctrlC' }。
// [/maint-note:8be9a0759a84]
export type ReplInputChunkAction =
  | { type: 'text'; content: string }
  | { type: 'ctrlC' }

/**
 * Reduce one Ink input boundary without losing the order of text and C0
 * control bytes. Ink normally presents one key at a time, but PTY writes can
 * contain several logical inputs in one callback.
 */
// [maint-note:c9b26a3af40d]
// 维护导航：reduceReplInputChunk；下列为本函数源码中的接口与路径索引。
// 输入 input：string。
// 输入 ctrlKey：类型由调用方与函数体约束；缺省表达式 false。
// 返回契约：readonly ReplInputChunkAction[]。
// 分支条件：ctrlKey && input === 'c'。
// 分支条件：!input.includes('\x03')。
// 分支条件：input[index] !== '\x03'。
// 分支条件：index > textStart。
// 分支条件：textStart < input.length。
// 遍历边界：let index = 0; index < input.length; index += 1。
// 返回路径：[{ type: 'ctrlC' }]。
// 返回路径：input ? [{ type: 'text', content: input }] : []。
// 返回路径：actions。
// 写入目标：index。
// 写入目标：textStart。
// 直接调用：input.includes。
// 直接调用：actions.push。
// 直接调用：input.slice。
// [/maint-note:c9b26a3af40d]
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
