import { createHash } from 'node:crypto'

/**
 * Bounded content-derived identity used only by trusted framework receipts.
 * Raw provider IDs remain the pairing keys in ModelMessage history.
 */
// [maint-note:e83e8fddb74b]
// 维护导航：canonicalToolCallRef；下列为本函数源码中的接口与路径索引。
// 输入 callId：string。
// 返回契约：string。
// 返回路径：`sha256:${createHash('sha256').update(JSON.stringify(callId)).digest('hex')}`。
// 直接调用：createHash('sha256').update(JSON.stringify(callId)).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// [/maint-note:e83e8fddb74b]
export function canonicalToolCallRef(callId: string): string {
  return `sha256:${createHash('sha256').update(JSON.stringify(callId)).digest('hex')}`
}
