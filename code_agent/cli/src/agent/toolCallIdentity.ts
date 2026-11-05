import { createHash } from 'node:crypto'

/**
 * Bounded content-derived identity used only by trusted framework receipts.
 * Raw provider IDs remain the pairing keys in ModelMessage history.
 */
export function canonicalToolCallRef(callId: string): string {
  return `sha256:${createHash('sha256').update(JSON.stringify(callId)).digest('hex')}`
}
