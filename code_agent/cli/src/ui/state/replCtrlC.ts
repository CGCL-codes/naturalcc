export type ReplCtrlCAction =
  | 'interruptTurn'
  | 'dismissSelection'
  | 'clearAndArmExit'
  | 'armExit'
  | 'exit'
  | 'ignore'

export interface ReplCtrlCSnapshot {
  approvalPending: boolean
  selectionOpen: boolean
  loading: boolean
  clearing: boolean
  inputLength: number
  exitArmed: boolean
}

/**
 * Decide the one action for an interactive Ctrl-C event.
 *
 * This is deliberately independent of Ink listener order. The Repl owns the
 * decision; input widgets only edit text and render their local completion UI.
 */
export function decideReplCtrlC(snapshot: ReplCtrlCSnapshot): ReplCtrlCAction {
  if (snapshot.approvalPending) return 'interruptTurn'
  if (snapshot.selectionOpen) return 'dismissSelection'
  if (snapshot.loading) return 'interruptTurn'
  if (snapshot.clearing) return 'ignore'
  if (snapshot.inputLength > 0) return 'clearAndArmExit'
  return snapshot.exitArmed ? 'exit' : 'armExit'
}
