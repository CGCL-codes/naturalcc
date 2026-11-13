// [maint-note:da1529f3c108]
// 数据契约导航：ReplCtrlCAction；字段含义以消费端校验为准。
// 类型组合：| 'interruptTurn' | 'dismissSelection' | 'clearAndArmExit' | 'armExit' | 'exit' | 'ignore'。
// [/maint-note:da1529f3c108]
export type ReplCtrlCAction =
  | 'interruptTurn'
  | 'dismissSelection'
  | 'clearAndArmExit'
  | 'armExit'
  | 'exit'
  | 'ignore'

// [maint-note:a7b32f8f657b]
// 数据契约导航：ReplCtrlCSnapshot；字段含义以消费端校验为准。
// 成员 approvalPending：boolean。
// 成员 selectionOpen：boolean。
// 成员 loading：boolean。
// 成员 clearing：boolean。
// 成员 inputLength：number。
// 成员 exitArmed：boolean。
// [/maint-note:a7b32f8f657b]
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
// [maint-note:b67674210b23]
// 维护导航：decideReplCtrlC；下列为本函数源码中的接口与路径索引。
// 输入 snapshot：ReplCtrlCSnapshot。
// 返回契约：ReplCtrlCAction。
// 分支条件：snapshot.approvalPending。
// 分支条件：snapshot.selectionOpen。
// 分支条件：snapshot.loading。
// 分支条件：snapshot.clearing。
// 分支条件：snapshot.inputLength > 0。
// 返回路径：'interruptTurn'。
// 返回路径：'dismissSelection'。
// 返回路径：'ignore'。
// 返回路径：'clearAndArmExit'。
// 返回路径：snapshot.exitArmed ? 'exit' : 'armExit'。
// [/maint-note:b67674210b23]
export function decideReplCtrlC(snapshot: ReplCtrlCSnapshot): ReplCtrlCAction {
  if (snapshot.approvalPending) return 'interruptTurn'
  if (snapshot.selectionOpen) return 'dismissSelection'
  if (snapshot.loading) return 'interruptTurn'
  if (snapshot.clearing) return 'ignore'
  if (snapshot.inputLength > 0) return 'clearAndArmExit'
  return snapshot.exitArmed ? 'exit' : 'armExit'
}
