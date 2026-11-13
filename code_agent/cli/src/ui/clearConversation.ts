import type { SessionManager } from '../app/session/SessionManager.js'

/** Replace the agent history first; callers can then clear their UI history. */
// [maint-note:b3983b452edc]
// 维护导航：clearConversation；下列为本函数源码中的接口与路径索引。
// 输入 sessionManager：SessionManager。
// 输入 clearUiHistory：() => void。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 直接调用：sessionManager.clear。
// 直接调用：clearUiHistory。
// [/maint-note:b3983b452edc]
export async function clearConversation(
  sessionManager: SessionManager,
  clearUiHistory: () => void,
): Promise<void> {
  await sessionManager.clear()
  clearUiHistory()
}
