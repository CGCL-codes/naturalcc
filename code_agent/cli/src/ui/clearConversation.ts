import type { SessionManager } from '../app/session/SessionManager.js'

/** Replace the agent history first; callers can then clear their UI history. */
export async function clearConversation(
  sessionManager: SessionManager,
  clearUiHistory: () => void,
): Promise<void> {
  await sessionManager.clear()
  clearUiHistory()
}
