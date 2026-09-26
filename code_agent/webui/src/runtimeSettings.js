const ACTIVE_THREAD_KEY = "code-agent.active-thread";

export function readActiveThreadId(storage = globalThis.localStorage) {
  try {
    return storage?.getItem(ACTIVE_THREAD_KEY) || null;
  } catch {
    return null;
  }
}

export function rememberActiveThread(id, storage = globalThis.localStorage) {
  try {
    if (id) storage?.setItem(ACTIVE_THREAD_KEY, id);
    else storage?.removeItem(ACTIVE_THREAD_KEY);
  } catch {
    // Storage may be disabled; the current conversation remains usable.
  }
}

export function threadToRestore(threads, savedId) {
  return threads.find((thread) => thread.id === savedId)?.id || null;
}

export function runtimeConfigForEdit(base, provider, model) {
  // Preserve endpoint and routing settings only within the same provider.
  return { ...(base?.provider === provider ? base : {}), provider, model };
}
