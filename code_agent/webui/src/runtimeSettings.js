const ACTIVE_THREAD_KEY = "code-agent.active-thread";

// [maint-note:f5c93dd1c9bb]
// 维护导航：readActiveThreadId；下列为本函数源码中的接口与路径索引。
// 输入 storage：类型由调用方与函数体约束；缺省表达式 globalThis.localStorage。
// 返回路径：storage?.getItem(ACTIVE_THREAD_KEY) || null。
// 返回路径：null。
// 直接调用：storage?.getItem。
// [/maint-note:f5c93dd1c9bb]
export function readActiveThreadId(storage = globalThis.localStorage) {
  try {
    return storage?.getItem(ACTIVE_THREAD_KEY) || null;
  } catch {
    return null;
  }
}

// [maint-note:b1e95df86222]
// 维护导航：rememberActiveThread；下列为本函数源码中的接口与路径索引。
// 输入 id：类型由调用方与函数体约束。
// 输入 storage：类型由调用方与函数体约束；缺省表达式 globalThis.localStorage。
// 分支条件：id。
// 直接调用：storage?.setItem。
// 直接调用：storage?.removeItem。
// [/maint-note:b1e95df86222]
export function rememberActiveThread(id, storage = globalThis.localStorage) {
  try {
    if (id) storage?.setItem(ACTIVE_THREAD_KEY, id);
    else storage?.removeItem(ACTIVE_THREAD_KEY);
  } catch {
    // Storage may be disabled; the current conversation remains usable.
  }
}

// [maint-note:19eb5e6d8314]
// 维护导航：threadToRestore；下列为本函数源码中的接口与路径索引。
// 输入 threads：类型由调用方与函数体约束。
// 输入 savedId：类型由调用方与函数体约束。
// 返回路径：threads.find((thread) => thread.id === savedId)?.id || null。
// 直接调用：threads.find。
// [/maint-note:19eb5e6d8314]
export function threadToRestore(threads, savedId) {
  return threads.find((thread) => thread.id === savedId)?.id || null;
}

// [maint-note:a80d2665378c]
// 维护导航：runtimeConfigForEdit；下列为本函数源码中的接口与路径索引。
// 输入 base：类型由调用方与函数体约束。
// 输入 provider：类型由调用方与函数体约束。
// 输入 model：类型由调用方与函数体约束。
// 返回路径：{ ...(base?.provider === provider ? base : {}), provider, model }。
// [/maint-note:a80d2665378c]
export function runtimeConfigForEdit(base, provider, model) {
  // Preserve endpoint and routing settings only within the same provider.
  return { ...(base?.provider === provider ? base : {}), provider, model };
}
