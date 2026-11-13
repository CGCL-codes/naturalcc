const STORAGE_KEY = "code-agent.layout";

export const layoutBounds = {
  sidebarWidth: { min: 220, max: 520, fallback: 260 },
  agentStackHeight: { min: 140, max: 520, fallback: 320 },
  drawerWidth: { min: 280, max: 620, fallback: 320 }
};

export const defaultLayoutPrefs = {
  sidebarWidth: layoutBounds.sidebarWidth.fallback,
  agentStackHeight: layoutBounds.agentStackHeight.fallback,
  drawerWidth: layoutBounds.drawerWidth.fallback
};

// [maint-note:58f9e3b26fa5]
// 维护导航：clampLayoutValue；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束。
// 输入 bounds：类型由调用方与函数体约束。
// 分支条件：!Number.isFinite(numeric)。
// 返回路径：bounds.fallback。
// 返回路径：Math.min(bounds.max, Math.max(bounds.min, Math.round(numeric)))。
// 直接调用：Number。
// 直接调用：Number.isFinite。
// 直接调用：Math.min。
// 直接调用：Math.max。
// 直接调用：Math.round。
// [/maint-note:58f9e3b26fa5]
export function clampLayoutValue(value, bounds) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return bounds.fallback;
  }
  return Math.min(bounds.max, Math.max(bounds.min, Math.round(numeric)));
}

// [maint-note:0626168eec81]
// 维护导航：normalizeLayoutPrefs；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束；缺省表达式 {}。
// 返回路径：{ sidebarWidth: clampLayoutValue(value.sidebarWidth, layoutBounds.sidebarWidth), agentStackHeight: clampLayoutValue(value.agentStackHeight, layoutBounds.agentStackHeight), drawerWidth: clampLayoutValue(value.drawerWidth, layoutBounds.drawer。
// 直接调用：clampLayoutValue。
// [/maint-note:0626168eec81]
export function normalizeLayoutPrefs(value = {}) {
  return {
    sidebarWidth: clampLayoutValue(value.sidebarWidth, layoutBounds.sidebarWidth),
    agentStackHeight: clampLayoutValue(value.agentStackHeight, layoutBounds.agentStackHeight),
    drawerWidth: clampLayoutValue(value.drawerWidth, layoutBounds.drawerWidth)
  };
}

// [maint-note:7fa8a2d3a267]
// 维护导航：loadLayoutPrefs；下列为本函数源码中的接口与路径索引。
// 输入 storage：类型由调用方与函数体约束；缺省表达式 globalThis.localStorage。
// 分支条件：!storage。
// 分支条件：!raw。
// 返回路径：{ ...defaultLayoutPrefs }。
// 返回路径：normalizeLayoutPrefs(JSON.parse(raw))。
// 直接调用：storage.getItem。
// 直接调用：normalizeLayoutPrefs。
// 直接调用：JSON.parse。
// [/maint-note:7fa8a2d3a267]
export function loadLayoutPrefs(storage = globalThis.localStorage) {
  if (!storage) {
    return { ...defaultLayoutPrefs };
  }
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) {
      return { ...defaultLayoutPrefs };
    }
    return normalizeLayoutPrefs(JSON.parse(raw));
  } catch {
    return { ...defaultLayoutPrefs };
  }
}

// [maint-note:4d47e12b29e9]
// 维护导航：saveLayoutPrefs；下列为本函数源码中的接口与路径索引。
// 输入 storage：类型由调用方与函数体约束；缺省表达式 globalThis.localStorage。
// 输入 value：类型由调用方与函数体约束。
// 分支条件：storage。
// 返回路径：normalized。
// 直接调用：normalizeLayoutPrefs。
// 直接调用：storage.setItem。
// 直接调用：JSON.stringify。
// [/maint-note:4d47e12b29e9]
export function saveLayoutPrefs(storage = globalThis.localStorage, value) {
  const normalized = normalizeLayoutPrefs(value);
  if (storage) {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(normalized));
    } catch {
      // Ignore storage failures; the in-memory layout state remains usable.
    }
  }
  return normalized;
}

// [maint-note:555879ecd679]
// 维护导航：toLayoutStyle；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束。
// 返回路径：{ "--sidebar-width": `${normalized.sidebarWidth}px`, "--agent-stack-height": `${normalized.agentStackHeight}px`, "--drawer-width": `${normalized.drawerWidth}px` }。
// 直接调用：normalizeLayoutPrefs。
// [/maint-note:555879ecd679]
export function toLayoutStyle(value) {
  const normalized = normalizeLayoutPrefs(value);
  return {
    "--sidebar-width": `${normalized.sidebarWidth}px`,
    "--agent-stack-height": `${normalized.agentStackHeight}px`,
    "--drawer-width": `${normalized.drawerWidth}px`
  };
}
