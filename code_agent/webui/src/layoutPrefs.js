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

export function clampLayoutValue(value, bounds) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return bounds.fallback;
  }
  return Math.min(bounds.max, Math.max(bounds.min, Math.round(numeric)));
}

export function normalizeLayoutPrefs(value = {}) {
  return {
    sidebarWidth: clampLayoutValue(value.sidebarWidth, layoutBounds.sidebarWidth),
    agentStackHeight: clampLayoutValue(value.agentStackHeight, layoutBounds.agentStackHeight),
    drawerWidth: clampLayoutValue(value.drawerWidth, layoutBounds.drawerWidth)
  };
}

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

export function toLayoutStyle(value) {
  const normalized = normalizeLayoutPrefs(value);
  return {
    "--sidebar-width": `${normalized.sidebarWidth}px`,
    "--agent-stack-height": `${normalized.agentStackHeight}px`,
    "--drawer-width": `${normalized.drawerWidth}px`
  };
}
