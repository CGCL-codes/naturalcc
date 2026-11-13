const THEME_STORAGE_KEY = "code-agent.theme";
const THEMES = new Set(["light", "dark"]);

export const defaultTheme = "dark";

// [maint-note:cbffe182517d]
// 维护导航：normalizeTheme；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束。
// 返回路径：THEMES.has(value) ? value : defaultTheme。
// 直接调用：THEMES.has。
// [/maint-note:cbffe182517d]
export function normalizeTheme(value) {
  return THEMES.has(value) ? value : defaultTheme;
}

// [maint-note:480ea15da696]
// 维护导航：loadThemePref；下列为本函数源码中的接口与路径索引。
// 输入 storage：类型由调用方与函数体约束；缺省表达式 globalThis.localStorage。
// 分支条件：!storage。
// 返回路径：defaultTheme。
// 返回路径：normalizeTheme(storage.getItem(THEME_STORAGE_KEY))。
// 直接调用：normalizeTheme。
// 直接调用：storage.getItem。
// [/maint-note:480ea15da696]
export function loadThemePref(storage = globalThis.localStorage) {
  if (!storage) {
    return defaultTheme;
  }
  try {
    return normalizeTheme(storage.getItem(THEME_STORAGE_KEY));
  } catch {
    return defaultTheme;
  }
}

// [maint-note:b7c860bf7293]
// 维护导航：saveThemePref；下列为本函数源码中的接口与路径索引。
// 输入 storage：类型由调用方与函数体约束；缺省表达式 globalThis.localStorage。
// 输入 value：类型由调用方与函数体约束。
// 分支条件：storage。
// 返回路径：theme。
// 直接调用：normalizeTheme。
// 直接调用：storage.setItem。
// [/maint-note:b7c860bf7293]
export function saveThemePref(storage = globalThis.localStorage, value) {
  const theme = normalizeTheme(value);
  if (storage) {
    try {
      storage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Keep the in-memory theme usable when storage is unavailable.
    }
  }
  return theme;
}

// [maint-note:7a063a4950f9]
// 维护导航：nextTheme；下列为本函数源码中的接口与路径索引。
// 输入 current：类型由调用方与函数体约束。
// 返回路径：normalizeTheme(current) === "dark" ? "light" : "dark"。
// 直接调用：normalizeTheme。
// [/maint-note:7a063a4950f9]
export function nextTheme(current) {
  return normalizeTheme(current) === "dark" ? "light" : "dark";
}
