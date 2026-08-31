const THEME_STORAGE_KEY = "code-agent.theme";
const THEMES = new Set(["light", "dark"]);

export const defaultTheme = "dark";

export function normalizeTheme(value) {
  return THEMES.has(value) ? value : defaultTheme;
}

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

export function nextTheme(current) {
  return normalizeTheme(current) === "dark" ? "light" : "dark";
}
