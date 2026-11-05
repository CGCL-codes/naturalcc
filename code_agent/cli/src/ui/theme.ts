/** A deliberately small, process-fixed semantic palette. */
export const UI_PALETTE = Object.freeze({
  boundary: 'white',
  neutral: 'gray',
  model: 'yellow',
  workspace: 'green',
  danger: 'red',
  markdownAccent: '#9BB7C9',
  markdownStrong: '#C99595',
  /**
   * one shared "user input surface" for BOTH the editable input line
   * and committed user messages — dark-gray background with a light paired
   * foreground so the surface stays readable even on light terminal themes.
   * Fixed first-version values; no theme switcher, env toggle, or background
   * detection (OSC queries are explicitly out of scope).
   */
  userSurfaceBackground: '#262626',
  userSurfaceForeground: '#D8DEE9',
} as const)

export type UiPaletteColor = typeof UI_PALETTE[keyof typeof UI_PALETTE]
