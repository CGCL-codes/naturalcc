export interface SelectOption<T> {
  readonly id: string
  readonly label: string
  readonly value: T
}

export function validateSelectOptions<T>(options: readonly SelectOption<T>[]): void {
  if (options.length === 0) throw new Error('SelectMenu requires at least one option')
  const ids = new Set<string>()
  for (const option of options) {
    if (!option.id.trim()) throw new Error('SelectMenu option IDs must be non-empty')
    if (ids.has(option.id)) throw new Error(`SelectMenu option ID must be unique: ${option.id}`)
    ids.add(option.id)
  }
}

export interface SelectState<T> {
  readonly menuId: string
  readonly generation: number
  readonly options: readonly SelectOption<T>[]
  readonly selectedIndex: number
  readonly active: boolean
}

export interface ConfirmedSelection<T> {
  readonly state: SelectState<T>
  readonly value: T | undefined
}

export function createSelectState<T>(
  menuId: string,
  options: readonly SelectOption<T>[],
  generation = 0,
  selectedIndex = 0,
): SelectState<T> {
  if (!menuId.trim()) throw new Error('SelectMenu menuId must be non-empty')
  validateSelectOptions(options)
  return {
    menuId,
    generation,
    options,
    selectedIndex: normalizeIndex(selectedIndex, options.length),
    active: true,
  }
}

export function moveSelection<T>(state: SelectState<T>, delta: number): SelectState<T> {
  if (!state.active || state.options.length === 0 || delta === 0) return state
  return {
    ...state,
    selectedIndex: normalizeIndex(state.selectedIndex + delta, state.options.length),
  }
}

export function confirmSelection<T>(state: SelectState<T>): ConfirmedSelection<T> {
  if (!state.active) return { state, value: undefined }
  return {
    state: { ...state, active: false },
    value: state.options[state.selectedIndex]?.value,
  }
}

export function cancelSelection<T>(state: SelectState<T>): SelectState<T> {
  return state.active ? { ...state, active: false } : state
}

function normalizeIndex(index: number, length: number): number {
  return ((index % length) + length) % length
}
