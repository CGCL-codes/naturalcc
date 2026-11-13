// [maint-note:ba09c438e98d]
// 数据契约导航：SelectOption；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 label：string。
// 成员 value：T。
// [/maint-note:ba09c438e98d]
export interface SelectOption<T> {
  readonly id: string
  readonly label: string
  readonly value: T
}

// [maint-note:4ac1c2582f71]
// 维护导航：validateSelectOptions；下列为本函数源码中的接口与路径索引。
// 输入 options：readonly SelectOption<T>[]。
// 返回契约：void。
// 分支条件：options.length === 0。
// 分支条件：!option.id.trim()。
// 分支条件：ids.has(option.id)。
// 遍历边界：const option ← options。
// 显式抛出：new Error('SelectMenu requires at least one option')。
// 显式抛出：new Error('SelectMenu option IDs must be non-empty')。
// 显式抛出：new Error(`SelectMenu option ID must be unique: ${option.id}`)。
// 直接调用：option.id.trim。
// 直接调用：ids.has。
// 直接调用：ids.add。
// [/maint-note:4ac1c2582f71]
export function validateSelectOptions<T>(options: readonly SelectOption<T>[]): void {
  if (options.length === 0) throw new Error('SelectMenu requires at least one option')
  const ids = new Set<string>()
  for (const option of options) {
    if (!option.id.trim()) throw new Error('SelectMenu option IDs must be non-empty')
    if (ids.has(option.id)) throw new Error(`SelectMenu option ID must be unique: ${option.id}`)
    ids.add(option.id)
  }
}

// [maint-note:5025ce3b79ed]
// 数据契约导航：SelectState；字段含义以消费端校验为准。
// 成员 menuId：string。
// 成员 generation：number。
// 成员 options：readonly SelectOption<T>[]。
// 成员 selectedIndex：number。
// 成员 active：boolean。
// [/maint-note:5025ce3b79ed]
export interface SelectState<T> {
  readonly menuId: string
  readonly generation: number
  readonly options: readonly SelectOption<T>[]
  readonly selectedIndex: number
  readonly active: boolean
}

// [maint-note:40716e478fe3]
// 数据契约导航：ConfirmedSelection；字段含义以消费端校验为准。
// 成员 state：SelectState<T>。
// 成员 value：T | undefined。
// [/maint-note:40716e478fe3]
export interface ConfirmedSelection<T> {
  readonly state: SelectState<T>
  readonly value: T | undefined
}

// [maint-note:61c157548177]
// 维护导航：createSelectState；下列为本函数源码中的接口与路径索引。
// 输入 menuId：string。
// 输入 options：readonly SelectOption<T>[]。
// 输入 generation：类型由调用方与函数体约束；缺省表达式 0。
// 输入 selectedIndex：类型由调用方与函数体约束；缺省表达式 0。
// 返回契约：SelectState<T>。
// 分支条件：!menuId.trim()。
// 返回路径：{ menuId, generation, options, selectedIndex: normalizeIndex(selectedIndex, options.length), active: true, }。
// 显式抛出：new Error('SelectMenu menuId must be non-empty')。
// 直接调用：menuId.trim。
// 直接调用：validateSelectOptions。
// 直接调用：normalizeIndex。
// [/maint-note:61c157548177]
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

// [maint-note:52198995bd4e]
// 维护导航：moveSelection；下列为本函数源码中的接口与路径索引。
// 输入 state：SelectState<T>。
// 输入 delta：number。
// 返回契约：SelectState<T>。
// 分支条件：!state.active || state.options.length === 0 || delta === 0。
// 返回路径：state。
// 返回路径：{ ...state, selectedIndex: normalizeIndex(state.selectedIndex + delta, state.options.length), }。
// 直接调用：normalizeIndex。
// [/maint-note:52198995bd4e]
export function moveSelection<T>(state: SelectState<T>, delta: number): SelectState<T> {
  if (!state.active || state.options.length === 0 || delta === 0) return state
  return {
    ...state,
    selectedIndex: normalizeIndex(state.selectedIndex + delta, state.options.length),
  }
}

// [maint-note:6130877299cf]
// 维护导航：confirmSelection；下列为本函数源码中的接口与路径索引。
// 输入 state：SelectState<T>。
// 返回契约：ConfirmedSelection<T>。
// 分支条件：!state.active。
// 返回路径：{ state, value: undefined }。
// 返回路径：{ state: { ...state, active: false }, value: state.options[state.selectedIndex]?.value, }。
// [/maint-note:6130877299cf]
export function confirmSelection<T>(state: SelectState<T>): ConfirmedSelection<T> {
  if (!state.active) return { state, value: undefined }
  return {
    state: { ...state, active: false },
    value: state.options[state.selectedIndex]?.value,
  }
}

// [maint-note:4b17d5722321]
// 维护导航：cancelSelection；下列为本函数源码中的接口与路径索引。
// 输入 state：SelectState<T>。
// 返回契约：SelectState<T>。
// 返回路径：state.active ? { ...state, active: false } : state。
// [/maint-note:4b17d5722321]
export function cancelSelection<T>(state: SelectState<T>): SelectState<T> {
  return state.active ? { ...state, active: false } : state
}

// [maint-note:52012e29d906]
// 维护导航：normalizeIndex；下列为本函数源码中的接口与路径索引。
// 输入 index：number。
// 输入 length：number。
// 返回契约：number。
// 返回路径：((index % length) + length) % length。
// [/maint-note:52012e29d906]
function normalizeIndex(index: number, length: number): number {
  return ((index % length) + length) % length
}
