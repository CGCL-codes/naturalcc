import { useEffect, useLayoutEffect, useRef } from 'react'
import { Box, Text } from 'ink'
import {
  cancelSelection,
  confirmSelection,
  createSelectState,
  moveSelection,
  validateSelectOptions,
  type SelectOption,
} from '../state/selectMenu.js'
import { calculateContentWidth, sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'
import { TerminalSectionStack, type TerminalSectionData } from './TerminalSection.js'
import type { KeyboardKey, KeyboardInputHandler } from '../state/keyboardInput.js'

// [maint-note:90248142bee1]
// 数据契约导航：SelectMenuProps；字段含义以消费端校验为准。
// 成员 menuId：string。
// 成员 generation（可省略）：number。
// 成员 title：string。
// 成员 body（可省略）：string。
// 成员 sections（可省略）：readonly TerminalSectionData[]。
// 成员 options：readonly SelectOption<T>[]。
// 成员 selected：T。
// 成员 active：boolean。
// 成员 onChange：(value: T) => void。
// 成员 onConfirm：(value: T) => void。
// 成员 onCancel：() => void。
// 成员 width（可省略）：number。
// 成员 onRegisterKeyboard（可省略）：(handler: KeyboardInputHandler | null) => void。
// [/maint-note:90248142bee1]
export interface SelectMenuProps<T> {
  menuId: string
  generation?: number
  title: string
  body?: string
  sections?: readonly TerminalSectionData[]
  options: readonly SelectOption<T>[]
  selected: T
  active: boolean
  onChange: (value: T) => void
  onConfirm: (value: T) => void
  onCancel: () => void
  width?: number
  /** Registers the handler with the single parent-owned keyboard listener. */
  onRegisterKeyboard?: (handler: KeyboardInputHandler | null) => void
}

const MAX_MENU_TEXT_CHARS = 2400

/** Controlled, business-agnostic selection UI. */
// [maint-note:964183d0ae6e]
// 维护导航：SelectMenu；下列为本函数源码中的接口与路径索引。
// 输入 { menuId, generation = 0, title, body, sections = [], options, selected, active, onChange, onConfirm, onCancel, width = 80, onRegisterKeyboard, }：SelectMenuProps<T>。
// 分支条件：!menuId.trim()。
// 返回路径：( <Box flexDirection="column" width={contentWidth}> {titleLines.map((line, index) => <Text color="yellow" key={`title-${index}`}>{line}</Text>)} {sections.length > 0 ? ( <> <Box height={1} /> <TerminalSectionStack sections={sections} width=。
// 显式抛出：new Error('SelectMenu menuId must be non-empty')。
// 写入目标：keyboardHandlerRef.current。
// 直接调用：menuId.trim。
// 直接调用：validateSelectOptions。
// 直接调用：useRef。
// 直接调用：useEffect。
// 直接调用：useLayoutEffect。
// 直接调用：calculateContentWidth。
// 直接调用：wrapMenuText。
// 直接调用：options.flatMap。
// 直接调用：titleLines.map。
// 直接调用：bodyLines.map。
// 直接调用：optionLines.map。
// 直接调用：helpLines.map。
// [/maint-note:964183d0ae6e]
export function SelectMenu<T>({
  menuId,
  generation = 0,
  title,
  body,
  sections = [],
  options,
  selected,
  active,
  onChange,
  onConfirm,
  onCancel,
  width = 80,
  onRegisterKeyboard,
}: SelectMenuProps<T>) {
  if (!menuId.trim()) throw new Error('SelectMenu menuId must be non-empty')
  validateSelectOptions(options)
  const activeRef = useRef(active)
  const menuIdRef = useRef(menuId)
  const generationRef = useRef(generation)
  const selectedRef = useRef(selected)
  const optionsRef = useRef(options)

  useEffect(() => {
    activeRef.current = active
    selectedRef.current = selected
    optionsRef.current = options
  }, [active, selected, options])

  useEffect(() => {
    if (menuIdRef.current === menuId && generationRef.current === generation) return
    menuIdRef.current = menuId
    generationRef.current = generation
    activeRef.current = active
    selectedRef.current = selected
    optionsRef.current = options
  }, [active, generation, menuId, options, selected])

  const keyboardHandlerRef = useRef<KeyboardInputHandler | null>(null)
  keyboardHandlerRef.current = (input, key) => {
    if (!activeRef.current || menuIdRef.current !== menuId) return
    if (key.upArrow || key.downArrow) {
      const delta = key.upArrow ? -1 : 1
      const currentIndex = optionsRef.current.findIndex((option) => Object.is(option.value, selectedRef.current))
      const state = createSelectState(menuId, optionsRef.current, generation, currentIndex < 0 ? 0 : currentIndex)
      const nextState = moveSelection(state, delta)
      const next = nextState.options[nextState.selectedIndex]
      if (next) {
        selectedRef.current = next.value
        onChange(next.value)
      }
      return
    }
    if (key.return) {
      const currentIndex = optionsRef.current.findIndex((option) => Object.is(option.value, selectedRef.current))
      const state = createSelectState(menuId, optionsRef.current, generation, currentIndex < 0 ? 0 : currentIndex)
      const confirmed = confirmSelection(state)
      activeRef.current = false
      const current = optionsRef.current[confirmed.state.selectedIndex]
      if (current) onConfirm(current.value)
      return
    }
    if (key.escape || input === '\x1b') {
      const currentIndex = optionsRef.current.findIndex((option) => Object.is(option.value, selectedRef.current))
      const state = createSelectState(menuId, optionsRef.current, generation, currentIndex < 0 ? 0 : currentIndex)
      activeRef.current = cancelSelection(state).active
      onCancel()
    }
  }

  useLayoutEffect(() => {
    if (!active) {
      onRegisterKeyboard?.(null)
      return
    }
    // [maint-note:a09e1faf55c6]
    // 维护导航：handler；下列为本函数源码中的接口与路径索引。
    // 输入 input：类型由调用方与函数体约束。
    // 输入 key：类型由调用方与函数体约束。
    // 直接调用：keyboardHandlerRef.current。
    // [/maint-note:a09e1faf55c6]
    const handler: KeyboardInputHandler = (input, key) => keyboardHandlerRef.current?.(input, key)
    onRegisterKeyboard?.(handler)
    return () => onRegisterKeyboard?.(null)
  }, [active, onRegisterKeyboard])

  const contentWidth = calculateContentWidth({ terminalWidth: width, minWidth: 1 })
  const titleLines = wrapMenuText(title, contentWidth)
  const bodyLines = body === undefined ? [] : wrapMenuText(body, contentWidth)
  const optionLines = options.flatMap((option) => wrapMenuText(
    `${contentWidth <= 3 ? (Object.is(option.value, selected) ? '>' : '') : Object.is(option.value, selected) ? '> ' : '  '}${safeMenuText(option.label)}`,
    contentWidth,
  ))
  const helpLines = wrapMenuText('Use ↑/↓ to select, Enter to confirm, Esc to interrupt.', contentWidth)

  return (
    <Box flexDirection="column" width={contentWidth}>
      {titleLines.map((line, index) => <Text color="yellow" key={`title-${index}`}>{line}</Text>)}
      {sections.length > 0 ? (
        <>
          <Box height={1} />
          <TerminalSectionStack sections={sections} width={contentWidth} />
          <Box height={1} />
        </>
      ) : bodyLines.map((line, index) => <Text key={`body-${index}`}>{line}</Text>)}
      {optionLines.map((line, index) => <Text key={`option-${index}`}>{line}</Text>)}
      {helpLines.map((line, index) => <Text color="gray" key={`help-${index}`}>{line}</Text>)}
    </Box>
  )
}

// [maint-note:20adce9ce131]
// 维护导航：safeMenuText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：[...stripped].slice(0, MAX_MENU_TEXT_CHARS).join('')。
// 直接调用：sanitizeTerminalText。
// 直接调用：[...stripped].slice(0, MAX_MENU_TEXT_CHARS).join。
// 直接调用：[...stripped].slice。
// [/maint-note:20adce9ce131]
function safeMenuText(value: string): string {
  const stripped = sanitizeTerminalText(value)
  return [...stripped].slice(0, MAX_MENU_TEXT_CHARS).join('')
}

// [maint-note:a40cbb2229c0]
// 维护导航：wrapMenuText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 width：number。
// 返回契约：string[]。
// 返回路径：wrapDisplayLines(safeMenuText(value), width, { hard: true, trim: false, wordWrap: true })。
// 直接调用：wrapDisplayLines。
// 直接调用：safeMenuText。
// [/maint-note:a40cbb2229c0]
function wrapMenuText(value: string, width: number): string[] {
  return wrapDisplayLines(safeMenuText(value), width, { hard: true, trim: false, wordWrap: true })
}
