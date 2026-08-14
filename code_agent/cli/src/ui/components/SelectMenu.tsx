import { useEffect, useRef } from 'react'
import { Box, Text, useInput } from 'ink'
import {
  cancelSelection,
  confirmSelection,
  createSelectState,
  moveSelection,
  validateSelectOptions,
  type SelectOption,
} from '../state/selectMenu.js'
import { calculateContentWidth, sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'

export interface SelectMenuProps<T> {
  menuId: string
  generation?: number
  title: string
  body?: string
  options: readonly SelectOption<T>[]
  selected: T
  active: boolean
  onChange: (value: T) => void
  onConfirm: (value: T) => void
  onCancel: () => void
  width?: number
}

const MAX_MENU_TEXT_CHARS = 2400

/** Controlled, business-agnostic selection UI. */
export function SelectMenu<T>({
  menuId,
  generation = 0,
  title,
  body,
  options,
  selected,
  active,
  onChange,
  onConfirm,
  onCancel,
  width = 80,
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

  useInput((input, key) => {
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
  }, { isActive: active })

  const contentWidth = calculateContentWidth({ terminalWidth: width, minWidth: 1 })
  const titleLines = wrapMenuText(title, contentWidth)
  const bodyLines = body === undefined ? [] : wrapMenuText(body, contentWidth)
  const optionLines = options.flatMap((option) => wrapMenuText(
    `${Object.is(option.value, selected) ? '> ' : '  '}${safeMenuText(option.label)}`,
    contentWidth,
  ))
  const helpLines = wrapMenuText('Use ↑/↓ to select, Enter to confirm, Esc to interrupt.', contentWidth)

  return (
    <Box flexDirection="column" width={contentWidth}>
      {titleLines.map((line, index) => <Text color="yellow" key={`title-${index}`}>{line}</Text>)}
      {bodyLines.map((line, index) => <Text key={`body-${index}`}>{line}</Text>)}
      {optionLines.map((line, index) => <Text key={`option-${index}`}>{line}</Text>)}
      {helpLines.map((line, index) => <Text color="gray" key={`help-${index}`}>{line}</Text>)}
    </Box>
  )
}

function safeMenuText(value: string): string {
  const stripped = sanitizeTerminalText(value)
  return [...stripped].slice(0, MAX_MENU_TEXT_CHARS).join('')
}

function wrapMenuText(value: string, width: number): string[] {
  return wrapDisplayLines(safeMenuText(value), width, { hard: true, trim: false, wordWrap: true })
}
