import { useState, useMemo, useEffect, useLayoutEffect, useRef, useImperativeHandle, forwardRef } from 'react'
import { Box, Text } from 'ink'
import { slashCommandDefs } from '../commands/registry.js'
import {
  calculateContentWidth,
  displayLineWidth,
  fillDisplayWidth,
  sanitizeTerminalText,
  splitDisplayGraphemes,
  truncateDisplayText,
} from '../terminal/index.js'
import { UI_PALETTE } from '../theme.js'
import { reduceReplInputChunk } from '../state/replInput.js'
import type { ReplCtrlCAction } from '../state/replCtrlC.js'
import type { DecodedKeyboardInput, KeyboardKey } from '../state/keyboardInput.js'

export interface InputLineHandle {
  handleInput: (input: string, key: KeyboardKey) => void
  handleInputBatch: (events: readonly DecodedKeyboardInput[]) => void
  /**
   * Apply a parent-owned replacement synchronously. When supplied, the
   * monotonically increasing generation also acknowledges the matching
   * controlled-prop reset so it cannot be applied a second time after newer
   * edits in the same stdin transaction.
   */
  resetInput: (value: string, resetGeneration?: number) => void
  /**
   * capture/restore the full editing state (value, cursor, completion
   * dropdown) around the detail viewer, which unmounts the input line while
   * open. Restoring must reproduce cursor-exact typing (abc + X -> aXbc).
   */
  snapshotInputState: () => { value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number }
  restoreInputState: (snapshot: { value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number }) => void
}

export interface InputLineProps {
  /** Parent mirror of the live edit; external replacement increments resetGeneration. */
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
  placeholder?: string
  allowEmptySubmit?: boolean
  /** Reports when this component is mounted for the parent keyboard owner. */
  onInputSubscriptionChange?: (attached: boolean) => void
  /** 下拉打开时返回 true，供父组件用于路由 Escape。 */
  onDropdownChange?: (open: boolean) => void
  /** Increment to reset the controlled cursor after a parent-owned clear. */
  resetGeneration?: number
  /** Parent-owned Ctrl-C decision; InputLine only frames/edit text around it. */
  onControlC?: (inputLength: number, exitArmed: boolean) => ReplCtrlCAction
  exitArmed?: boolean
  renderMarker?: string
  width: number
  /** Keep the editor/keyboard handle mounted while another overlay is visible. */
  hidden?: boolean
  /**
   * editing state captured before the detail viewer unmounted this line.
   * Consumed exactly once at REMOUNT (initializer), so the saved cursor is
   * live in the same commit that brings the input back — a keystroke racing
   * the remount can no longer land at the end-of-line default cursor.
   */
  initialEditingState?: { value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number }
}

interface CompletionItem {
  kind: 'slash'
  value: string
  label: string
  description?: string
}

function parseSlashCommand(value: string, cursorPos: number): { filter: string } | null {
  const textBeforeCursor = value.slice(0, cursorPos)
  if (!textBeforeCursor.startsWith('/')) return null
  if (textBeforeCursor.includes(' ')) return null
  return { filter: textBeforeCursor }
}

function completionItemsFor(value: string, cursorPos: number): CompletionItem[] | null {
  const slashInfo = parseSlashCommand(value, cursorPos)
  if (!slashInfo) return null
  const filter = slashInfo.filter.toLowerCase()
  return slashCommandDefs
    .filter(command => command.name.toLowerCase().startsWith(filter))
    .map(command => ({
      kind: 'slash' as const,
      value: command.name,
      label: command.name,
      description: command.description,
    }))
}

const MAX_DROPDOWN_ITEMS = 10

interface InputLineProjection {
  before: string
  at: string
  after: string
}

/** Keep the editable row to one physical terminal line, even for a paste. */
export function projectInputLine(value: string, cursorPos: number, width: number, placeholder = ''): InputLineProjection {
  const safeWidth = calculateContentWidth({ terminalWidth: width, prefixWidth: 2, minWidth: 1 })
  // Newlines can arrive in a paste chunk. They are editable data, but the
  // interactive projection must not turn them into extra terminal rows.
  const safeValue = sanitizeTerminalText(value).replace(/[\r\n\t]+/gu, ' ')
  const characters = splitDisplayGraphemes(safeValue)
  const cursorPrefix = sanitizeTerminalText(value.slice(0, Math.max(0, cursorPos))).replace(/[\r\n\t]+/gu, ' ')
  const cursor = Math.min(characters.length, splitDisplayGraphemes(cursorPrefix).length)
  const characterWidths = characters.map(character => displayLineWidth(character))
  const widthPrefix = new Array<number>(characterWidths.length + 1).fill(0)
  for (let index = 0; index < characterWidths.length; index += 1) {
    widthPrefix[index + 1] = widthPrefix[index]! + characterWidths[index]!
  }
  const sourceWidth = (from: number, to: number): number => widthPrefix[to]! - widthPrefix[from]!
  const ellipsisWidth = displayLineWidth('…')
  let start = 0
  let end = characters.length
  const emptyPlaceholder = characters.length === 0 ? sanitizeTerminalText(placeholder) : ''

  // Measure each grapheme once. The shrinking loop below updates only
  // indices and reads this prefix sum, so a wide viewport cannot repeatedly
  // rescan and concatenate the whole candidate window.
  if (characters.length > 0 && sourceWidth(0, characters.length) + 1 > safeWidth) {
    const radius = Math.max(4, safeWidth * 2 + 4)
    start = Math.max(0, cursor - radius)
    end = Math.min(characters.length, cursor + radius)
  }

  const visibleContentWidth = (): number =>
    sourceWidth(start, cursor) +
    (cursor < characters.length ? characterWidths[cursor]! : 1) +
    sourceWidth(Math.min(cursor + 1, end), end)
  const projectedWidth = (): number =>
    (start > 0 ? ellipsisWidth : 0) +
    visibleContentWidth() +
    (end < characters.length ? ellipsisWidth : 0)

  while (projectedWidth() > safeWidth && (start < cursor || end > cursor)) {
    if (cursor - start >= end - cursor && start < cursor) start += 1
    else if (end > cursor) end -= 1
    else start += 1
  }

  let showPrefix = start > 0
  let showSuffix = end < characters.length
  const visibleSourceWidth = visibleContentWidth()
  if (visibleSourceWidth + (showPrefix ? ellipsisWidth : 0) + (showSuffix ? ellipsisWidth : 0) > safeWidth) {
    // At very narrow widths a wide cursor grapheme plus two markers cannot
    // fit even after every surrounding grapheme has been removed. Preserve
    // the cursor and the marker for the side hiding more source text.
    if (showPrefix && showSuffix) {
      if (start >= characters.length - end) showSuffix = false
      else showPrefix = false
    } else if (showPrefix) {
      showPrefix = false
    } else if (showSuffix) {
      showSuffix = false
    }
  }
  const prefix = showPrefix ? '…' : ''
  const suffix = showSuffix ? '…' : ''
  return {
    before: prefix + characters.slice(start, cursor).join(''),
    at: truncateDisplayText(characters[cursor] ?? ' ', Math.max(1, safeWidth)),
    after: characters.length === 0
      ? truncateDisplayText(emptyPlaceholder, Math.max(0, safeWidth - 1))
      : characters.slice(cursor + 1, end).join('') + suffix,
  }
}

export const InputLine = forwardRef<InputLineHandle, InputLineProps>(function InputLine({
  value: controlledValue, onChange, onSubmit, placeholder, allowEmptySubmit = false, onDropdownChange,
  resetGeneration = 0, onControlC, exitArmed = false, renderMarker = '', width, onInputSubscriptionChange,
  initialEditingState, hidden = false,
}: InputLineProps, ref) {
  // The viewer restore is mount-time state: if the line remounts with the
  // saved editing snapshot, the FIRST render already carries the saved
  // cursor — no effect window where a racing keystroke could land at the
  // default end-of-line cursor.
  const initialCursor = initialEditingState !== undefined && initialEditingState.value === controlledValue
    ? Math.max(0, Math.min(initialEditingState.cursor, controlledValue.length))
    : controlledValue.length
  const initialEditingRevision = useRef({ value: controlledValue, cursor: initialCursor, revision: 0 })
  const [editingState, setEditingState] = useState(initialEditingRevision.current)
  // One object assignment is the synchronous editing truth. Controlled
  // props acknowledge it; only resetGeneration may replace it from outside.
  const editingStateRef = useRef(initialEditingRevision.current)
  const value = editingState.value
  const cursorPos = editingState.cursor
  const exitArmedRef = useRef(exitArmed)
  const dropdownOpenRef = useRef(initialEditingState?.dropdownOpen === true)
  const selectedIndexRef = useRef(initialEditingState?.dropdownOpen === true ? initialEditingState.selectedIndex : 0)
  const completionItemsRef = useRef<CompletionItem[] | null>(null)
  const batchCompletionItemsRef = useRef<CompletionItem[] | null>(null)
  const batchModeRef = useRef(false)
  const completionFilterRef = useRef<string | null>(parseSlashCommand(controlledValue, controlledValue.length)?.filter ?? null)
  exitArmedRef.current = exitArmed
  const resetGenerationRef = useRef(resetGeneration)
  useLayoutEffect(() => {
    // A lower/equal generation is either a stale parent render or the ACK of
    // a reset already installed through the imperative keyboard-owner path.
    // Neither may overwrite edits/cursor movement performed after that
    // synchronous reset. A genuinely newer generation still deliberately
    // replaces the editing snapshot, even when its value is unchanged.
    if (resetGeneration <= resetGenerationRef.current) return
    resetGenerationRef.current = resetGeneration
    const next = {
      value: controlledValue,
      cursor: controlledValue.length,
      revision: editingStateRef.current.revision + 1,
    }
    editingStateRef.current = next
    setEditingState(next)
  }, [controlledValue, resetGeneration])

  const slashInfo = useMemo(() => parseSlashCommand(value, cursorPos), [value, cursorPos])

  const completionItems = useMemo<CompletionItem[] | null>(() => {
    return completionItemsFor(value, cursorPos)
  }, [value, cursorPos])

  // 筛选文本变化时重置选择
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [scrollOffset, setScrollOffset] = useState(0)
  useEffect(() => {
    selectedIndexRef.current = 0
    setSelectedIndex(0)
    setScrollOffset(0)
  }, [slashInfo?.filter])

  const dropdownOpen = completionItems !== null && completionItems.length > 0
  if (!batchModeRef.current) {
    completionItemsRef.current = completionItems
    dropdownOpenRef.current = dropdownOpen
    selectedIndexRef.current = selectedIndex
    completionFilterRef.current = slashInfo?.filter ?? null
  }
  const matchCount = useRef(completionItems?.length ?? 0)
  useEffect(() => { matchCount.current = completionItems?.length ?? 0 }, [completionItems])

  // 保持选中项在可视窗口内
  useEffect(() => {
    if (!completionItems) return
    const total = completionItems.length
    const windowSize = Math.min(MAX_DROPDOWN_ITEMS, total)
    setScrollOffset(prev => {
      if (selectedIndex < prev) return selectedIndex
      if (selectedIndex >= prev + windowSize) return selectedIndex - windowSize + 1
      return prev
    })
  }, [selectedIndex, completionItems])

  useEffect(() => {
    onDropdownChange?.(dropdownOpen)
  }, [dropdownOpen])

  const installEditingState = (nextValue: string, nextCursor: number): void => {
    const next = {
      value: nextValue,
      cursor: Math.max(0, Math.min(nextCursor, nextValue.length)),
      revision: editingStateRef.current.revision + 1,
    }
    editingStateRef.current = next
    setEditingState(next)
    if (batchModeRef.current) {
      const nextItems = completionItemsFor(next.value, next.cursor)
      const nextFilter = parseSlashCommand(next.value, next.cursor)?.filter ?? null
      if (completionFilterRef.current !== nextFilter) {
        completionFilterRef.current = nextFilter
        selectedIndexRef.current = 0
        setSelectedIndex(0)
        setScrollOffset(0)
      }
      batchCompletionItemsRef.current = nextItems
      dropdownOpenRef.current = nextItems !== null && nextItems.length > 0
      matchCount.current = nextItems?.length ?? 0
    }
  }

  const setCursor = (next: number): void => {
    installEditingState(editingStateRef.current.value, next)
  }

  const commitEdit = (nextValue: string, nextCursor: number): void => {
    // Publish the complete edit snapshot before invoking the controlled
    // parent. A synchronous callback/commit can therefore never observe a
    // new value paired with the preceding cursor revision.
    installEditingState(nextValue, nextCursor)
    onChange(nextValue)
  }

  const resetInput = (next: string, generation?: number): void => {
    if (generation !== undefined) {
      if (generation <= resetGenerationRef.current) return
      resetGenerationRef.current = generation
    }
    installEditingState(next, next.length)
    dropdownOpenRef.current = false
    selectedIndexRef.current = 0
  }

  const snapshotInputState = (): { value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number } => ({
    value: editingStateRef.current.value,
    cursor: editingStateRef.current.cursor,
    dropdownOpen: dropdownOpenRef.current,
    selectedIndex: selectedIndexRef.current,
  })

  const restoreInputState = (snapshot: { value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number }): void => {
    // The parent keeps the controlled value while the viewer is open, so the
    // restore mainly returns the CURSOR (and completion state) the user had;
    // typing after close must land exactly where it would have (abc + X -> aXbc).
    installEditingState(snapshot.value, snapshot.cursor)
    // The dropdown is derived from value+cursor (completionItems), so only
    // the selection index needs carrying; the menu itself re-opens naturally.
    selectedIndexRef.current = snapshot.dropdownOpen ? snapshot.selectedIndex : 0
  }

  const handlePastedInput = (input: string): void => {
    const safeInput = sanitizeTerminalText(input)
    if (!safeInput) return
    const currentValue = editingStateRef.current.value
    const currentCursor = editingStateRef.current.cursor
    const nextValue = currentValue.slice(0, currentCursor) + safeInput + currentValue.slice(currentCursor)
    commitEdit(nextValue, currentCursor + safeInput.length)
  }

  const handleInput = (input: string, key: KeyboardKey): void => {
    const chunkActions = reduceReplInputChunk(input, key.ctrl)
    if (chunkActions.some(action => action.type === 'ctrlC')) {
      let nextValue = editingStateRef.current.value
      let nextCursor = editingStateRef.current.cursor
      let nextExitArmed = exitArmedRef.current

      for (const action of chunkActions) {
        if (action.type === 'text') {
          nextValue = nextValue.slice(0, nextCursor) + action.content + nextValue.slice(nextCursor)
          nextCursor += action.content.length
          nextExitArmed = false
          // Commit each text segment before processing the next control byte.
          // This preserves the original order for chunks such as
          // "abc\\x03def\\x03" and lets the following Ctrl-C observe "def".
          commitEdit(nextValue, nextCursor)
          continue
        }

        const decision = onControlC?.(nextValue.length, nextExitArmed)
        if (!decision) continue
        if (decision === 'clearAndArmExit') {
          nextValue = ''
          nextCursor = 0
          nextExitArmed = true
          continue
        }
        if (decision === 'armExit') {
          nextExitArmed = true
          continue
        }
        if (decision === 'exit' || decision === 'interruptTurn' || decision === 'dismissSelection' || decision === 'ignore') {
          setCursor(nextCursor)
          return
        }
      }

      setCursor(nextCursor)
      return
    }

    const currentValue = editingStateRef.current.value
    const currentCursor = editingStateRef.current.cursor
    const liveCompletionItems = batchModeRef.current ? batchCompletionItemsRef.current : completionItemsRef.current
    const currentDropdownOpen = liveCompletionItems !== null && liveCompletionItems.length > 0

    // ─── 下拉打开时的特殊处理 ───
    if (currentDropdownOpen) {
      if (key.upArrow) {
        const next = selectedIndexRef.current > 0 ? selectedIndexRef.current - 1 : matchCount.current - 1
        selectedIndexRef.current = next
        setSelectedIndex(next)
        return
      }
      if (key.downArrow || key.tab) {
        const next = selectedIndexRef.current < matchCount.current - 1 ? selectedIndexRef.current + 1 : 0
        selectedIndexRef.current = next
        setSelectedIndex(next)
        return
      }
      if (key.return && liveCompletionItems) {
        const selected = liveCompletionItems[selectedIndexRef.current]
        if (selected?.kind === 'slash' && selected.value === currentValue.trim()) {
          onSubmit(currentValue)
          return
        }
        if (selected?.kind === 'slash') {
          const newVal = selected.value + ' ' + currentValue.slice(currentCursor)
          const newCursor = selected.value.length + 1
          commitEdit(newVal, newCursor)
        }
        return
      }
      // Escape 不下发（不传回父组件，让本组件独吞）
      if (key.escape) {
        if (parseSlashCommand(currentValue, currentCursor)) {
          commitEdit(currentValue.slice(currentCursor), 0)
          return
        }
        return
      }
    }

    // ─── 正常文本编辑 ───
    if (key.return) {
      if (!currentDropdownOpen && (currentValue.trim() || allowEmptySubmit)) {
        onSubmit(currentValue)
      }
      return
    }

    if (key.backspace || key.delete) {
      if (currentCursor > 0) {
        const newVal = currentValue.slice(0, currentCursor - 1) + currentValue.slice(currentCursor)
        commitEdit(newVal, currentCursor - 1)
      }
      return
    }

    if (key.leftArrow) {
      if (!currentDropdownOpen && currentCursor > 0) setCursor(currentCursor - 1)
      return
    }

    if (key.rightArrow) {
      if (!currentDropdownOpen && currentCursor < currentValue.length) setCursor(currentCursor + 1)
      return
    }

    if (key.home) { setCursor(0); return }
    if (key.end) { setCursor(currentValue.length); return }

    if (input && !key.ctrl && !key.meta && !key.return && !key.escape) {
      const newVal = currentValue.slice(0, currentCursor) + input + currentValue.slice(currentCursor)
      commitEdit(newVal, currentCursor + input.length)
      return
    }

  }

  const handleInputBatch = (events: readonly DecodedKeyboardInput[]): void => {
    batchModeRef.current = true
    const initialItems = completionItemsFor(editingStateRef.current.value, editingStateRef.current.cursor)
    completionFilterRef.current = parseSlashCommand(editingStateRef.current.value, editingStateRef.current.cursor)?.filter ?? null
    batchCompletionItemsRef.current = initialItems
    dropdownOpenRef.current = initialItems !== null && initialItems.length > 0
    matchCount.current = initialItems?.length ?? 0
    try {
      for (const event of events) {
        if (event.paste) handlePastedInput(event.input)
        else handleInput(event.input, event.key)
      }
    } finally {
      // Keep the ref projection until React commits the controlled value. A
      // following stdin chunk can therefore observe the same completion
      // snapshot even when React batches the preceding onChange calls.
      batchModeRef.current = false
    }
  }

  useImperativeHandle(ref, () => ({ handleInput, handleInputBatch, resetInput, snapshotInputState, restoreInputState }), [value, cursorPos, dropdownOpen, completionItems, selectedIndex, slashInfo, allowEmptySubmit, exitArmed, onChange, onSubmit, onControlC])

  // Reporting this boundary lets the parent keep keystrokes that arrive in
  // the mount-to-ref window instead of dropping them during a remount.
  useEffect(() => {
    onInputSubscriptionChange?.(true)
    return () => { onInputSubscriptionChange?.(false) }
  }, [onInputSubscriptionChange])

  const projection = useMemo(
    () => projectInputLine(value, cursorPos, width, placeholder),
    [cursorPos, placeholder, value, width],
  )

  // The viewer is only a visual/keyboard overlay. Keeping this component
  // mounted preserves the synchronous value+cursor refs, so a close control
  // and following edit in one decoded stdin chunk target the exact cursor.
  if (hidden) return null

  if (width <= 4) {
    const narrowValue = sanitizeTerminalText(value || placeholder || ' ')
    // the narrow one-row fallback keeps the same user surface so the
    // editing area stays visually identified even without the dividers.
    return (
      <Box width={Math.max(1, width)} backgroundColor={UI_PALETTE.userSurfaceBackground}>
        <Text color={UI_PALETTE.userSurfaceForeground}>{renderMarker}{truncateDisplayText(narrowValue, Math.max(1, width))}</Text>
      </Box>
    )
  }

  const dividerWidth = calculateContentWidth({ terminalWidth: width, paddingLeft: 2, minWidth: 0 })
  const dropdownDividerWidth = calculateContentWidth({ terminalWidth: width, paddingLeft: 10, minWidth: 0 })

  return (
    <Box flexDirection="column">
      {/* 上横线/编辑行/下横线三行构成一个连续的用户输入 surface：
          行 Box 的宽度铺满现有安全可绘制宽度（背景来自 Ink box 布局的
          UI 填充，不是写入 value 的空格），横线长度/编辑 viewport/光标
          位置与三行高度保持原样。 */}
      <Box width={Math.max(1, width)} backgroundColor={UI_PALETTE.userSurfaceBackground}>
        <Text color={UI_PALETTE.userSurfaceForeground}>{renderMarker}{fillDisplayWidth('─', dividerWidth)}</Text>
      </Box>
      {/* 输入行 */}
      <Box width={Math.max(1, width)} backgroundColor={UI_PALETTE.userSurfaceBackground}>
        <Text key="prompt" color="green">{renderMarker}{'> '}</Text>
        <Text key="before" color={UI_PALETTE.userSurfaceForeground}>{projection.before}</Text>
        <Text key="cursor" color={UI_PALETTE.userSurfaceForeground} inverse>{projection.at}</Text>
        {!value && placeholder
          ? <Text key="placeholder" color="gray">{projection.after}</Text>
          : <Text key="after" color={UI_PALETTE.userSurfaceForeground}>{projection.after}</Text>}
      </Box>
      <Box width={Math.max(1, width)} backgroundColor={UI_PALETTE.userSurfaceBackground}>
        <Text color={UI_PALETTE.userSurfaceForeground}>{renderMarker}{fillDisplayWidth('─', dividerWidth)}</Text>
      </Box>

      {/* 下拉列表 */}
      {dropdownOpen && completionItems && (
        <Box flexDirection="column" marginBottom={1}>
          {scrollOffset > 0 && (
            <Text color="gray">  ↑ {scrollOffset} more</Text>
          )}
          {completionItems.slice(scrollOffset, scrollOffset + MAX_DROPDOWN_ITEMS).map((item, i) => {
            const realIndex = scrollOffset + i
            const label = item.description
              ? `${item.label}  ${item.description}`
              : item.label
            return (
              <Box key={`${item.kind}:${item.value}`}>
                {realIndex === selectedIndex ? (
                  <Text color="yellow" inverse>{`> ${label}`}</Text>
                ) : (
                  <Text>  {label}</Text>
                )}
              </Box>
            )
          })}
          {scrollOffset + MAX_DROPDOWN_ITEMS < completionItems.length && (
            <Text color="gray">  ↓ {completionItems.length - scrollOffset - MAX_DROPDOWN_ITEMS} more</Text>
          )}
          <Text>{fillDisplayWidth('─', dropdownDividerWidth)}</Text>
        </Box>
      )}
    </Box>
  )
})
