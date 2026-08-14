import { useState, useMemo, useEffect, useRef } from 'react'
import { Box, Text, useInput } from 'ink'
import { slashCommandDefs } from '../commands/registry.js'
import { calculateContentWidth, fillDisplayWidth, sanitizeTerminalText, truncateDisplayText } from '../terminal/index.js'

interface inputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
  placeholder?: string
  allowEmptySubmit?: boolean
  /** 下拉打开时返回 true，供父组件用于阻止全局快捷键 */
  onDropdownChange?: (open: boolean) => void
  width: number
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

const MAX_DROPDOWN_ITEMS = 10

export function InputLine({
  value, onChange, onSubmit, placeholder, allowEmptySubmit = false, onDropdownChange, width
}: inputProps) {
  const [cursorPos, setCursorPos] = useState(value.length)
  useEffect(() => {
    setCursorPos(pos => Math.min(pos, value.length))
  },[value.length])

  const slashInfo = useMemo(() => parseSlashCommand(value, cursorPos), [value, cursorPos])

  const slashMatches = useMemo(() => {
    if (!slashInfo) return null
    const filter = slashInfo.filter.toLowerCase()
    return slashCommandDefs.filter(command => command.name.toLowerCase().startsWith(filter))
  }, [slashInfo?.filter])

  const completionItems = useMemo<CompletionItem[] | null>(() => {
    if (slashMatches === null) return null
    return slashMatches.map(command => ({
      kind: 'slash',
      value: command.name,
      label: command.name,
      description: command.description,
    }))
  }, [slashMatches])

  // 筛选文本变化时重置选择
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [scrollOffset, setScrollOffset] = useState(0)
  useEffect(() => { setSelectedIndex(0); setScrollOffset(0) }, [slashInfo?.filter])

  const dropdownOpen = completionItems !== null && completionItems.length > 0
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

  useInput((input, key) => {
    // ─── 下拉打开时的特殊处理 ───
    if (dropdownOpen) {
      if (key.upArrow) {
        setSelectedIndex(i => (i > 0 ? i - 1 : matchCount.current - 1))
        return
      }
      if (key.downArrow || key.tab) {
        setSelectedIndex(i => (i < matchCount.current - 1 ? i + 1 : 0))
        return
      }
      if (key.return && completionItems) {
        const selected = completionItems[selectedIndex]
        if (selected?.kind === 'slash' && selected.value === value.trim()) {
          onSubmit(value)
          return
        }
        if (selected?.kind === 'slash') {
          const newVal = selected.value + ' ' + value.slice(cursorPos)
          onChange(newVal)
          const newCursor = selected.value.length + 1
          setCursorPos(newCursor)
        }
        return
      }
      // Escape 不下发（不传回父组件，让本组件独吞）
      if (key.escape) {
        if (slashInfo) {
          onChange(value.slice(cursorPos))
          setCursorPos(0)
          return
        }
        return
      }
    }

    // ─── 正常文本编辑 ───
    if (key.return) {
      if (!dropdownOpen && (value.trim() || allowEmptySubmit)) {
        onSubmit(value)
      }
      return
    }

    if (key.backspace || key.delete) {
      if (cursorPos > 0) {
        const newVal = value.slice(0, cursorPos - 1) + value.slice(cursorPos)
        onChange(newVal)
        setCursorPos(cursorPos - 1)
      }
      return
    }

    if (key.leftArrow) {
      if (!dropdownOpen && cursorPos > 0) setCursorPos(p => p - 1)
      return
    }

    if (key.rightArrow) {
      if (!dropdownOpen && cursorPos < value.length) setCursorPos(p => p + 1)
      return
    }

    if (key.home) { setCursorPos(0); return }
    if (key.end) { setCursorPos(value.length); return }

    if (input && !key.ctrl && !key.meta && !key.return && !key.escape) {
      const newVal = value.slice(0, cursorPos) + input + value.slice(cursorPos)
      onChange(newVal)
      setCursorPos(p => p + input.length)
      return
    }

    if (input === 'c' && key.ctrl) {
      onChange('')
      setCursorPos(0)
      return
    }
  })

  if (width <= 4) {
    const narrowValue = sanitizeTerminalText(value || placeholder || ' ')
    return (
      <Box width={Math.max(1, width)}>
        <Text>{truncateDisplayText(narrowValue, Math.max(1, width))}</Text>
      </Box>
    )
  }

  const before = value.slice(0, cursorPos)
  const at = value[cursorPos] ?? ' '
  const after = value.slice(cursorPos + 1)
  const dividerWidth = calculateContentWidth({ terminalWidth: width, paddingLeft: 2, minWidth: 0 })
  const dropdownDividerWidth = calculateContentWidth({ terminalWidth: width, paddingLeft: 10, minWidth: 0 })

  return (
    <Box flexDirection="column">
      <Text>{fillDisplayWidth('─', dividerWidth)}</Text>
      {/* 输入行 */}
      <Box>
        <Text color="green">{'> '}</Text>
        <Text>
          {before}
          <Text inverse>{at}</Text>
          {after}
        </Text>
        {!value && placeholder && <Text color="gray">{placeholder}</Text>}
      </Box>
      <Text>{fillDisplayWidth('─', dividerWidth)}</Text>

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
}
