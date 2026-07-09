import { useState, useMemo, useEffect, useRef } from 'react'
import { Box, Text, useInput } from 'ink'
import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

interface inputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
  placeholder?: string
  projectDir: string
  /** 下拉打开时返回 true，供父组件用于阻止全局快捷键 */
  onDropdownChange?: (open: boolean) => void
  width: number
}

function parseAt(value: string, cursorPos: number): { atPos: number; filter: string } | null {
  const textBeforeCursor = value.slice(0, cursorPos)
  const atPos = textBeforeCursor.lastIndexOf('@')
  if (atPos === -1) return null

  // @ 必须在行首或前面是空格
  if (atPos > 0 && textBeforeCursor[atPos - 1] !== ' ') return null

  const filter = value.slice(atPos + 1, cursorPos)

  // 筛选文本不能包含空格
  if (filter.includes(' ')) return null

  return { atPos, filter }
}

const IGNORE_DIRS = new Set(['node_modules', '.git', '__pycache__', '.venv', 'venv', 'build', 'dist', 'out', '.idea', '.vscode'])

// 递归扫描全项目，缓存文件列表
let fileCache: string[] | null = null
let cacheDir: string = ''

function scanAllFiles(projectDir: string): string[] {
  const results: string[] = []

  function walk(dir: string, rel: string) {
    let entries: string[]
    try { entries = readdirSync(dir) } catch { return }

    for (const name of entries) {
      if (name.startsWith('.')) continue
      if (IGNORE_DIRS.has(name)) continue

      const fullPath = join(dir, name)
      const relPath = rel ? `${rel}/${name}` : name

      try {
        if (statSync(fullPath).isDirectory()) {
          results.push(relPath + '/')
          walk(fullPath, relPath)
        } else {
          results.push(relPath)
        }
      } catch { /* 权限问题等，跳过 */ }
    }
  }

  walk(projectDir, '')
  return results
}

function getCachedFiles(projectDir: string): string[] {
  if (cacheDir === projectDir && fileCache) return fileCache
  fileCache = scanAllFiles(projectDir)
  cacheDir = projectDir
  return fileCache
}

function scanFiles(baseDir: string, filter: string): string[] {
  const allFiles = getCachedFiles(baseDir)

  // 将筛选文本按 / 拆开，前缀匹配相对路径
  const match = filter.toLowerCase()
  const matched = allFiles.filter(f => f.toLowerCase().startsWith(match))

  // 目录在前，按字母排序
  matched.sort((a, b) => {
    const aDir = a.endsWith('/') ? 0 : 1
    const bDir = b.endsWith('/') ? 0 : 1
    if (aDir !== bDir) return aDir - bDir
    return a.localeCompare(b)
  })

  return matched
}

const MAX_DROPDOWN_ITEMS = 10

export function InputLine({
  value, onChange, onSubmit, placeholder, projectDir, onDropdownChange, width
}: inputProps) {
  const [cursorPos, setCursorPos] = useState(value.length)
  useEffect(() => {
    setCursorPos(pos => Math.min(pos, value.length))
  },[value.length])

  // 解析 @ 位置和筛选文本
  const atInfo = useMemo(() => parseAt(value, cursorPos), [value, cursorPos])

  // 扫描匹配文件
  const matches = useMemo(() => {
    if (!atInfo) return null
    return scanFiles(projectDir, atInfo.filter)
  }, [atInfo?.atPos, atInfo?.filter, projectDir])

  // 筛选文本变化时重置选择
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [scrollOffset, setScrollOffset] = useState(0)
  useEffect(() => { setSelectedIndex(0); setScrollOffset(0) }, [atInfo?.filter])

  const dropdownOpen = matches !== null && matches.length > 0
  const matchCount = useRef(matches?.length ?? 0)
  useEffect(() => { matchCount.current = matches?.length ?? 0 }, [matches])

  // 保持选中项在可视窗口内
  useEffect(() => {
    if (!matches) return
    const total = matches.length
    const windowSize = Math.min(MAX_DROPDOWN_ITEMS, total)
    setScrollOffset(prev => {
      if (selectedIndex < prev) return selectedIndex
      if (selectedIndex >= prev + windowSize) return selectedIndex - windowSize + 1
      return prev
    })
  }, [selectedIndex, matches])

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
      if (key.return && matches) {
        const selected = matches[selectedIndex]
        if (selected && atInfo) {
          const prefix = value.slice(0, atInfo.atPos) // 到 @ 为止
          const newVal = prefix + selected + ' ' + value.slice(cursorPos)
          onChange(newVal)
          const newCursor = atInfo.atPos + selected.length + 1
          setCursorPos(newCursor)
        }
        return
      }
      // Escape 不下发（不传回父组件，让本组件独吞）
      if (key.escape) {
        // 关闭下拉：删掉 @ 及后面的筛选文本
        if (atInfo) {
          onChange(value.slice(0, atInfo.atPos) + value.slice(cursorPos))
          setCursorPos(atInfo.atPos)
        }
        return
      }
    }

    // ─── 正常文本编辑 ───
    if (key.return) {
      if (!dropdownOpen && value.trim()) {
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

  const before = value.slice(0, cursorPos)
  const at = value[cursorPos] ?? ' '
  const after = value.slice(cursorPos + 1)

  return (
    <Box flexDirection="column">
      <Text>{'─'.repeat(Math.max(0, width - 2))}</Text>
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
      <Text>{'─'.repeat(Math.max(0, width - 2))}</Text>

      {/* 下拉列表 */}
      {dropdownOpen && matches && (
        <Box flexDirection="column" marginBottom={1}>
          {scrollOffset > 0 && (
            <Text color="gray">  ↑ {scrollOffset} more</Text>
          )}
          {matches.slice(scrollOffset, scrollOffset + MAX_DROPDOWN_ITEMS).map((name, i) => {
            const realIndex = scrollOffset + i
            return (
              <Box key={name}>
                {realIndex === selectedIndex ? (
                  <Text color="yellow" inverse>{`> ${name}`}</Text>
                ) : (
                  <Text>  {name}</Text>
                )}
              </Box>
            )
          })}
          {scrollOffset + MAX_DROPDOWN_ITEMS < matches.length && (
            <Text color="gray">  ↓ {matches.length - scrollOffset - MAX_DROPDOWN_ITEMS} more</Text>
          )}
          <Text>{'─'.repeat(Math.max(0, width - 10))}</Text>
        </Box>
      )}
    </Box>
  )
}
