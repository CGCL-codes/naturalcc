import { useState, useEffect, useRef } from 'react'
import { Box, Text, useApp, useInput, useStdout } from 'ink'

import { useSettings } from './hooks/useSettings.js'
import { useMessages } from './hooks/useMessages.js'
import { useTimer } from './hooks/useTimer.js'
import { useExecutor } from './hooks/useExecutor.js'
import { dispatch } from './commands/registry.js'
import { Header } from './components/Header.js'
import { MessageList } from './components/MessageList.js'
import { ThinkingBar } from './components/ThinkingBar.js'
import { SettingsPanel } from './components/SettingsPanel.js'
import { InputLine } from './components/InputLine.js'
import type { Ctx } from '../types/ctx.js'

function useTerminalWidth() {
  const { stdout } = useStdout()
  const [width, setWidth] = useState(stdout?.columns ?? 80)
  const prevWidthRef = useRef(stdout?.columns ?? 80)

  useEffect(() => {
    const onResize = () => {
      const newWidth = stdout.columns ?? 80
      if (newWidth < prevWidthRef.current) {
        stdout.write('\x1b[H\x1b[2J\x1b[H')
      }
      prevWidthRef.current = newWidth
      setWidth(newWidth)
    }
    stdout?.on('resize', onResize)
    return () => {
      stdout?.off('resize', onResize)
    }
  }, [stdout])

  return width
}

export function Repl() {
  const { exit } = useApp()
  const width = useTerminalWidth()

  const [input, setInput] = useState('')
  const [exitWarning, setExitWarning] = useState(false)

  const settings = useSettings()
  const msgs = useMessages()
  const executor = useExecutor({
    files: settings.files,
    model: settings.model,
    apiKey: settings.apiKey,
    projectDir: settings.projectDir,
    symbol: settings.symbol,
    completionType: settings.completionType,
    prefix: settings.prefix,
    preview: settings.preview,
    addMsg: msgs.addMsg,
  })
  const { elapsed, dots } = useTimer(executor.loading)

  const ctx: Ctx = {
    addMsg: msgs.addMsg,
    clearMessages: msgs.clearMessages,
    error: (msg) => msgs.addMsg('error', msg),
    files: settings.files, setFiles: settings.setFiles,
    model: settings.model, setModel: settings.setModel,
    apiKey: settings.apiKey, setApiKey: settings.setApiKey,
    projectDir: settings.projectDir, setProjectDir: settings.setProjectDir,
    symbol: settings.symbol, setSymbol: settings.setSymbol,
    completionType: settings.completionType, setCompletionType: settings.setCompletionType,
    prefix: settings.prefix, setPrefix: settings.setPrefix,
    preview: settings.preview, togglePreview: settings.togglePreview,
    toggleSettings: settings.toggleSettings,
    resetSettings: settings.resetSettings,
    execute: executor.execute,
    rerun: executor.rerun,
    exit: () => { process.stdout.write('\n'); exit() },
  }

  useInput((input, key) => {
    if (input === 'c' && key.ctrl) {
      if (exitWarning) {
        process.stdout.write('\n')
        exit()
        return
      }
      setExitWarning(true)
      setTimeout(() => setExitWarning(false), 3000)
      return
    }
    if (key.escape) {
      executor.interrupt()
    }
  })

  const handleSubmit = (value: string) => {
    const trimmed = value.trim()
    if (!trimmed || executor.loading) return
    dispatch(trimmed, ctx)
    setInput('')
  }

  return (
    <Box flexDirection="column">
      <Header width={width} />
      <MessageList messages={msgs.messages} />
      <ThinkingBar
        loading={executor.loading}
        isStreaming={executor.isStreaming}
        streamingContent={executor.streamingContent}
        thinkTimeRef={executor.thinkTimeRef}
        elapsed={elapsed}
        dots={dots}
        width={width}
      />
      {!executor.loading && (
        <Box flexDirection="column">
          <SettingsPanel
            visible={settings.settingsWindow}
            files={settings.files}
            model={settings.model}
            apiKey={settings.apiKey}
            projectDir={settings.projectDir}
            symbol={settings.symbol}
            completionType={settings.completionType}
            prefix={settings.prefix}
            preview={settings.preview}
          />
          <Text color="gray">enter{' '}
            <Text color="red">/help</Text> for help{'  '}
            <Text color="red">/settings</Text> to open or close settings window{'  '}
            <Text color="red">--run</Text> to run last instruction{'  '}
          </Text>
          <Text>{'─'.repeat(width - 2)}</Text>
          <InputLine input={input} onChange={setInput} onSubmit={handleSubmit} />
          <Text>{'─'.repeat(width - 2)}</Text>
          {exitWarning && <Text color='gray'>press ctrl+c again to exit</Text>}
        </Box>
      )}
    </Box>
  )
}
