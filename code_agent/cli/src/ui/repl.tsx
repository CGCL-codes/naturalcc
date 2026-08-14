import { useState, useEffect, useRef } from 'react'
import { Box, Text, useApp, useInput } from 'ink'

import { useMessages } from './hooks/useMessages.js'
import { useTimer } from './hooks/useTimer.js'
import { useExecutor } from './hooks/useExecutor.js'
import { dispatch } from './commands/registry.js'
import { Header } from './components/Header.js'
import { MessageList } from './components/MessageList.js'
import { ThinkingBar } from './components/ThinkingBar.js'
import { InputLine } from './components/Input.js'
import type { Ctx } from './types/ctx.js'
import type { SessionManager } from '../app/session/SessionManager.js'
import { clearConversation } from './clearConversation.js'
import { buildApprovalPresentation, type ApprovalController, type ApprovalRequest } from '../tools/approval.js'
import type { Message } from './types/message.js'
import { SelectMenu } from './components/SelectMenu.js'
import type { SelectOption } from './state/selectMenu.js'
import { calculateContentWidth, useTerminalWidth } from './terminal/index.js'

interface ReplProps {
  sessionManager: SessionManager
  approvalProvider?: ApprovalController
  initialPrompt?: string
  /** Test/diagnostic observation point; it receives the same stable UI projection that is rendered. */
  onMessagesChange?: (messages: readonly Message[]) => void
}

export function Repl({ sessionManager, approvalProvider, initialPrompt, onMessagesChange }: ReplProps) {
  const { exit } = useApp()
  const terminal = useTerminalWidth()
  const layout = {
    terminalWidth: terminal.width,
    contentWidth: calculateContentWidth({ terminalWidth: terminal.width, minWidth: 1 }),
    generation: terminal.generation,
  }

  const [input, setInput] = useState('')
  const [exitWarning, setExitWarning] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [clearing, setClearing] = useState(false)
  const clearingRef = useRef(false)

  const msgs = useMessages()
  useEffect(() => {
    onMessagesChange?.(msgs.messages)
  }, [msgs.messages, onMessagesChange])
  const executor = useExecutor({
    sessionManager,
    addMsg: msgs.addMsg,
    beginRun: msgs.beginRun,
    handleAgentEvent: msgs.handleAgentEvent,
    finishRun: msgs.finishRun,
    approvalProvider,
  })
  const initialPromptStarted = useRef(false)
  useEffect(() => {
    if (!initialPrompt || initialPromptStarted.current) return
    initialPromptStarted.current = true
    executor.execute(initialPrompt)
  }, [executor.execute, initialPrompt])
  const { elapsed, dots } = useTimer(executor.loading)

  const ctx: Ctx = {
    addMsg: msgs.addMsg,
    clearMessages: async () => {
      if (clearingRef.current || executor.loading) return
      clearingRef.current = true
      setClearing(true)
      try {
        await clearConversation(sessionManager, msgs.clearMessages)
      } catch (error) {
        process.stderr.write(`${formatError(error)}\n`)
      } finally {
        clearingRef.current = false
        setClearing(false)
      }
    },
    execute: executor.execute,
    exit: () => { process.stdout.write('\n'); exit() },
  }

  useInput((input, key) => {
    if (executor.approvalPending) {
      if ((input === 'c' && key.ctrl) || key.escape || input === '\x1b') {
        executor.interrupt()
      }
      // SelectMenu owns arrows and Enter. In particular, y/n are ordinary
      // input here and must not directly approve a request.
      return
    }
    if (input === 'c' && key.ctrl) {
      if (executor.loading) {
        executor.interrupt()
        return
      }
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
      if (dropdownOpen) return // 下拉打开时 AutocompleteInput 自己处理
      executor.interrupt()
    }
  })

  const handleSubmit = (value: string) => {
    const trimmed = value.trim()
    if (!trimmed || executor.loading || clearingRef.current) return
    void dispatch(trimmed, ctx).then((accepted) => {
      if (accepted) setInput('')
    }).catch((error) => {
      process.stderr.write(`${formatError(error)}\n`)
    })
  }

  return (
    <Box flexDirection="column">
      <Header width={layout.terminalWidth} />
      <MessageList messages={msgs.messages} width={layout.contentWidth} layoutGeneration={layout.generation} />
      <ThinkingBar
        loading={executor.loading}
        isStreaming={executor.isStreaming}
        streamingContent={executor.streamingContent}
        thinkTime={executor.thinkTime}
        elapsed={elapsed}
        dots={dots}
        width={layout.contentWidth}
      />
      {executor.approvalPending && (
        <ApprovalPrompt
          key={executor.approvalPending.requestId}
          request={executor.approvalPending}
          width={layout.contentWidth}
          onDecision={(requestId, approved) => executor.resolveApproval(requestId, approved)}
          onCancel={executor.interrupt}
        />
      )}
      {clearing && <Text color="gray">clearing conversation...</Text>}
      {!executor.loading && !clearing && (
        <Box flexDirection="column">
          <Text color="gray">enter{' '}
            <Text color="red">/help</Text> for help{'  '}
          </Text>
          <InputLine
            value={input}
            onChange={setInput}
            onSubmit={handleSubmit}
            placeholder="Enter command..."
            onDropdownChange={setDropdownOpen}
            width={layout.contentWidth} />
          {exitWarning && <Text color='gray'>press ctrl+c again to exit</Text>}
        </Box>
      )}
    </Box>
  )
}

function ApprovalPrompt({
  request,
  width,
  onDecision,
  onCancel,
}: {
  request: ApprovalRequest
  width: number
  onDecision: (requestId: string, approved: boolean) => void
  onCancel: () => void
}) {
  const [selected, setSelected] = useState<ApprovalChoice>('no')
  const [active, setActive] = useState(true)
  const presentation = buildApprovalPresentation(request)
  const options: readonly SelectOption<ApprovalChoice>[] = [
    { id: 'yes', label: 'Yes', value: 'yes' },
    { id: 'no', label: 'No', value: 'no' },
  ]
  useEffect(() => {
    setSelected('no')
    setActive(true)
  }, [request.requestId])

  return (
    <SelectMenu
      menuId={request.requestId}
      title={presentation.title}
      body={`${presentation.instructionLabel}:\n${presentation.instruction}\n\nReason:\n${presentation.reason}\n`}
      width={width}
      options={options}
      selected={selected}
      active={active}
      onChange={setSelected}
      onConfirm={(choice) => {
        setActive(false)
        onDecision(request.requestId, choice === 'yes')
      }}
      onCancel={() => {
        setActive(false)
        onCancel()
      }}
    />
  )
}

type ApprovalChoice = 'yes' | 'no'

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
