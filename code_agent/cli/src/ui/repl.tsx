import { useState, useEffect, useRef, useCallback, useReducer, useMemo } from 'react'
import { Box, Text, useApp } from 'ink'

import { useMessageController } from './state/messageController.js'
import { DeliveryEnvelopeOwner, DeliveryGate } from './state/deliveryGate.js'
import { useSessionTiming } from './hooks/useSessionTiming.js'
import { useExecutor } from './hooks/useExecutor.js'
import { dispatch } from './commands/registry.js'
import { StableHistory } from './render/StableHistory.js'
import { ActiveTurn } from './render/ActiveTurn.js'
import { InteractiveArea } from './render/InteractiveArea.js'
import type { Ctx } from './types/ctx.js'
import type { SessionManager } from '../app/session/SessionManager.js'
import type { AgentEvent } from '../agent/types/event.js'
import { clearConversation } from './clearConversation.js'
import type { ApprovalController } from '../tools/approval.js'
import type { Message } from './types/message.js'
import { calculateContentWidth, sanitizeTerminalText, useTerminalWidth, wrapDisplayLines } from './terminal/index.js'
import { calculateFrameLayout } from './render/frameLayout.js'
import { buildHeaderProjection, type HeaderProjection } from './components/Header.js'
import type { SessionStatusMetadata } from './components/SessionStatusSlot.js'
import { decideReplCtrlC, type ReplCtrlCAction } from './state/replCtrlC.js'
import { reduceReplInputChunk } from './state/replInput.js'
import { appendTypeahead, extractTypeaheadText } from './state/replTypeahead.js'
import { useKeyboardOwner } from './hooks/useKeyboardOwner.js'
import type { DecodedKeyboardInput, KeyboardInputBatchHandler, KeyboardInputHandler } from './state/keyboardInput.js'
import { detailStoreGet } from './state/detailStore.js'
import {
  detailViewerGeometry,
  detailViewerPage,
  detailViewerReducer,
  initialDetailViewerState,
  projectDetailViewerRecords,
} from './state/detailViewer.js'
import { DetailViewerPanel } from './render/DetailViewerPanel.js'
import type { InputLineHandle } from './components/Input.js'
import type { SessionTimingSnapshot } from './state/sessionTiming.js'
import {
  createStartupReservationState,
  createStartupReservationProjector,
  reduceStartupReservationState,
  type StartupReservationEvent,
} from './render/startupReservation.js'
import { useFrameLayoutObservation } from './render/frameLayoutObservation.js'

interface ReplProps {
  sessionManager: SessionManager
  approvalProvider?: ApprovalController
  initialPrompt?: string
  onKeyboardReady?: () => void
  onInputChange?: (value: string) => void
  /** Test/diagnostic observation point; it receives the same stable UI projection that is rendered. */
  onMessagesChange?: (messages: readonly Message[]) => void
  /** Test/diagnostic observation point; it never changes the rendered timing projection. */
  onTimingChange?: (snapshot: SessionTimingSnapshot) => void
  /** Read-only app metadata for the fixed interactive status slot. */
  statusMetadata?: SessionStatusMetadata
}

export function Repl({ sessionManager, approvalProvider, initialPrompt, onMessagesChange, onTimingChange, onKeyboardReady, onInputChange, statusMetadata }: ReplProps) {
  const { exit } = useApp()
  const terminal = useTerminalWidth()
  const layout = {
    terminalWidth: terminal.width,
    terminalRows: terminal.rows,
    contentWidth: calculateContentWidth({ terminalWidth: terminal.width, minWidth: 1 }),
    generation: terminal.generation,
  }

  const [input, setInput] = useState('')
  const [exitWarning, setExitWarning] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [inputResetGeneration, setInputResetGeneration] = useState(0)
  const inputResetGenerationRef = useRef(0)
  const inputValueRef = useRef('')
  const exitWarningRef = useRef(false)
  const clearingRef = useRef(false)
  const inputLineRef = useRef<InputLineHandle | null>(null)
  const exitWarningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const exitWarningGenerationRef = useRef(0)

  const cancelExitWarning = useCallback(() => {
    exitWarningGenerationRef.current += 1
    if (exitWarningTimerRef.current !== null) {
      clearTimeout(exitWarningTimerRef.current)
      exitWarningTimerRef.current = null
    }
    setExitWarning(false)
    exitWarningRef.current = false
  }, [])

  const armExitWarning = useCallback(() => {
    cancelExitWarning()
    const generation = exitWarningGenerationRef.current + 1
    exitWarningGenerationRef.current = generation
    setExitWarning(true)
    exitWarningRef.current = true
    exitWarningTimerRef.current = setTimeout(() => {
      if (exitWarningGenerationRef.current !== generation) return
      exitWarningTimerRef.current = null
      setExitWarning(false)
      exitWarningRef.current = false
    }, 3000)
  }, [cancelExitWarning])

  useEffect(() => () => {
    exitWarningGenerationRef.current += 1
    if (exitWarningTimerRef.current !== null) clearTimeout(exitWarningTimerRef.current)
    exitWarningTimerRef.current = null
    exitWarningRef.current = false
  }, [])

  const clearInputState = useCallback(() => {
    const resetGeneration = inputResetGenerationRef.current + 1
    inputResetGenerationRef.current = resetGeneration
    setInput('')
    inputValueRef.current = ''
    inputLineRef.current?.resetInput('', resetGeneration)
    setInputResetGeneration(resetGeneration)
    setDropdownOpen(false)
  }, [])

  const clearInput = useCallback(() => {
    clearInputState()
    cancelExitWarning()
  }, [cancelExitWarning, clearInputState])

  const clearAndArmExit = useCallback(() => {
    clearInputState()
    armExitWarning()
  }, [armExitWarning, clearInputState])

  const handleInputChange = useCallback((value: string) => {
    setInput(value)
    inputValueRef.current = value
    onInputChange?.(value)
    // Any new edit cancels a pending empty-input exit warning.
    cancelExitWarning()
  }, [cancelExitWarning, onInputChange])

  const msgs = useMessageController()
  const presentationContentWidth = calculateContentWidth({ terminalWidth: layout.terminalWidth, paddingRight: 4, minWidth: 1 })
  const handleMessageAgentEvent = useCallback((runId: number, event: AgentEvent): void => {
    const latest = terminal.readCurrent()
    msgs.handleAgentEvent(runId, event, {
      contentWidth: calculateContentWidth({ terminalWidth: latest.width, paddingRight: 4, minWidth: 1 }),
      terminalWidth: latest.width,
      generation: latest.generation,
    })
  }, [msgs.handleAgentEvent, terminal.readCurrent])
  const finishMessageRun = useCallback((runId: number): void => {
    const latest = terminal.readCurrent()
    msgs.finishRun(
      runId,
      calculateContentWidth({ terminalWidth: latest.width, paddingRight: 4, minWidth: 1 }),
      latest.generation,
      latest.width,
    )
  }, [msgs.finishRun, terminal.readCurrent])
  // Header is an append-only static sentinel. Keep its ownership explicit so
  // /clear can start a new logical generation without printing a second
  // Header into the native scrollback.
  const headerOwnerRef = useRef<{ generation: number; projection: HeaderProjection } | null>(null)
  if (headerOwnerRef.current === null) {
    headerOwnerRef.current = {
      generation: msgs.stableGeneration,
      projection: buildHeaderProjection(layout.terminalWidth),
    }
  }
  const headerOwner = headerOwnerRef.current!
  const headerWidth = headerOwnerRef.current.generation === msgs.stableGeneration
    ? headerOwnerRef.current.projection.width
    : undefined

  const [startupReservationState, dispatchStartupReservation] = useReducer(
    (state: ReturnType<typeof createStartupReservationState>, event: StartupReservationEvent) =>
      reduceStartupReservationState(state, event),
    undefined,
    () => createStartupReservationState(headerOwner.generation, headerOwner.projection),
  )
  const startupReservationProjectorRef = useRef<ReturnType<typeof createStartupReservationProjector> | null>(null)
  if (startupReservationProjectorRef.current === null) {
    startupReservationProjectorRef.current = createStartupReservationProjector()
  }
  const startupReservation = startupReservationProjectorRef.current(startupReservationState, {
    generation: msgs.stableGeneration,
    width: layout.terminalWidth,
  })
  const handleStableBatchCommit = useCallback((batchIds: readonly string[], generation: number): void => {
    msgs.acknowledgeStableBatches(generation, batchIds)
  }, [msgs.acknowledgeStableBatches])
  // An early user/segment batch no longer means the first run is complete.
  // Keep the frozen Header reservation through that run, then release it in
  // a passive effect after the finish render (and any final Static suffix)
  // has committed. pendingBoundary also covers an empty final suffix.
  useEffect(() => {
    if (msgs.state.pendingBoundary !== null) {
      dispatchStartupReservation({ type: 'stable_batch_committed', generation: msgs.stableGeneration })
    }
  }, [msgs.stableGeneration, msgs.state.pendingBoundary])
  const deliveryGateRef = useRef(new DeliveryGate())
  const deliveryOwnerRef = useRef(new DeliveryEnvelopeOwner(deliveryGateRef.current.sessionGeneration))
  const timing = useSessionTiming(sessionManager.current()?.id ?? 'initial')
  useEffect(() => {
    onMessagesChange?.(msgs.state.messages.messages)
  }, [msgs.state.messages.messages, onMessagesChange])
  useEffect(() => {
    onTimingChange?.(timing.snapshot)
  }, [onTimingChange, timing.snapshot])
  const executor = useExecutor({
    sessionManager,
    addMsg: msgs.addMsg,
    addRunError: msgs.addRunError,
    beginRun: msgs.beginRun,
    handleAgentEvent: handleMessageAgentEvent,
    finishRun: finishMessageRun,
    onRunStarted: timing.beginRun,
    onRunEvent: timing.handleEvent,
    onRunTerminal: (runId) => { timing.finishRun(runId) },
    onRunFinished: timing.finishRun,
    approvalProvider,
    deliveryGate: deliveryGateRef.current,
    deliveryOwner: deliveryOwnerRef.current,
  })
  const exitRepl = useCallback(() => {
    timing.finalizeSession()
    onTimingChange?.(timing.controller.snapshot())
    process.stdout.write('\n')
    exit()
  }, [exit, onTimingChange, timing])
  const handleControlC = useCallback((inputLength: number, exitArmed = exitWarning): ReplCtrlCAction => {
    const action = decideReplCtrlC({
      approvalPending: executor.approvalPending !== null,
      selectionOpen: false,
      loading: executor.loading,
      clearing: clearing || clearingRef.current,
      inputLength,
      exitArmed,
    })
    switch (action) {
      case 'interruptTurn':
        executor.interrupt()
        break
      case 'dismissSelection':
        setDropdownOpen(false)
        break
      case 'clearAndArmExit':
        clearAndArmExit()
        break
      case 'armExit':
        armExitWarning()
        break
      case 'exit':
        exitRepl()
        break
      case 'ignore':
        break
    }
    return action
  }, [armExitWarning, clearAndArmExit, clearing, executor.approvalPending, executor.interrupt, executor.loading, exitRepl, exitWarning])
  const initialPromptStarted = useRef(false)
  useEffect(() => {
    if (!initialPrompt || initialPromptStarted.current) return
    initialPromptStarted.current = true
    executor.execute(initialPrompt)
  }, [executor.execute, initialPrompt])
  const ctx: Ctx = {
    addMsg: msgs.addMsg,
    clearMessages: async () => {
      if (clearingRef.current || executor.loading) return
      clearingRef.current = true
      setClearing(true)
      try {
        await clearConversation(sessionManager, () => {
          msgs.clearMessages()
          timing.clearSession(sessionManager.current()?.id ?? 'cleared')
          // A new session generation invalidates every in-flight envelope:
          // old events can no longer update active, stable, timing, or footer.
          deliveryGateRef.current.reset(deliveryGateRef.current.sessionGeneration + 1)
          deliveryOwnerRef.current.reset(deliveryGateRef.current.sessionGeneration)
        })
      } catch (error) {
        process.stderr.write(`${formatError(error)}\n`)
      } finally {
        clearingRef.current = false
        setClearing(false)
      }
    },
    execute: executor.execute,
    exit: exitRepl,
  }

  // detail viewer. The bounded detail store (inside the message
  // controller) is the ONLY holder of retained lines: stable messages carry
  // light references, so eviction actually frees and no re-put of history
  // happens on re-render. Records are derived from the frozen batches only —
  // the viewer never inspects the AgentThread or re-runs tools.
  const [viewer, dispatchViewer] = useReducer(detailViewerReducer, initialDetailViewerState)
  // The owner flag must move SYNCHRONOUSLY with the key that toggles it: a
  // later keystroke in the SAME stdin chunk must already see the new owner,
  // before React re-renders.
  const viewerOpenRef = useRef(false)
  if (!viewer.open) viewerOpenRef.current = false
  const viewerGeometry = detailViewerGeometry(layout.terminalWidth, layout.terminalRows)
  const viewerViewportRows = viewerGeometry.viewportRows
  const viewerPanelRows = viewerGeometry.panelRows
  const viewerWidth = viewerGeometry.innerWidth
  const detailStore = msgs.detailStore
  const viewableRecords = useMemo(
    () => projectDetailViewerRecords(msgs.batches, detailStore, viewerWidth),
    [msgs.batches, detailStore, viewerWidth],
  )
  useEffect(() => {
    dispatchViewer({ type: 'recordsChanged', records: viewableRecords })
  }, [viewableRecords])
  // Editing state crosses the viewer open/close boundary: snapshot before
  // the input line unmounts, restore after it remounts (cursor-exact).
  const viewerInputSnapshotRef = useRef<ReturnType<InputLineHandle['snapshotInputState']> | null>(null)
  const openDetailViewer = useCallback(() => {
    if (viewerRecordsRef.current.length === 0) return
    viewerInputSnapshotRef.current = inputLineRef.current?.snapshotInputState() ?? null
    viewerOpenRef.current = true
    dispatchViewer({ type: 'open', records: viewerRecordsRef.current })
  }, [])
  const closeDetailViewer = useCallback(() => {
    viewerOpenRef.current = false
    dispatchViewer({ type: 'close' })
  }, [])
  const viewerRecordsRef = useRef(viewableRecords)
  viewerRecordsRef.current = viewableRecords
  const openViewerRef = useRef(openDetailViewer)
  openViewerRef.current = openDetailViewer
  const closeViewerRef = useRef(closeDetailViewer)
  closeViewerRef.current = closeDetailViewer
  // The saved editing state is consumed AT REMOUNT via initialEditingState
  // (mount-time initializer — no effect window for a racing keystroke), and
  // released once the line's subscription handshake completes.
  // Width changes drive FUTURE projections only (committed summaries keep
  // their frozen objects); the viewer re-pages at the current width. The
  // width is the CONTENT width messages render into (MessageColumn pads 4).
  useEffect(() => {
    msgs.setPresentationWidth(presentationContentWidth, layout.generation)
  }, [layout.generation, presentationContentWidth, msgs.setPresentationWidth])

  // Keystrokes that arrive while a turn is running are kept as bounded
  // type-ahead instead of being dropped, and are committed to the input line
  // in one update as soon as it is interactive again. Without this delivery,
  // exit or command text typed in the window between a turn's terminal frame
  // and the input-line remount is silently lost and the REPL never exits.
  // Approval menus and clear transactions keep typed text inert.
  const inputOwnerActive = !executor.loading && !executor.approvalPending && !clearing && !clearingRef.current
  // The overlay can only be open while the input owner is idle: a running
  // turn, approval menu or clear transaction takes the screen back and closes
  // it, so the viewer can never hijack another owner's keys.
  useEffect(() => {
    if (viewer.open && !inputOwnerActive) {
      viewerOpenRef.current = false
      dispatchViewer({ type: 'close' })
    }
  }, [inputOwnerActive, viewer.open])
  const typeaheadRef = useRef('')
  const inputLineSubscribedRef = useRef(false)
  const approvalKeyboardRef = useRef<KeyboardInputHandler | null>(null)
  const drainTypeahead = useCallback(() => {
    const queued = typeaheadRef.current
    if (!queued) return
    typeaheadRef.current = ''
    const next = input + queued
    const resetGeneration = inputResetGenerationRef.current + 1
    inputResetGenerationRef.current = resetGeneration
    setInput(next)
    inputValueRef.current = next
    inputLineRef.current?.resetInput(next, resetGeneration)
    setInputResetGeneration(resetGeneration)
    onInputChange?.(next)
  }, [input, onInputChange])
  const handleInputLineSubscription = useCallback((attached: boolean) => {
    inputLineSubscribedRef.current = attached
    if (attached) {
      // The remounted line has consumed the viewer restore snapshot.
      viewerInputSnapshotRef.current = null
      drainTypeahead()
    }
  }, [drainTypeahead])

  const registerApprovalKeyboard = useCallback((handler: KeyboardInputHandler | null) => {
    approvalKeyboardRef.current = handler
  }, [])

  const handleKeyboardInput: KeyboardInputBatchHandler = useCallback((events) => {
    const handleOne = (event: DecodedKeyboardInput): void => {
      const { input: keyInput, key } = event
      if (viewerOpenRef.current) {
        // while the detail viewer is open it owns every key. Esc/Ctrl+O
        // close it (the flag moves synchronously so a SAME-CHUNK following
        // key already sees the new owner); navigation stays inside the
        // overlay; all other input is swallowed so nothing leaks into the
        // idle input line.
        if (key.ctrl && (keyInput === 'c' || keyInput === 'o')) {
          closeViewerRef.current()
          return
        }
        if (key.escape || keyInput === '\x1b') {
          closeViewerRef.current()
          return
        }
        if (key.pageUp) { dispatchViewer({ type: 'pageUp', viewportRows: viewerViewportRows }); return }
        if (key.pageDown) { dispatchViewer({ type: 'pageDown', viewportRows: viewerViewportRows }); return }
        if (key.home) { dispatchViewer({ type: 'home' }); return }
        if (key.end) { dispatchViewer({ type: 'end', viewportRows: viewerViewportRows }); return }
        if (key.tab && key.shift) { dispatchViewer({ type: 'previousRecord' }); return }
        if (key.tab) { dispatchViewer({ type: 'nextRecord' }); return }
        return
      }
      if (!inputOwnerActive) {
        // Bracketed paste is data, not a sequence of live controls. It may be
        // queued while a turn is running, but it must never submit, interrupt,
        // or navigate an approval/selection owner.
        if (event.paste) {
          if (executor.loading && !executor.approvalPending && !clearing && !clearingRef.current) {
            const text = extractTypeaheadText(sanitizeTerminalText(keyInput))
            if (text) typeaheadRef.current = appendTypeahead(typeaheadRef.current, text)
          }
          return
        }
      const actions = reduceReplInputChunk(keyInput, key.ctrl)
      const hasCtrlC = actions.some(action => action.type === 'ctrlC')
      if (hasCtrlC) handleControlC(0, exitWarning)
      // While an approval menu or a clear transaction owns the screen, typed
      // text stays inert (y/n must not become decisions); while a turn is
      // merely running, it is kept as bounded type-ahead for the next line.
      const typeaheadEligible = executor.loading && !executor.approvalPending && !clearing && !clearingRef.current
      if (executor.approvalPending) {
        if ((key.escape || keyInput === '\x1b') && approvalKeyboardRef.current === null) {
          typeaheadRef.current = ''
          executor.interrupt()
        }
        // The approval menu is a presentation-only child. Its handler is
        // registered with this owner and receives arrows/Enter/Esc here.
        approvalKeyboardRef.current?.(keyInput, key)
      } else if (executor.loading && key.escape) {
        typeaheadRef.current = ''
        executor.interrupt()
      }
      for (const action of actions) {
        if (action.type === 'ctrlC') {
          // An interrupt flushes pending type-ahead, like INTR in a real tty.
          typeaheadRef.current = ''
        } else if (action.type === 'text' && typeaheadEligible) {
          const text = extractTypeaheadText(action.content)
          if (text) typeaheadRef.current = appendTypeahead(typeaheadRef.current, text)
        }
      }
      return
    }
    // Ctrl+O opens the most recent collapsed detail while the input owner
    // is idle. It never opens during a turn, approval or clear transaction.
    // Two Ctrl+O in one chunk toggle in order: open then close.
    if (key.ctrl && keyInput === 'o') {
      openViewerRef.current()
      return
    }
    if (executor.approvalPending) {
      approvalKeyboardRef.current?.(keyInput, key)
      return
    }
    if (key.escape) {
      if (dropdownOpen) {
        inputLineRef.current?.handleInput(keyInput, key)
        return
      }
      executor.interrupt()
      return
    }
    if (!inputLineSubscribedRef.current || inputLineRef.current === null) {
      // Window between the input line mounting and its Ink listener being
      // attached: no one else will receive this text, so keep it as
      // type-ahead and let the subscription boundary commit it.
      if (event.paste) {
        const text = extractTypeaheadText(sanitizeTerminalText(keyInput))
        if (text) typeaheadRef.current = appendTypeahead(typeaheadRef.current, text)
        return
      }
      for (const action of reduceReplInputChunk(keyInput, key.ctrl)) {
        if (action.type === 'ctrlC') {
          handleControlC(inputValueRef.current.length, exitWarningRef.current)
        } else if (action.type === 'text') {
          const text = extractTypeaheadText(action.content)
          if (text) typeaheadRef.current = appendTypeahead(typeaheadRef.current, text)
        }
      }
      return
    }
    if (event.paste) {
      inputLineRef.current.handleInputBatch([event])
      return
    }
    const actions = reduceReplInputChunk(keyInput, key.ctrl)
    if (actions.some((action) => action.type === 'ctrlC')) {
      // Keep Ctrl-C in the parent owner. React state updates from a preceding
      // text data event may not have committed when the next PTY chunk
      // arrives, so the ref is the authoritative synchronous input length.
      for (const action of actions) {
        if (action.type === 'text') {
          inputLineRef.current.handleInput(action.content, { ...key, ctrl: false })
        } else {
          handleControlC(inputValueRef.current.length, exitWarningRef.current)
        }
      }
      return
    }
    inputLineRef.current.handleInput(keyInput, key)
    }

    // The decoder frames a stdin chunk as one owner transaction. The loop is
    // still ordered for approval/typeahead/Ctrl-C semantics, while InputLine
    // receives the complete batch when no parent-owned control action splits
    // it. This prevents React render timing from becoming cursor state.
    // Ctrl+O (detail viewer) is parent-owned exactly like Ctrl-C: it must
    // reach the owner loop, not the input line.
    const parentOwnedControl = (event: DecodedKeyboardInput): boolean =>
      !event.paste && (reduceReplInputChunk(event.input, event.key.ctrl).some(action => action.type === 'ctrlC')
        || (event.key.ctrl && event.input === 'o'))
    if (inputOwnerActive && !viewerOpenRef.current && inputLineSubscribedRef.current && inputLineRef.current !== null &&
      !events.some(parentOwnedControl)) {
      inputLineRef.current.handleInputBatch(events)
      return
    }
    for (const event of events) handleOne(event)
  }, [dropdownOpen, executor.approvalPending, executor.interrupt, executor.loading, exitWarning, handleControlC, input, inputOwnerActive, viewerViewportRows])

  useKeyboardOwner(handleKeyboardInput)

  // The owner is attached before handing SIGINT ownership back to the
  // interactive Repl launcher.
  useEffect(() => {
    const readyTimer = setTimeout(() => onKeyboardReady?.(), 0)
    return () => clearTimeout(readyTimer)
  }, [onKeyboardReady])

  const handleSubmit = (value: string) => {
    const trimmed = value.trim()
    if (!trimmed || executor.loading || clearingRef.current) return
    void dispatch(trimmed, ctx).then((accepted) => {
      if (accepted) clearInput()
    }).catch((error) => {
      process.stderr.write(`${formatError(error)}\n`)
    })
  }

  // At a one-row terminal the status slot is the only renderable owner. The
  // viewer's explicit compact fallback has no panel rows there, so suppress
  // the otherwise second idle-footer row rather than claiming a clamped
  // ledger while Ink still paints two rows.
  const visibleFooter = viewer.open && layout.terminalRows < 2
    ? undefined
    : timing.snapshot.footer ?? undefined
  const showHelpHint = !msgs.state.hasStartedRun
  const frame = calculateFrameLayout({
    width: layout.terminalWidth,
    rows: layout.terminalRows,
    footerVisible: visibleFooter !== undefined,
    inputVisible: !executor.loading && !clearing && !viewer.open,
    showHelpHint,
    clearing,
    exitArmed: exitWarning,
    dropdownOpen,
    reservedStaticRows: startupReservation.reservedRows,
    // Once the Header reservation is released, retain one physical row of
    // slack. Ink clears and replays its accumulated Static output when the
    // previous mutable frame exactly filled the terminal and then shrinks.
    rendererSafetyRows: startupReservation.active ? 0 : 1,
    reservedOverlayRows: viewer.open ? viewerPanelRows : 0,
  })
  const viewerRecord = viewer.open ? viewer.records[viewer.index] : undefined
  const viewerEntry = viewerRecord ? detailStoreGet(detailStore, viewerRecord.recordId) : undefined
  // Page over WRAPPED visual rows so long logical lines stay reachable by
  // paging instead of being permanently truncated sideways.
  const viewerVisualLines = useMemo(() => {
    if (!viewerEntry) return []
    const visual: string[] = []
    for (const line of viewerEntry.lines) {
      const segments = line === ''
        ? ['']
        : wrapDisplayLines(line, viewerWidth, { hard: true, trim: false, wordWrap: false })
      visual.push(...segments)
    }
    return visual
  }, [viewerEntry, viewerWidth])
  const viewerPage = viewerRecord && viewerEntry
    ? detailViewerPage(viewer, viewerVisualLines, viewerViewportRows)
    : { start: 1, end: 0, total: 0 }
  const viewerLogicalTotal = viewerEntry?.lines.length ?? viewerRecord?.totalLogicalLines ?? 0
  const interactiveAreaProps = {
    footer: visibleFooter,
    inputVisible: !executor.loading && !clearing && !viewer.open,
    showHelpHint,
    keepInputMounted: viewer.open,
    input,
    onInputChange: handleInputChange,
    onSubmit: handleSubmit,
    placeholder: 'Enter command...',
    onDropdownChange: setDropdownOpen,
    resetGeneration: inputResetGeneration,
    onControlC: handleControlC,
    exitArmed: exitWarning,
    clearing,
    width: layout.terminalWidth,
    onInputSubscriptionChange: handleInputLineSubscription,
    initialEditingState: viewerInputSnapshotRef.current ?? undefined,
    inputRef: inputLineRef,
    statusMetadata,
  }
  const onFrameLayoutChange = useFrameLayoutObservation()
  useEffect(() => {
    onFrameLayoutChange?.(frame)
  }, [frame.activeRows, frame.interactiveRows, frame.rendererSafetyRows, frame.reservedStaticRows, frame.terminalRows,
    msgs.activeMessages.length, msgs.batches.length, onFrameLayoutChange])

  return (
    <Box flexDirection="column">
      <StableHistory
        batches={msgs.batches}
        width={layout.terminalWidth}
        generation={msgs.stableGeneration}
        headerWidth={headerWidth}
        onBatchCommit={handleStableBatchCommit}
      />
      <ActiveTurn
        messages={msgs.activeMessages}
        loading={executor.loading && timing.snapshot.live}
        streaming={executor.isStreaming}
        durationMs={timing.snapshot.currentTurnMs}
        approvalPaused={timing.snapshot.approvalPaused}
        approvalPending={executor.approvalPending}
        width={layout.terminalWidth}
        onApprovalDecision={(requestId, approved) => executor.resolveApproval(requestId, approved)}
        onApprovalCancel={executor.interrupt}
        onRegisterApprovalKeyboard={registerApprovalKeyboard}
        availableRows={frame.activeRows}
        streamGeneration={msgs.stableGeneration}
        layoutGeneration={layout.generation}
      />
      {viewer.open && viewerRecord && viewerPanelRows > 0 ? (
        <DetailViewerPanel
          recordOrdinal={viewer.index + 1}
          recordCount={viewer.records.length}
          label={viewerRecord.label}
          lines={viewerVisualLines.slice(viewerPage.start - 1, viewerPage.end)}
          start={viewerPage.start}
          end={viewerPage.end}
          totalVisual={viewerPage.total}
          totalLogical={viewerLogicalTotal}
          sourceTruncated={viewerRecord.sourceTruncated}
          evicted={viewerRecord.evicted === true || !viewerEntry}
          width={viewerGeometry.outerWidth}
          height={viewerViewportRows}
          compact={viewerGeometry.compact}
        />
      ) : null}
      <InteractiveArea {...interactiveAreaProps} />
    </Box>
  )
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
