import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react'
import type { AgentEvent } from '../../agent/types/event.js'
import type { MessageType, Role } from '../types/message.js'
import type { Message } from '../types/message.js'
import {
  createMessageState,
  messageReducer,
  releaseActiveToolPresentationSources,
  type ActiveToolPresentationSource,
  type MessageAction,
  type MessageState,
} from './messageReducer.js'
import {
  AssistantStreamRefreshController,
  type AssistantStreamOwner,
} from './assistantStreamRefresh.js'

/**
 * One frozen, immutable turn batch. Batches are append-only: a batch is
 * created once (with a stable ID) and never mutated afterwards, so the
 * stable region prints its records exactly once.
 */
// [maint-note:5f773d410f99]
// 数据契约导航：StableBatch；字段含义以消费端校验为准。
// 成员 batchId：string。
// 成员 messages：readonly Message[]。
// 成员 endsTurn：boolean。
// 成员 separatorOnly（可省略）：boolean。
// 成员 terminalWidth（可省略）：number。
// 成员 presentationGeneration（可省略）：number。
// [/maint-note:5f773d410f99]
export interface StableBatch {
  readonly batchId: string
  readonly messages: readonly Message[]
  /** Whether this item contains the pure separator for a completed turn. */
  readonly endsTurn: boolean
  /** Boundary-only items have no messages and are appended on the next turn. */
  readonly separatorOnly?: boolean
  /** Terminal columns captured by the sole resize owner at handoff. */
  readonly terminalWidth?: number
  /** Native resize generation paired with terminalWidth. */
  readonly presentationGeneration?: number
}

// [maint-note:fe6b671b1bb1]
// 数据契约导航：PendingBoundary；字段含义以消费端校验为准。
// 成员 runId：number。
// [/maint-note:fe6b671b1bb1]
export interface PendingBoundary {
  readonly runId: number
}

// [maint-note:cd6f138bd3dc]
// 数据契约导航：MessageControllerState；字段含义以消费端校验为准。
// 成员 messages：MessageState。
// 成员 batches：readonly StableBatch[]。
// 成员 activeMessages：readonly Message[]。
// 成员 stableGeneration：number。
// 成员 frozenCount：number。
// 成员 nextBatchSeq：number。
// 成员 pendingBoundary：PendingBoundary | null。
// 成员 hasStartedRun：boolean。
// 成员 pendingToolPresentationSources：ReadonlyMap< string, ReadonlyMap<string, ActiveToolPresentationSource> >。
// [/maint-note:cd6f138bd3dc]
export interface MessageControllerState {
  /** Full message history (compat view; the reducer's own state). */
  readonly messages: MessageState
  /** Append-only frozen batches consumed by the stable region. */
  readonly batches: readonly StableBatch[]
  /** Messages not yet frozen; they render in the dynamic region. */
  readonly activeMessages: readonly Message[]
  /** Bumps on /clear: the canonical batch view starts a new generation. */
  readonly stableGeneration: number
  readonly frozenCount: number
  readonly nextBatchSeq: number
  /** The latest completed turn's boundary, not yet frozen into Static. */
  readonly pendingBoundary: PendingBoundary | null
  /** Monotonic within one session generation; reset only by committed clear. */
  readonly hasStartedRun: boolean
  /**
   * Sources detached from resize ownership but retained until the matching
   * Static batch is actually committed. Values are references, not a second
   * transcript or a duplicate detail store.
   */
  readonly pendingToolPresentationSources: ReadonlyMap<
    string,
    ReadonlyMap<string, ActiveToolPresentationSource>
  >
}

// [maint-note:4af0e9ae8392]
// 数据契约导航：PresentationLayoutSnapshot；字段含义以消费端校验为准。
// 成员 contentWidth：number。
// 成员 terminalWidth：number。
// 成员 generation：number。
// [/maint-note:4af0e9ae8392]
export interface PresentationLayoutSnapshot {
  readonly contentWidth: number
  readonly terminalWidth: number
  readonly generation: number
}

// [maint-note:ef85b322ee25]
// 数据契约导航：MessageControllerAction；字段含义以消费端校验为准。
// 类型组合：| { type: 'begin_run'; runId: number } | { type: 'add'; role: Role; content: string; status: MessageType } | { type: 'add_run_error'; runId: number; content: string } | { type: 'agent_event'; runId: number; event: AgentEvent; presentation?:。
// [/maint-note:ef85b322ee25]
export type MessageControllerAction =
  | { type: 'begin_run'; runId: number }
  | { type: 'add'; role: Role; content: string; status: MessageType }
  | { type: 'add_run_error'; runId: number; content: string }
  | { type: 'agent_event'; runId: number; event: AgentEvent; presentation?: PresentationLayoutSnapshot }
  | {
      type: 'finish_run'
      runId: number
      presentationWidth?: number
      presentationGeneration?: number
      terminalWidth?: number
    }
  | { type: 'stable_batches_committed'; generation: number; batchIds: readonly string[] }
  | { type: 'clear' }
  | { type: 'set_presentation_width'; width: number; presentationGeneration?: number }

// [maint-note:4a4d68d05313]
// 维护导航：createMessageControllerState；下列为本函数源码中的接口与路径索引。
// 输入 clock：() => number；缺省表达式 () => Date.now()。
// 返回契约：MessageControllerState。
// 返回路径：{ messages, batches: [], activeMessages: messages.messages, stableGeneration: 1, frozenCount: 0, nextBatchSeq: 1, pendingBoundary: null, hasStartedRun: false, pendingToolPresentationSources: new Map(), }。
// 直接调用：createMessageState。
// [/maint-note:4a4d68d05313]
export function createMessageControllerState(clock: () => number = () => Date.now()): MessageControllerState {
  const messages = createMessageState(clock)
  return {
    messages,
    batches: [],
    activeMessages: messages.messages,
    stableGeneration: 1,
    frozenCount: 0,
    nextBatchSeq: 1,
    pendingBoundary: null,
    hasStartedRun: false,
    pendingToolPresentationSources: new Map(),
  }
}

// [maint-note:f44b2938c885]
// 维护导航：freezeUnfrozen；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageControllerState。
// 输入 previousBatches：readonly StableBatch[]。
// 输入 count：类型由调用方与函数体约束；缺省表达式 state.messages.messages.length - state.frozenCount。
// 输入 options：{ readonly completedRunId?: number readonly terminalWidth?: number readonly presentationGeneration?: number readonly sourcePool?: ReadonlyMap<string, ActiveToolPresentationSource> }；缺省表达式 {}。
// 返回契约：MessageControllerState。
// 分支条件：transferCount === 0。
// 分支条件：!message.correlationKey。
// 分支条件：!source。
// 遍历边界：const message ← unfrozen。
// 返回路径：options.completedRunId === undefined ? state : { ...state, pendingBoundary: { runId: options.completedRunId } }。
// 返回路径：{ ...state, messages, batches: [...previousBatches, batch], activeMessages: all.slice(frozenCount), frozenCount, nextBatchSeq: state.nextBatchSeq + 1, pendingBoundary: options.completedRunId === undefined ? state.pendingBoundary : { runId: 。
// 直接调用：Math.max。
// 直接调用：Math.min。
// 直接调用：all.slice。
// 直接调用：sourcePool.get。
// 直接调用：batchSources.set。
// 直接调用：transferredSourceKeys.push。
// 直接调用：new Map(state.pendingToolPresentationSources).set。
// 直接调用：releaseActiveToolPresentationSources。
// [/maint-note:f44b2938c885]
function freezeUnfrozen(
  state: MessageControllerState,
  previousBatches: readonly StableBatch[],
  count = state.messages.messages.length - state.frozenCount,
  options: {
    readonly completedRunId?: number
    readonly terminalWidth?: number
    readonly presentationGeneration?: number
    readonly sourcePool?: ReadonlyMap<string, ActiveToolPresentationSource>
  } = {},
): MessageControllerState {
  const all = state.messages.messages
  const available = Math.max(0, all.length - state.frozenCount)
  const transferCount = Math.min(Math.max(0, count), available)
  if (transferCount === 0) {
    return options.completedRunId === undefined
      ? state
      : { ...state, pendingBoundary: { runId: options.completedRunId } }
  }
  const unfrozen = all.slice(state.frozenCount, state.frozenCount + transferCount)
  const batch: StableBatch = {
    batchId: `stable-${state.stableGeneration}-${state.nextBatchSeq}`,
    messages: unfrozen,
    endsTurn: false,
    terminalWidth: options.terminalWidth,
    presentationGeneration: options.presentationGeneration,
  }
  const sourcePool = options.sourcePool ?? state.messages.activeToolPresentationSources
  const batchSources = new Map<string, ActiveToolPresentationSource>()
  const transferredSourceKeys: string[] = []
  for (const message of unfrozen) {
    if (!message.correlationKey) continue
    const source = sourcePool.get(message.correlationKey)
    if (!source) continue
    batchSources.set(message.correlationKey, source)
    transferredSourceKeys.push(message.correlationKey)
  }
  const pendingToolPresentationSources = batchSources.size === 0
    ? state.pendingToolPresentationSources
    : new Map(state.pendingToolPresentationSources).set(batch.batchId, batchSources)
  const messages = releaseActiveToolPresentationSources(state.messages, transferredSourceKeys)
  const frozenCount = state.frozenCount + transferCount
  return {
    ...state,
    messages,
    batches: [...previousBatches, batch],
    activeMessages: all.slice(frozenCount),
    frozenCount,
    nextBatchSeq: state.nextBatchSeq + 1,
    pendingBoundary: options.completedRunId === undefined
      ? state.pendingBoundary
      : { runId: options.completedRunId },
    pendingToolPresentationSources,
  }
}

// [maint-note:badf56deb26e]
// 维护导航：eligiblePrefixLength；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageControllerState。
// 返回契约：number。
// 分支条件：runId === null。
// 分支条件：!message || !isStableDuringRun(message, runId)。
// 遍历边界：let index = state.frozenCount; index < state.messages.messages.length; index += 1。
// 返回路径：state.messages.messages.length - state.frozenCount。
// 返回路径：count。
// 写入目标：index。
// 写入目标：count。
// 直接调用：isStableDuringRun。
// [/maint-note:badf56deb26e]
function eligiblePrefixLength(state: MessageControllerState): number {
  const runId = state.messages.activeRunId
  if (runId === null) return state.messages.messages.length - state.frozenCount
  let count = 0
  for (let index = state.frozenCount; index < state.messages.messages.length; index += 1) {
    const message = state.messages.messages[index]
    if (!message || !isStableDuringRun(message, runId)) break
    count += 1
  }
  return count
}

// [maint-note:96a5384dd559]
// 维护导航：isStableDuringRun；下列为本函数源码中的接口与路径索引。
// 输入 message：Message。
// 输入 runId：number。
// 返回契约：boolean。
// 分支条件：!message.correlationKey?.startsWith(prefix) || !message.id。
// 分支条件：message.role === 'user'。
// 分支条件：message.role === 'assistant'。
// 分支条件：message.role !== 'tool'。
// 返回路径：false。
// 返回路径：message.correlationKey === `${prefix}user`。
// 返回路径：message.isDraft !== true && message.correlationKey.startsWith(`${prefix}assistant-segment:`)。
// 返回路径：message.toolPhase === 'succeeded' || message.toolPhase === 'failed' || message.toolPhase === 'denied'。
// 直接调用：message.correlationKey?.startsWith。
// 直接调用：message.correlationKey.startsWith。
// [/maint-note:96a5384dd559]
function isStableDuringRun(message: Message, runId: number): boolean {
  const prefix = `run:${runId}:`
  if (!message.correlationKey?.startsWith(prefix) || !message.id) return false
  if (message.role === 'user') return message.correlationKey === `${prefix}user`
  if (message.role === 'assistant') {
    return message.isDraft !== true && message.correlationKey.startsWith(`${prefix}assistant-segment:`)
  }
  if (message.role !== 'tool') return false
  return message.toolPhase === 'succeeded' || message.toolPhase === 'failed' || message.toolPhase === 'denied'
}

// [maint-note:ef2cb07ddff9]
// 维护导航：applyEventPresentation；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 presentation：PresentationLayoutSnapshot | undefined。
// 返回契约：MessageState。
// 分支条件：!presentation。
// 返回路径：state。
// 返回路径：messageReducer(state, { type: 'set_presentation_width', width: presentation.contentWidth, presentationGeneration: presentation.generation, })。
// 直接调用：messageReducer。
// [/maint-note:ef2cb07ddff9]
function applyEventPresentation(
  state: MessageState,
  presentation: PresentationLayoutSnapshot | undefined,
): MessageState {
  if (!presentation) return state
  return messageReducer(state, {
    type: 'set_presentation_width',
    width: presentation.contentWidth,
    presentationGeneration: presentation.generation,
  })
}

// [maint-note:95fcc2306561]
// 维护导航：acknowledgeStableBatches；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageControllerState。
// 输入 generation：number。
// 输入 batchIds：readonly string[]。
// 返回契约：MessageControllerState。
// 分支条件：generation !== state.stableGeneration || batchIds.length === 0。
// 遍历边界：const batchId ← batchIds。
// 返回路径：state。
// 返回路径：changed ? { ...state, pendingToolPresentationSources: pending } : state。
// 写入目标：changed。
// 直接调用：pending.delete。
// [/maint-note:95fcc2306561]
function acknowledgeStableBatches(
  state: MessageControllerState,
  generation: number,
  batchIds: readonly string[],
): MessageControllerState {
  if (generation !== state.stableGeneration || batchIds.length === 0) return state
  const pending = new Map(state.pendingToolPresentationSources)
  let changed = false
  for (const batchId of batchIds) changed = pending.delete(batchId) || changed
  return changed ? { ...state, pendingToolPresentationSources: pending } : state
}

// [maint-note:944998b14f73]
// 维护导航：freezePendingBoundary；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageControllerState。
// 返回契约：MessageControllerState。
// 分支条件：!state.pendingBoundary。
// 返回路径：state。
// 返回路径：{ ...state, batches: [...state.batches, boundary], nextBatchSeq: state.nextBatchSeq + 1, pendingBoundary: null, }。
// [/maint-note:944998b14f73]
function freezePendingBoundary(state: MessageControllerState): MessageControllerState {
  if (!state.pendingBoundary) return state
  const boundary: StableBatch = {
    batchId: `boundary-${state.stableGeneration}-${state.nextBatchSeq}`,
    messages: [],
    endsTurn: true,
    separatorOnly: true,
  }
  return {
    ...state,
    batches: [...state.batches, boundary],
    nextBatchSeq: state.nextBatchSeq + 1,
    pendingBoundary: null,
  }
}

/**
 * The controller delegates message semantics to messageReducer, then moves
 * the longest immutable prefix to append-only batches. The move and removal
 * from ActiveTurn happen in one reducer transition; run timing and the sole
 * turn boundary remain owned by finish_run.
 */
// [maint-note:161ba32ccd13]
// 维护导航：messageControllerReducer；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageControllerState。
// 输入 action：MessageControllerAction。
// 返回契约：MessageControllerState。
// 分支条件：switch action.type。
// 分支条件：case 'begin_run'。
// 分支条件：case 'clear'。
// 分支条件：case 'stable_batches_committed'。
// 分支条件：action.type === 'agent_event' && ( state.messages.activeRunId !== action.runId || state.messages.terminalRunId === action.runId )。
// 分支条件：action.type === 'add_run_error' && state.messages.activeRunId !== action.runId。
// 分支条件：action.type === 'finish_run' && state.messages.activeRunId !== action.runId。
// 分支条件：next === state.messages。
// 分支条件：action.type === 'finish_run' && state.messages.activeRunId === action.runId。
// 分支条件：next.activeRunId !== null。
// 返回路径：next.activeRunId !== null ? { ...base, activeMessages: next.messages.slice(withBoundary.frozenCount) } : freezeUnfrozen(base, withBoundary.batches)。
// 返回路径：{ messages: cleared, batches: [], activeMessages: cleared.messages, stableGeneration: state.stableGeneration + 1, frozenCount: 0, nextBatchSeq: 1, pendingBoundary: null, hasStartedRun: false, pendingToolPresentationSources: new Map(), }。
// 返回路径：acknowledgeStableBatches(state, action.generation, action.batchIds)。
// 返回路径：state。
// 返回路径：freezeUnfrozen(base, state.batches, undefined, { completedRunId: action.runId, terminalWidth: action.terminalWidth, presentationGeneration: action.presentationGeneration, sourcePool: beforeSources, })。
// 返回路径：count > 0 ? freezeUnfrozen(base, state.batches, count, { terminalWidth: action.type === 'agent_event' ? action.presentation?.terminalWidth : undefined, presentationGeneration: action.type === 'agent_event' ? action.presentation?.generation 。
// 返回路径：freezeUnfrozen(base, state.batches)。
// 直接调用：freezePendingBoundary。
// 直接调用：messageReducer。
// 直接调用：next.messages.slice。
// 直接调用：freezeUnfrozen。
// 直接调用：acknowledgeStableBatches。
// 直接调用：applyEventPresentation。
// 直接调用：eligiblePrefixLength。
// [/maint-note:161ba32ccd13]
export function messageControllerReducer(state: MessageControllerState, action: MessageControllerAction): MessageControllerState {
  switch (action.type) {
    case 'begin_run': {
      // The latest run boundary stays in the interactive area while idle. It
      // becomes a pure separator only when a subsequent turn really starts.
      const withBoundary = freezePendingBoundary(state)
      const next = messageReducer(withBoundary.messages, action)
      const base: MessageControllerState = {
        ...withBoundary,
        messages: next,
        // begin_run follows a successful ExecutionOwner.acquire(). A busy
        // duplicate cannot enter this transition with an idle message owner.
        hasStartedRun: next.activeRunId !== null ? true : withBoundary.hasStartedRun,
      }
      return next.activeRunId !== null
        ? { ...base, activeMessages: next.messages.slice(withBoundary.frozenCount) }
        : freezeUnfrozen(base, withBoundary.batches)
    }
    case 'clear': {
      const cleared = messageReducer(state.messages, { type: 'clear' })
      return {
        messages: cleared,
        batches: [],
        activeMessages: cleared.messages,
        stableGeneration: state.stableGeneration + 1,
        frozenCount: 0,
        nextBatchSeq: 1,
        pendingBoundary: null,
        hasStartedRun: false,
        pendingToolPresentationSources: new Map(),
      }
    }
    case 'stable_batches_committed':
      return acknowledgeStableBatches(state, action.generation, action.batchIds)
    default: {
      if (action.type === 'agent_event' && (
        state.messages.activeRunId !== action.runId || state.messages.terminalRunId === action.runId
      )) return state
      if (action.type === 'add_run_error' && state.messages.activeRunId !== action.runId) return state
      if (action.type === 'finish_run' && state.messages.activeRunId !== action.runId) return state
      const beforeSources = state.messages.activeToolPresentationSources
      const withPresentation = action.type === 'agent_event'
        ? applyEventPresentation(state.messages, action.presentation)
        : state.messages
      const next = messageReducer(withPresentation, action as MessageAction)
      if (next === state.messages) return state
      const base: MessageControllerState = { ...state, messages: next }
      if (action.type === 'finish_run' && state.messages.activeRunId === action.runId) {
        return freezeUnfrozen(base, state.batches, undefined, {
          completedRunId: action.runId,
          terminalWidth: action.terminalWidth,
          presentationGeneration: action.presentationGeneration,
          sourcePool: beforeSources,
        })
      }
      if (next.activeRunId !== null) {
        const count = eligiblePrefixLength(base)
        return count > 0
          ? freezeUnfrozen(base, state.batches, count, {
              terminalWidth: action.type === 'agent_event' ? action.presentation?.terminalWidth : undefined,
              presentationGeneration: action.type === 'agent_event' ? action.presentation?.generation : undefined,
            })
          : { ...base, activeMessages: next.messages.slice(state.frozenCount) }
      }
      return freezeUnfrozen(base, state.batches)
    }
  }
}

// [maint-note:420745c748bd]
// 维护导航：useMessageController；下列为本函数源码中的接口与路径索引。
// 输入 clock：() => number（可省略）。
// 分支条件：streamRefreshRef.current === null。
// 返回路径：{ state, batches: state.batches, activeMessages: state.activeMessages, stableGeneration: state.stableGeneration, stableBatchIds, detailStore: state.messages.detailStore, addMsg, addRunError, beginRun, handleAgentEvent, finishRun, clearMessa。
// 写入目标：streamRefreshRef.current。
// 直接调用：useReducer。
// 直接调用：useRef。
// 直接调用：useEffect。
// 直接调用：useCallback。
// 直接调用：useMemo。
// [/maint-note:420745c748bd]
export function useMessageController(clock?: () => number) {
  const [state, dispatch] = useReducer(messageControllerReducer, undefined, () => createMessageControllerState(clock))
  const streamGenerationRef = useRef(1)
  const streamOwnerRef = useRef<AssistantStreamOwner | null>(null)
  const streamRefreshRef = useRef<AssistantStreamRefreshController | null>(null)
  if (streamRefreshRef.current === null) {
    streamRefreshRef.current = new AssistantStreamRefreshController((owner, content) => {
      dispatch({
        type: 'agent_event',
        runId: owner.runId,
        event: { type: 'assistant_delta', content },
      })
    })
  }
  useEffect(() => () => streamRefreshRef.current?.dispose(), [])

  const addMsg = useCallback((role: Role, content: string, status: MessageType = 'normal'): void => {
    dispatch({ type: 'add', role, content, status })
  }, [])
  const addRunError = useCallback((runId: number, content: string): void => {
    dispatch({ type: 'add_run_error', runId, content })
  }, [])
  const beginRun = useCallback((runId: number): void => {
    const owner = { runId, generation: streamGenerationRef.current }
    streamOwnerRef.current = owner
    streamRefreshRef.current?.begin(owner)
    dispatch({ type: 'begin_run', runId })
  }, [])
  const handleAgentEvent = useCallback((
    runId: number,
    event: AgentEvent,
    presentation?: PresentationLayoutSnapshot,
  ): void => {
    const owner = streamOwnerRef.current
    if (event.type === 'assistant_delta' && owner?.runId === runId) {
      streamRefreshRef.current?.push(owner, event.content)
      return
    }
    if (owner?.runId === runId) {
      if (eventReplacesOrDiscardsDraft(event)) streamRefreshRef.current?.discard(owner)
      else streamRefreshRef.current?.flush(owner)
    }
    dispatch({ type: 'agent_event', runId, event, presentation })
  }, [])
  const finishRun = useCallback((
    runId: number,
    presentationWidth?: number,
    presentationGeneration?: number,
    terminalWidth?: number,
  ): void => {
    const owner = streamOwnerRef.current
    if (owner?.runId === runId) {
      streamRefreshRef.current?.discard(owner)
      streamOwnerRef.current = null
    }
    dispatch({ type: 'finish_run', runId, presentationWidth, presentationGeneration, terminalWidth })
  }, [])
  const clearMessages = useCallback((): void => {
    streamGenerationRef.current += 1
    streamOwnerRef.current = null
    streamRefreshRef.current?.invalidate()
    dispatch({ type: 'clear' })
  }, [])
  const setPresentationWidth = useCallback((width: number, presentationGeneration?: number): void => {
    dispatch({ type: 'set_presentation_width', width, presentationGeneration })
  }, [])
  const acknowledgeStableBatches = useCallback((generation: number, batchIds: readonly string[]): void => {
    dispatch({ type: 'stable_batches_committed', generation, batchIds })
  }, [])

  const stableBatchIds = useMemo(() => state.batches.map((batch) => batch.batchId), [state.batches])

  return {
    state,
    batches: state.batches,
    activeMessages: state.activeMessages,
    stableGeneration: state.stableGeneration,
    stableBatchIds,
    detailStore: state.messages.detailStore,
    addMsg,
    addRunError,
    beginRun,
    handleAgentEvent,
    finishRun,
    clearMessages,
    setPresentationWidth,
    acknowledgeStableBatches,
    dispatch,
  }
}

// [maint-note:d98780aaaa64]
// 维护导航：eventReplacesOrDiscardsDraft；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 返回契约：boolean。
// 分支条件：switch event.type。
// 分支条件：case 'assistant_segment'。
// 分支条件：case 'assistant_message'。
// 分支条件：case 'error'。
// 分支条件：case 'interrupted'。
// 分支条件：case 'paused'。
// 分支条件：case 'done'。
// 返回路径：true。
// 返回路径：false。
// [/maint-note:d98780aaaa64]
function eventReplacesOrDiscardsDraft(event: AgentEvent): boolean {
  switch (event.type) {
    case 'assistant_segment':
    case 'assistant_message':
    case 'error':
    case 'interrupted':
    case 'paused':
    case 'done':
      return true
    default:
      return false
  }
}
