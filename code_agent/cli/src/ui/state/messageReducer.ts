import type { AgentEvent } from '../../agent/types/event.js'
import type { ToolDisplayFacts } from '../../tools/types.js'
import { assertNeverAgentEvent } from '../../agent/terminal.js'
import type { Message, MessageTerminalKind, MessageTerminalReason, MessageType, Role, ToolMessagePhase } from '../types/message.js'
import { presentDeniedTool, presentToolResult, type ToolResultPresentation } from '../toolPresenters/index.js'
import { formatTime } from '../utils/now-time.js'
import {
  commitDetailRecord,
  createDetailStoreState,
  type DetailStoreState,
} from './detailStore.js'

// [maint-note:fce587e336b2]
// 数据契约导航：MessageState；字段含义以消费端校验为准。
// 成员 messages：Message[]。
// 成员 nextId：number。
// 成员 activeRunId：number | null。
// 成员 terminalRunId：number | null。
// 成员 detailStore：DetailStoreState。
// 成员 presentationWidth：number。
// 成员 presentationGeneration：number。
// 成员 activeToolPresentationSources：ReadonlyMap<string, ActiveToolPresentationSource>。
// 成员 clock：() => number。
// [/maint-note:fce587e336b2]
export interface MessageState {
  messages: Message[]
  nextId: number
  activeRunId: number | null
  terminalRunId: number | null
  /**
   * the single bounded holder of retained detail lines. Messages keep
   * only light references; eviction here actually frees the content.
   */
  detailStore: DetailStoreState
  /**
   * content width for NEW physical projections. Committed summaries are
   * never re-presented on resize; only presentations built after the change
   * see the new width.
   */
  presentationWidth: number
  /** Native layout generation used to build the current active projection. */
  presentationGeneration: number
  /**
   * Ephemeral facts for final tool results that have not reached their first
   * Static commit. Resize may re-project these at the commit width; finish
   * releases the facts so stable history never retains a second heavy copy.
   */
  activeToolPresentationSources: ReadonlyMap<string, ActiveToolPresentationSource>
  /** Injectable wall clock; never read during component render. */
  readonly clock: () => number
}

// [maint-note:5959db0659ad]
// 数据契约导航：ActiveToolPresentationSource；字段含义以消费端校验为准。
// 成员 toolName：string。
// 成员 output：string。
// 成员 input：unknown。
// 成员 display（可省略）：ToolDisplayFacts。
// [/maint-note:5959db0659ad]
export interface ActiveToolPresentationSource {
  readonly toolName: string
  readonly output: string
  readonly input: unknown
  readonly display?: ToolDisplayFacts
}

// [maint-note:0a88f6f9eb1d]
// 维护导航：createMessageState；下列为本函数源码中的接口与路径索引。
// 输入 clock：() => number；缺省表达式 () => Date.now()。
// 返回契约：MessageState。
// 返回路径：{ messages: [], nextId: 1, activeRunId: null, terminalRunId: null, detailStore: createDetailStoreState(), presentationWidth: 80, // -1 leaves generation 0 available for the first real terminal snapshot. presentationGeneration: -1, activeToo。
// 直接调用：createDetailStoreState。
// [/maint-note:0a88f6f9eb1d]
export function createMessageState(clock: () => number = () => Date.now()): MessageState {
  return {
    messages: [],
    nextId: 1,
    activeRunId: null,
    terminalRunId: null,
    detailStore: createDetailStoreState(),
    presentationWidth: 80,
    // -1 leaves generation 0 available for the first real terminal snapshot.
    presentationGeneration: -1,
    activeToolPresentationSources: new Map(),
    clock,
  }
}

export const initialMessageState: MessageState = createMessageState()

// [maint-note:24b7c1e150b1]
// 数据契约导航：MessageAction；字段含义以消费端校验为准。
// 类型组合：| { type: 'begin_run'; runId: number } | { type: 'add'; role: Role; content: string; status: MessageType } | { type: 'add_run_error'; runId: number; content: string } | { type: 'agent_event'; runId: number; event: AgentEvent } | { type: 'fi。
// [/maint-note:24b7c1e150b1]
export type MessageAction =
  | { type: 'begin_run'; runId: number }
  | { type: 'add'; role: Role; content: string; status: MessageType }
  | { type: 'add_run_error'; runId: number; content: string }
  | { type: 'agent_event'; runId: number; event: AgentEvent }
  | { type: 'finish_run'; runId: number; presentationWidth?: number; presentationGeneration?: number }
  | { type: 'clear' }
  | { type: 'set_presentation_width'; width: number; presentationGeneration?: number }

// [maint-note:4220bd7e2558]
// 维护导航：messageReducer；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 action：MessageAction。
// 返回契约：MessageState。
// 分支条件：switch action.type。
// 分支条件：case 'begin_run'。
// 分支条件：state.activeRunId !== null。
// 分支条件：case 'add'。
// 分支条件：case 'add_run_error'。
// 分支条件：state.activeRunId !== action.runId。
// 分支条件：case 'agent_event'。
// 分支条件：case 'finish_run'。
// 分支条件：case 'clear'。
// 分支条件：case 'set_presentation_width'。
// 返回路径：state。
// 返回路径：{ ...state, activeRunId: action.runId, terminalRunId: null }。
// 返回路径：appendMessage(state, message(action.role, action.content, action.status))。
// 返回路径：appendMessage(state, { ...message('assistant', action.content, 'error'), tone: 'error', terminalKind: 'error', workspaceMayHaveChanged: false, correlationKey: turnTerminalKey(action.runId), })。
// 返回路径：reduceAgentEvent(state, action.runId, action.event)。
// 返回路径：{ ...removeAssistantDraft(finalizeRunningTools(atCommitLayout, 'interrupted'), action.runId), activeRunId: null, terminalRunId: null, activeToolPresentationSources: new Map(), }。
// 返回路径：{ ...createMessageState(state.clock), presentationWidth: state.presentationWidth, presentationGeneration: state.presentationGeneration, // A new generation invalidates every retained entry; late old-run // events cannot resurrect detail. de。
// 返回路径：applyPresentationLayout(state, action.width, action.presentationGeneration)。
// 直接调用：appendMessage。
// 直接调用：message。
// 直接调用：turnTerminalKey。
// 直接调用：reduceAgentEvent。
// 直接调用：applyPresentationLayout。
// 直接调用：removeAssistantDraft。
// 直接调用：finalizeRunningTools。
// 直接调用：createMessageState。
// 直接调用：createDetailStoreState。
// [/maint-note:4220bd7e2558]
export function messageReducer(state: MessageState, action: MessageAction): MessageState {
  switch (action.type) {
    case 'begin_run':
      if (state.activeRunId !== null) return state
      return { ...state, activeRunId: action.runId, terminalRunId: null }
    case 'add':
      return appendMessage(state, message(action.role, action.content, action.status))
    case 'add_run_error':
      if (state.activeRunId !== action.runId) return state
      return appendMessage(state, {
        ...message('assistant', action.content, 'error'),
        tone: 'error',
        terminalKind: 'error',
        workspaceMayHaveChanged: false,
        correlationKey: turnTerminalKey(action.runId),
      })
    case 'agent_event':
      if (state.activeRunId !== action.runId) return state
      return reduceAgentEvent(state, action.runId, action.event)
    case 'finish_run':
      if (state.activeRunId !== action.runId) return state
      {
        const atCommitLayout = action.presentationWidth === undefined
          ? state
          : applyPresentationLayout(state, action.presentationWidth, action.presentationGeneration)
      return {
        ...removeAssistantDraft(finalizeRunningTools(atCommitLayout, 'interrupted'), action.runId),
        activeRunId: null,
        terminalRunId: null,
        activeToolPresentationSources: new Map(),
      }
      }
    case 'clear':
      return {
        ...createMessageState(state.clock),
        presentationWidth: state.presentationWidth,
        presentationGeneration: state.presentationGeneration,
        // A new generation invalidates every retained entry; late old-run
        // events cannot resurrect detail.
        detailStore: createDetailStoreState(state.detailStore.generation + 1),
        activeToolPresentationSources: new Map(),
      }
    case 'set_presentation_width': {
      return applyPresentationLayout(state, action.width, action.presentationGeneration)
    }
  }
}

// [maint-note:fe77f90421f5]
// 维护导航：applyPresentationLayout；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 widthValue：number。
// 输入 generationValue：number（可省略）。
// 返回契约：MessageState。
// 分支条件：generation < state.presentationGeneration。
// 分支条件：generation === state.presentationGeneration && width !== state.presentationWidth。
// 分支条件：width === state.presentationWidth && generation === state.presentationGeneration。
// 返回路径：state。
// 返回路径：reprojectActiveToolResults({ ...state, presentationWidth: width, presentationGeneration: generation, })。
// 直接调用：Number.isFinite。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：reprojectActiveToolResults。
// [/maint-note:fe77f90421f5]
function applyPresentationLayout(
  state: MessageState,
  widthValue: number,
  generationValue?: number,
): MessageState {
  const width = Number.isFinite(widthValue) ? Math.max(1, Math.floor(widthValue)) : 1
  const generation = generationValue === undefined
    ? state.presentationGeneration + (width === state.presentationWidth ? 0 : 1)
    : Math.max(0, Math.floor(generationValue))
  if (generation < state.presentationGeneration) return state
  // One layout generation names one immutable width fact. A conflicting
  // duplicate cannot rewrite a projection that has already observed it.
  if (generation === state.presentationGeneration && width !== state.presentationWidth) return state
  if (width === state.presentationWidth && generation === state.presentationGeneration) return state
  // Stable summaries have no source entry and remain frozen. Active results
  // retain their neutral display facts until this projection is finalized.
  return reprojectActiveToolResults({
    ...state,
    presentationWidth: width,
    presentationGeneration: generation,
  })
}

// [maint-note:a1687674cd90]
// 维护导航：reduceAgentEvent；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 event：AgentEvent。
// 返回契约：MessageState。
// 分支条件：state.terminalRunId === runId。
// 分支条件：switch event.type。
// 分支条件：case 'user_message'。
// 分支条件：case 'assistant_delta'。
// 分支条件：case 'assistant_segment'。
// 分支条件：case 'assistant_message'。
// 分支条件：case 'tool_start'。
// 分支条件：case 'approval_required'。
// 分支条件：case 'approval_resolved'。
// 分支条件：case 'tool_result'。
// 分支条件：case 'error'。
// 分支条件：case 'interrupted'。
// 分支条件：case 'paused'。
// 分支条件：case 'done'。
// 返回路径：state。
// 返回路径：appendMessage(state, { ...message('user', event.content, 'normal'), correlationKey: turnUserKey(runId), })。
// 返回路径：appendAssistantDelta(state, runId, event.content)。
// 返回路径：appendAssistantSegment(state, runId, event.content)。
// 返回路径：settleAssistantDraft(state, runId, event.content)。
// 返回路径：upsertToolStart(state, runId, event.callId, event.name, event.input)。
// 返回路径：markApprovalPending(state, runId, event.callId, event.name, event.requestId, event.input, event.reason)。
// 返回路径：resolveApproval(state, event.requestId, event.approved)。
// 返回路径：upsertToolResult(state, runId, event.callId, event.name, event.output, event.isError, event.code, event.display)。
// 返回路径：{ ...appendMessage( removeAssistantDraft(finalizeRunningTools(state, 'failed', event.workspaceMayHaveChanged), runId), terminalMessage('assistant', event.message, 'error', 'error', event.workspaceMayHaveChanged, turnTerminalKey(runId)), ), 。
// 返回路径：{ ...appendMessage( removeAssistantDraft(finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged), runId), terminalMessage('assistant', event.message, 'normal', 'interrupted', event.workspaceMayHaveChanged, turnTerminalKey(。
// 返回路径：{ ...settlePausedCheckpoint( finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged), runId, event.content, event.reason, event.workspaceMayHaveChanged, ), terminalRunId: runId, }。
// 返回路径：{ ...state, terminalRunId: runId }。
// 返回路径：assertNeverAgentEvent(event)。
// 直接调用：appendMessage。
// 直接调用：message。
// 直接调用：turnUserKey。
// 直接调用：appendAssistantDelta。
// 直接调用：appendAssistantSegment。
// 直接调用：settleAssistantDraft。
// 直接调用：upsertToolStart。
// 直接调用：markApprovalPending。
// 直接调用：resolveApproval。
// 直接调用：upsertToolResult。
// 直接调用：removeAssistantDraft。
// 直接调用：finalizeRunningTools。
// 直接调用：terminalMessage。
// 直接调用：turnTerminalKey。
// 直接调用：settlePausedCheckpoint。
// 直接调用：assertNeverAgentEvent。
// [/maint-note:a1687674cd90]
function reduceAgentEvent(state: MessageState, runId: number, event: AgentEvent): MessageState {
  if (state.terminalRunId === runId) return state
  switch (event.type) {
    case 'user_message':
      return appendMessage(state, {
        ...message('user', event.content, 'normal'),
        correlationKey: turnUserKey(runId),
      })
    case 'assistant_delta':
      return appendAssistantDelta(state, runId, event.content)
    case 'assistant_segment':
      return appendAssistantSegment(state, runId, event.content)
    case 'assistant_message':
      return settleAssistantDraft(state, runId, event.content)
    case 'tool_start':
      return upsertToolStart(state, runId, event.callId, event.name, event.input)
    case 'approval_required':
      return markApprovalPending(state, runId, event.callId, event.name, event.requestId, event.input, event.reason)
    case 'approval_resolved':
      return resolveApproval(state, event.requestId, event.approved)
    case 'tool_result':
      return upsertToolResult(state, runId, event.callId, event.name, event.output, event.isError, event.code, event.display)
    case 'error':
      return {
        ...appendMessage(
        removeAssistantDraft(finalizeRunningTools(state, 'failed', event.workspaceMayHaveChanged), runId),
        terminalMessage('assistant', event.message, 'error', 'error', event.workspaceMayHaveChanged, turnTerminalKey(runId)),
        ),
        terminalRunId: runId,
      }
    case 'interrupted':
      return {
        ...appendMessage(
        removeAssistantDraft(finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged), runId),
        terminalMessage('assistant', event.message, 'normal', 'interrupted', event.workspaceMayHaveChanged, turnTerminalKey(runId)),
        ),
        terminalRunId: runId,
      }
    case 'paused':
      // The paused terminal is the only carrier of the checkpoint body. A
      // streaming draft for this run is a revocable draft of the same text,
      // so it is settled in place instead of being dropped and re-appended.
      return {
        ...settlePausedCheckpoint(
          finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged),
          runId,
          event.content,
          event.reason,
          event.workspaceMayHaveChanged,
        ),
        terminalRunId: runId,
      }
    case 'done':
      return { ...state, terminalRunId: runId }
    default:
      return assertNeverAgentEvent(event)
  }
}

// [maint-note:1c0130b0b6ba]
// 维护导航：appendAssistantDelta；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 content：string。
// 返回契约：MessageState。
// 分支条件：!content。
// 分支条件：index === -1。
// 分支条件：!existing。
// 返回路径：state。
// 返回路径：appendMessage(state, { ...message('assistant', content, 'normal'), correlationKey, isDraft: true, })。
// 返回路径：replaceMessage(state, index, { ...existing, content: existing.content + content })。
// 直接调用：assistantDraftKey。
// 直接调用：state.messages.findIndex。
// 直接调用：appendMessage。
// 直接调用：message。
// 直接调用：replaceMessage。
// [/maint-note:1c0130b0b6ba]
function appendAssistantDelta(state: MessageState, runId: number, content: string): MessageState {
  if (!content) return state
  const correlationKey = assistantDraftKey(runId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey && item.isDraft)
  if (index === -1) {
    return appendMessage(state, {
      ...message('assistant', content, 'normal'),
      correlationKey,
      isDraft: true,
    })
  }
  const existing = state.messages[index]
  if (!existing) return state
  return replaceMessage(state, index, { ...existing, content: existing.content + content })
}

// [maint-note:0fc3bed1ab66]
// 维护导航：settleAssistantDraft；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 content：string。
// 返回契约：MessageState。
// 分支条件：index === -1。
// 分支条件：!existing。
// 返回路径：appendMessage(state, { ...message('assistant', content, 'normal'), correlationKey: turnTerminalKey(runId), })。
// 返回路径：state。
// 返回路径：replaceMessage(state, index, { ...existing, content, isDraft: false, correlationKey: turnTerminalKey(runId), tone: 'normal', terminalKind: undefined, })。
// 直接调用：assistantDraftKey。
// 直接调用：state.messages.findIndex。
// 直接调用：appendMessage。
// 直接调用：message。
// 直接调用：turnTerminalKey。
// 直接调用：replaceMessage。
// [/maint-note:0fc3bed1ab66]
function settleAssistantDraft(state: MessageState, runId: number, content: string): MessageState {
  const correlationKey = assistantDraftKey(runId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey && item.isDraft)
  if (index === -1) return appendMessage(state, {
    ...message('assistant', content, 'normal'),
    correlationKey: turnTerminalKey(runId),
  })
  const existing = state.messages[index]
  if (!existing) return state
  return replaceMessage(state, index, {
    ...existing,
    content,
    isDraft: false,
    correlationKey: turnTerminalKey(runId),
    tone: 'normal',
    terminalKind: undefined,
  })
}

// [maint-note:bb4abf3b625f]
// 维护导航：appendAssistantSegment；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 content：string。
// 返回契约：MessageState。
// 分支条件：!content.trim()。
// 分支条件：draftIndex !== -1。
// 分支条件：!draft。
// 返回路径：state。
// 返回路径：replaceMessage(state, draftIndex, { ...draft, content, isDraft: false, correlationKey: segmentKey, tone: 'normal', terminalKind: undefined, })。
// 返回路径：appendMessage(state, { ...message('assistant', content, 'normal'), correlationKey: segmentKey, })。
// 直接调用：content.trim。
// 直接调用：assistantDraftKey。
// 直接调用：state.messages.findIndex。
// 直接调用：assistantSegmentKey。
// 直接调用：replaceMessage。
// 直接调用：appendMessage。
// 直接调用：message。
// [/maint-note:bb4abf3b625f]
function appendAssistantSegment(state: MessageState, runId: number, content: string): MessageState {
  if (!content.trim()) return state
  const draftKey = assistantDraftKey(runId)
  const draftIndex = state.messages.findIndex((item) => item.correlationKey === draftKey && item.isDraft)
  const segmentKey = assistantSegmentKey(runId, state.messages)
  if (draftIndex !== -1) {
    const draft = state.messages[draftIndex]
    if (!draft) return state
    return replaceMessage(state, draftIndex, {
      ...draft,
      content,
      isDraft: false,
      correlationKey: segmentKey,
      tone: 'normal',
      terminalKind: undefined,
    })
  }
  return appendMessage(state, {
    ...message('assistant', content, 'normal'),
    correlationKey: segmentKey,
  })
}

// [maint-note:fa03c892cbb8]
// 维护导航：settlePausedCheckpoint；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 content：string。
// 输入 reason：MessageTerminalReason。
// 输入 workspaceMayHaveChanged：boolean。
// 返回契约：MessageState。
// 分支条件：draftIndex === -1。
// 分支条件：!draft。
// 返回路径：appendMessage(state, { ...terminal, terminalReason: reason })。
// 返回路径：replaceMessage(state, draftIndex, { ...draft, ...terminal, terminalReason: reason, isDraft: false, })。
// 直接调用：terminalMessage。
// 直接调用：turnTerminalKey。
// 直接调用：assistantDraftKey。
// 直接调用：state.messages.findIndex。
// 直接调用：appendMessage。
// 直接调用：replaceMessage。
// [/maint-note:fa03c892cbb8]
function settlePausedCheckpoint(
  state: MessageState,
  runId: number,
  content: string,
  reason: MessageTerminalReason,
  workspaceMayHaveChanged: boolean,
): MessageState {
  const terminal = terminalMessage('assistant', content, 'normal', 'paused', workspaceMayHaveChanged, turnTerminalKey(runId))
  const draftKey = assistantDraftKey(runId)
  const draftIndex = state.messages.findIndex((item) => item.correlationKey === draftKey && item.isDraft)
  if (draftIndex === -1) return appendMessage(state, { ...terminal, terminalReason: reason })
  const draft = state.messages[draftIndex]
  if (!draft) return appendMessage(state, { ...terminal, terminalReason: reason })
  return replaceMessage(state, draftIndex, {
    ...draft,
    ...terminal,
    terminalReason: reason,
    isDraft: false,
  })
}

// [maint-note:25feebebcd74]
// 维护导航：removeAssistantDraft；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 返回契约：MessageState。
// 返回路径：messages.length === state.messages.length ? state : { ...state, messages }。
// 直接调用：assistantDraftKey。
// 直接调用：state.messages.filter。
// [/maint-note:25feebebcd74]
function removeAssistantDraft(state: MessageState, runId: number): MessageState {
  const correlationKey = assistantDraftKey(runId)
  const messages = state.messages.filter((item) => !(item.correlationKey === correlationKey && item.isDraft))
  return messages.length === state.messages.length ? state : { ...state, messages }
}

// [maint-note:71323b5801ad]
// 维护导航：upsertToolStart；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 callId：string。
// 输入 toolName：string。
// 输入 input：unknown。
// 返回契约：MessageState。
// 分支条件：index === -1。
// 分支条件：!existing || isFinalToolPhase(existing.toolPhase) || existing.approvalRequestId !== undefined || existing.approvalDecision !== undefined。
// 返回路径：appendMessage(state, { ...toolMessage('tool', toolName, 'running', `Running ${toolName}...`, correlationKey), toolInput: input })。
// 返回路径：state。
// 返回路径：replaceMessage(state, index, { ...existing, toolName, toolPhase: 'running', status: 'normal', tone: 'normal', content: `Running ${toolName}...`, toolInput: input, terminalKind: undefined, workspaceMayHaveChanged: undefined, })。
// 直接调用：toolCorrelationKey。
// 直接调用：state.messages.findIndex。
// 直接调用：appendMessage。
// 直接调用：toolMessage。
// 直接调用：isFinalToolPhase。
// 直接调用：replaceMessage。
// [/maint-note:71323b5801ad]
function upsertToolStart(state: MessageState, runId: number, callId: string, toolName: string, input: unknown): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    return appendMessage(state, { ...toolMessage('tool', toolName, 'running', `Running ${toolName}...`, correlationKey), toolInput: input })
  }
  const existing = state.messages[index]
  // The first approval request owns this tool call for the rest of the
  // projection. A repeated/late request must not replace its input, reason,
  // or request owner after a decision has already been made.
  if (
    !existing ||
    isFinalToolPhase(existing.toolPhase) ||
    existing.approvalRequestId !== undefined ||
    existing.approvalDecision !== undefined
  ) return state
  return replaceMessage(state, index, {
    ...existing,
    toolName,
    toolPhase: 'running',
    status: 'normal',
    tone: 'normal',
    content: `Running ${toolName}...`,
    toolInput: input,
    terminalKind: undefined,
    workspaceMayHaveChanged: undefined,
  })
}

// [maint-note:63d16bd91e97]
// 维护导航：markApprovalPending；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 callId：string。
// 输入 toolName：string。
// 输入 requestId：string。
// 输入 input：unknown。
// 输入 reason：string。
// 返回契约：MessageState。
// 分支条件：index === -1。
// 分支条件：!existing || isFinalToolPhase(existing.toolPhase) || existing.approvalRequestId !== undefined || existing.approvalDecision !== undefined。
// 返回路径：appendMessage(state, { ...toolMessage('tool', toolName, 'awaiting_approval', `Awaiting approval: ${toolName}...`, correlationKey, requestId), toolInput: input, approvalReason: reason, })。
// 返回路径：state。
// 返回路径：replaceMessage(state, index, { ...existing, toolName, toolPhase: 'awaiting_approval', status: 'normal', tone: 'normal', content: `Awaiting approval: ${toolName}...`, approvalRequestId: requestId, toolInput: input, approvalReason: reason, })。
// 直接调用：toolCorrelationKey。
// 直接调用：state.messages.findIndex。
// 直接调用：appendMessage。
// 直接调用：toolMessage。
// 直接调用：isFinalToolPhase。
// 直接调用：replaceMessage。
// [/maint-note:63d16bd91e97]
function markApprovalPending(
  state: MessageState,
  runId: number,
  callId: string,
  toolName: string,
  requestId: string,
  input: unknown,
  reason: string,
): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    return appendMessage(state, {
      ...toolMessage('tool', toolName, 'awaiting_approval', `Awaiting approval: ${toolName}...`, correlationKey, requestId),
      toolInput: input,
      approvalReason: reason,
    })
  }
  const existing = state.messages[index]
  if (
    !existing ||
    isFinalToolPhase(existing.toolPhase) ||
    existing.approvalRequestId !== undefined ||
    existing.approvalDecision !== undefined
  ) return state
  return replaceMessage(state, index, {
    ...existing,
    toolName,
    toolPhase: 'awaiting_approval',
    status: 'normal',
    tone: 'normal',
    content: `Awaiting approval: ${toolName}...`,
    approvalRequestId: requestId,
    toolInput: input,
    approvalReason: reason,
  })
}

// [maint-note:55c81c755f12]
// 维护导航：resolveApproval；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 requestId：string。
// 输入 approved：boolean。
// 返回契约：MessageState。
// 分支条件：index === -1。
// 分支条件：!existing || isFinalToolPhase(existing.toolPhase)。
// 分支条件：existing.approvalDecision !== undefined。
// 分支条件：!approved。
// 返回路径：state。
// 返回路径：commitToolPresentation(replaced, replaced.messages[index]!, presentation, existing.toolName ?? 'tool')。
// 返回路径：replaceMessage(state, index, { ...existing, toolPhase: 'running', status: 'normal', tone: 'normal', content: `Running ${existing.toolName ?? 'tool'}...`, approvalRequestId: undefined, approvalDecision: 'approved', toolPresentation: undefine。
// 直接调用：state.messages.findIndex。
// 直接调用：isFinalToolPhase。
// 直接调用：presentDeniedTool。
// 直接调用：replaceMessage。
// 直接调用：commitToolPresentation。
// [/maint-note:55c81c755f12]
function resolveApproval(state: MessageState, requestId: string, approved: boolean): MessageState {
  const index = state.messages.findIndex((item) => item.approvalRequestId === requestId)
  if (index === -1) return state
  const existing = state.messages[index]
  if (!existing || isFinalToolPhase(existing.toolPhase)) return state
  if (existing.approvalDecision !== undefined) return state
  if (!approved) {
    const presentation = presentDeniedTool(existing.toolName ?? 'tool', existing.toolInput, existing.approvalReason ?? 'User denied this tool call.')
    const replaced = replaceMessage(state, index, {
      ...existing,
      toolPhase: 'denied',
      status: 'normal',
      tone: 'normal',
      content: `Denied: ${existing.toolName ?? 'tool'}`,
      output: 'Denied by user.',
      approvalRequestId: undefined,
      approvalDecision: 'denied',
      toolPresentation: presentation,
    })
    return commitToolPresentation(replaced, replaced.messages[index]!, presentation, existing.toolName ?? 'tool')
  }
  return replaceMessage(state, index, {
    ...existing,
    toolPhase: 'running',
    status: 'normal',
    tone: 'normal',
    content: `Running ${existing.toolName ?? 'tool'}...`,
    approvalRequestId: undefined,
    approvalDecision: 'approved',
    toolPresentation: undefined,
  })
}

// [maint-note:a91c961da562]
// 维护导航：upsertToolResult；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 runId：number。
// 输入 callId：string。
// 输入 toolName：string。
// 输入 output：string。
// 输入 isError：boolean。
// 输入 code：string（可省略）。
// 输入 display：import('../../tools/types.js').ToolDisplayFacts（可省略）。
// 返回契约：MessageState。
// 分支条件：index === -1。
// 分支条件：!existingAtIndex || isFinalToolPhase(existingAtIndex.toolPhase)。
// 返回路径：rememberActiveToolPresentation( commitToolPresentation(appended, created, presentation, label), correlationKey, { toolName, output, input: existing?.toolInput, display }, )。
// 返回路径：state。
// 返回路径：rememberActiveToolPresentation( commitToolPresentation(replaced, replaced.messages[index]!, presentation, label), correlationKey, { toolName, output, input: existing?.toolInput, display }, )。
// 直接调用：toolCorrelationKey。
// 直接调用：state.messages.find。
// 直接调用：presentToolResult。
// 直接调用：state.messages.findIndex。
// 直接调用：appendMessage。
// 直接调用：toolMessage。
// 直接调用：formatToolOutput。
// 直接调用：appended.messages.at。
// 直接调用：rememberActiveToolPresentation。
// 直接调用：commitToolPresentation。
// 直接调用：isFinalToolPhase。
// 直接调用：replaceMessage。
// [/maint-note:a91c961da562]
function upsertToolResult(
  state: MessageState,
  runId: number,
  callId: string,
  toolName: string,
  output: string,
  isError: boolean,
  code?: string,
  display?: import('../../tools/types.js').ToolDisplayFacts,
): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const existing = state.messages.find((item) => item.correlationKey === correlationKey)
  const presentation = presentToolResult(toolName, output, {
    input: existing?.toolInput,
    display,
    contentWidth: state.presentationWidth,
  })
  const phase: ToolMessagePhase = code === 'interrupted' ? 'interrupted' : isError ? 'failed' : 'succeeded'
  const suffix = phase === 'failed' ? ' failed' : phase === 'interrupted' ? ' interrupted' : ''
  const content = `${toolName}${suffix}: ${presentation.summary}`
  const status = phase === 'failed' ? 'error' : 'normal'
  const tone = phase === 'failed' || phase === 'interrupted' ? 'error' : 'normal'
  const label = toolName || 'tool'
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    const appended = appendMessage(state, {
      ...toolMessage('tool', toolName, phase, content, correlationKey),
      status,
      tone,
      output: formatToolOutput(output || 'completed'),
      toolPresentation: presentation,
    })
    const created = appended.messages.at(-1)!
    return rememberActiveToolPresentation(
      commitToolPresentation(appended, created, presentation, label),
      correlationKey,
      { toolName, output, input: existing?.toolInput, display },
    )
  }
  const existingAtIndex = state.messages[index]
  if (!existingAtIndex || isFinalToolPhase(existingAtIndex.toolPhase)) return state
  const {
    approvalRequestId: _approvalRequestId,
    approvalDecision: _approvalDecision,
    ...resultBase
  } = existingAtIndex
  const replaced = replaceMessage(state, index, {
    ...resultBase,
    toolName,
    toolPhase: phase,
    status,
    tone,
    content,
    output: formatToolOutput(output || 'completed'),
    toolPresentation: presentation,
  })
  return rememberActiveToolPresentation(
    commitToolPresentation(replaced, replaced.messages[index]!, presentation, label),
    correlationKey,
    { toolName, output, input: existing?.toolInput, display },
  )
}

// [maint-note:6d35b86044f7]
// 维护导航：rememberActiveToolPresentation；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 correlationKey：string。
// 输入 source：ActiveToolPresentationSource。
// 返回契约：MessageState。
// 返回路径：{ ...state, activeToolPresentationSources }。
// 直接调用：activeToolPresentationSources.set。
// [/maint-note:6d35b86044f7]
function rememberActiveToolPresentation(
  state: MessageState,
  correlationKey: string,
  source: ActiveToolPresentationSource,
): MessageState {
  const activeToolPresentationSources = new Map(state.activeToolPresentationSources)
  activeToolPresentationSources.set(correlationKey, source)
  return { ...state, activeToolPresentationSources }
}

/**
 * End resize ownership for presentations that have atomically left the
 * active suffix. The message controller retains the same source objects
 * until Ink acknowledges their Static batch; this helper only prevents a
 * later resize from rewriting an already-frozen message projection.
 */
// [maint-note:8c7db8bdea40]
// 维护导航：releaseActiveToolPresentationSources；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 correlationKeys：readonly string[]。
// 返回契约：MessageState。
// 分支条件：correlationKeys.length === 0 || state.activeToolPresentationSources.size === 0。
// 遍历边界：const correlationKey ← correlationKeys。
// 返回路径：state。
// 返回路径：changed ? { ...state, activeToolPresentationSources } : state。
// 写入目标：changed。
// 直接调用：activeToolPresentationSources.delete。
// [/maint-note:8c7db8bdea40]
export function releaseActiveToolPresentationSources(
  state: MessageState,
  correlationKeys: readonly string[],
): MessageState {
  if (correlationKeys.length === 0 || state.activeToolPresentationSources.size === 0) return state
  const activeToolPresentationSources = new Map(state.activeToolPresentationSources)
  let changed = false
  for (const correlationKey of correlationKeys) {
    changed = activeToolPresentationSources.delete(correlationKey) || changed
  }
  return changed ? { ...state, activeToolPresentationSources } : state
}

// [maint-note:b6d28579c286]
// 维护导航：reprojectActiveToolResults；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 返回契约：MessageState。
// 分支条件：state.activeToolPresentationSources.size === 0。
// 返回路径：state。
// 返回路径：changed ? { ...state, messages } : state。
// 直接调用：state.messages.map。
// [/maint-note:b6d28579c286]
function reprojectActiveToolResults(state: MessageState): MessageState {
  if (state.activeToolPresentationSources.size === 0) return state
  let changed = false
  const messages = state.messages.map((message) => {
    if (!message.correlationKey || !isFinalToolPhase(message.toolPhase)) return message
    const source = state.activeToolPresentationSources.get(message.correlationKey)
    if (!source) return message
    const projected = presentToolResult(source.toolName, source.output, {
      input: source.input,
      display: source.display,
      contentWidth: state.presentationWidth,
    })
    const { detailSource: _detailSource, ...light } = projected
    const previousSourceTruncated = message.toolPresentation?.detail.sourceTruncated === true
    const presentation: ToolResultPresentation = previousSourceTruncated
      ? {
          ...light,
          truncated: true,
          sourceTruncated: true,
          detail: { ...light.detail, sourceTruncated: true },
        }
      : light
    changed = true
    return {
      ...message,
      content: `${source.toolName}${message.toolPhase === 'failed' ? ' failed' : ''}: ${presentation.summary}`,
      toolPresentation: presentation,
      viewCollapsed: presentation.viewCollapsed,
    }
  })
  return changed ? { ...state, messages } : state
}

// [maint-note:c46189ccc9a6]
// 维护导航：finalizeRunningTools；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 phase：'failed' | 'interrupted'。
// 输入 workspaceMayHaveChanged：类型由调用方与函数体约束；缺省表达式 false。
// 返回契约：MessageState。
// 返回路径：changed ? { ...state, messages } : state。
// 直接调用：state.messages.map。
// [/maint-note:c46189ccc9a6]
function finalizeRunningTools(state: MessageState, phase: 'failed' | 'interrupted', workspaceMayHaveChanged = false): MessageState {
  let changed = false
  const messages = state.messages.map((item) => {
    if (item.role !== 'tool' || !item.toolPhase || !isActiveToolPhase(item.toolPhase)) return item
    changed = true
    const toolName = item.toolName ?? 'tool'
    const terminalKind: MessageTerminalKind = phase === 'failed' ? 'error' : 'interrupted'
    return {
      ...item,
      toolPhase: phase,
      status: (phase === 'failed' ? 'error' : 'normal') as MessageType,
      tone: 'error' as const,
      terminalKind,
      workspaceMayHaveChanged,
      content: phase === 'failed' ? `${toolName} failed` : `${toolName} interrupted`,
      output: phase === 'failed' ? 'failed' : 'interrupted',
      approvalRequestId: undefined,
    }
  })
  return changed ? { ...state, messages } : state
}

/**
 * Split a freshly built presentation: the message keeps the light detail
 * reference (counters + dual truncation state); the heavy bounded lines are
 * committed to the detail store exactly once under the message's stable ID.
 */
// [maint-note:86af89ee3dcf]
// 维护导航：commitToolPresentation；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 message：Message。
// 输入 presentation：ToolResultPresentation。
// 输入 label：string。
// 返回契约：MessageState。
// 分支条件：!source || source.lines.length === 0。
// 分支条件：index === -1。
// 返回路径：state。
// 返回路径：{ ...state, detailStore: committed.state }。
// 返回路径：{ ...state, messages, detailStore: committed.state }。
// 写入目标：messages[index]。
// 直接调用：commitDetailRecord。
// 直接调用：state.messages.indexOf。
// 直接调用：state.messages.slice。
// [/maint-note:86af89ee3dcf]
function commitToolPresentation(
  state: MessageState,
  message: Message,
  presentation: ToolResultPresentation,
  label: string,
): MessageState {
  const source = presentation.detailSource
  if (!source || source.lines.length === 0) return state
  const { detailSource: _stripped, ...light } = presentation
  const committed = commitDetailRecord(state.detailStore, {
    recordId: message.id,
    label,
    lines: source.lines,
    sourceTruncated: source.sourceTruncated,
    viewCollapsed: source.viewCollapsed,
  })
  const merged: ToolResultPresentation = committed.clipped
    ? { ...light, detail: { ...light.detail, sourceTruncated: true } }
    : light
  const index = state.messages.indexOf(message)
  if (index === -1) return { ...state, detailStore: committed.state }
  const messages = state.messages.slice()
  messages[index] = { ...message, toolPresentation: merged, viewCollapsed: merged.viewCollapsed }
  return { ...state, messages, detailStore: committed.state }
}

// [maint-note:5dbc563f13dd]
// 维护导航：appendMessage；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 input：Omit<Message, 'id' | 'createdAtMs' | 'time'>。
// 返回契约：MessageState。
// 返回路径：{ ...state, messages: [...state.messages, appended], nextId: state.nextId + 1, }。
// 直接调用：state.clock。
// 直接调用：formatTime。
// [/maint-note:5dbc563f13dd]
function appendMessage(state: MessageState, input: Omit<Message, 'id' | 'createdAtMs' | 'time'>): MessageState {
  const createdAtMs = state.clock()
  const appended: Message = {
    ...input,
    id: `message-${state.nextId}`,
    createdAtMs,
    time: formatTime(createdAtMs),
  }
  return {
    ...state,
    messages: [...state.messages, appended],
    nextId: state.nextId + 1,
  }
}

// [maint-note:59dbd42324b5]
// 维护导航：replaceMessage；下列为本函数源码中的接口与路径索引。
// 输入 state：MessageState。
// 输入 index：number。
// 输入 value：Message。
// 返回契约：MessageState。
// 返回路径：{ ...state, messages }。
// 写入目标：messages[index]。
// 直接调用：state.messages.slice。
// [/maint-note:59dbd42324b5]
function replaceMessage(state: MessageState, index: number, value: Message): MessageState {
  const messages = state.messages.slice()
  messages[index] = value
  return { ...state, messages }
}

// [maint-note:a1c054448d90]
// 维护导航：message；下列为本函数源码中的接口与路径索引。
// 输入 role：Role。
// 输入 content：string。
// 输入 status：MessageType。
// 返回契约：Omit<Message, 'id' | 'createdAtMs' | 'time'>。
// 返回路径：{ role, status, content, tone: status === 'error' ? 'error' : 'normal' }。
// [/maint-note:a1c054448d90]
function message(role: Role, content: string, status: MessageType): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return { role, status, content, tone: status === 'error' ? 'error' : 'normal' }
}

// [maint-note:96c06349e7b3]
// 维护导航：terminalMessage；下列为本函数源码中的接口与路径索引。
// 输入 role：'assistant'。
// 输入 content：string。
// 输入 status：MessageType。
// 输入 terminalKind：'error' | 'interrupted' | 'paused'。
// 输入 workspaceMayHaveChanged：boolean。
// 输入 correlationKey：string。
// 返回契约：Omit<Message, 'id' | 'createdAtMs' | 'time'>。
// 返回路径：{ ...message(role, content, status), tone: terminalKind === 'paused' ? 'normal' : 'error', terminalKind, workspaceMayHaveChanged, correlationKey, }。
// 直接调用：message。
// [/maint-note:96c06349e7b3]
function terminalMessage(
  role: 'assistant',
  content: string,
  status: MessageType,
  terminalKind: 'error' | 'interrupted' | 'paused',
  workspaceMayHaveChanged: boolean,
  correlationKey: string,
): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return {
    ...message(role, content, status),
    tone: terminalKind === 'paused' ? 'normal' : 'error',
    terminalKind,
    workspaceMayHaveChanged,
    correlationKey,
  }
}

// [maint-note:93bc9d3cb1b9]
// 维护导航：toolMessage；下列为本函数源码中的接口与路径索引。
// 输入 role：'tool'。
// 输入 toolName：string。
// 输入 toolPhase：ToolMessagePhase。
// 输入 content：string。
// 输入 correlationKey：string。
// 输入 approvalRequestId：string（可省略）。
// 返回契约：Omit<Message, 'id' | 'createdAtMs' | 'time'>。
// 返回路径：{ ...message(role, content, toolPhase === 'failed' ? 'error' : 'normal'), tone: toolPhase === 'failed' || toolPhase === 'interrupted' ? 'error' : 'normal', toolName, toolPhase, correlationKey, approvalRequestId, }。
// 直接调用：message。
// [/maint-note:93bc9d3cb1b9]
function toolMessage(
  role: 'tool',
  toolName: string,
  toolPhase: ToolMessagePhase,
  content: string,
  correlationKey: string,
  approvalRequestId?: string,
): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return {
    ...message(role, content, toolPhase === 'failed' ? 'error' : 'normal'),
    tone: toolPhase === 'failed' || toolPhase === 'interrupted' ? 'error' : 'normal',
    toolName,
    toolPhase,
    correlationKey,
    approvalRequestId,
  }
}

// [maint-note:2319560c4d00]
// 维护导航：isActiveToolPhase；下列为本函数源码中的接口与路径索引。
// 输入 phase：ToolMessagePhase。
// 返回契约：boolean。
// 返回路径：phase === 'running' || phase === 'awaiting_approval'。
// [/maint-note:2319560c4d00]
function isActiveToolPhase(phase: ToolMessagePhase): boolean {
  return phase === 'running' || phase === 'awaiting_approval'
}

// [maint-note:a661409d8193]
// 维护导航：isFinalToolPhase；下列为本函数源码中的接口与路径索引。
// 输入 phase：ToolMessagePhase | undefined。
// 返回契约：boolean。
// 返回路径：phase === 'succeeded' || phase === 'failed' || phase === 'denied' || phase === 'interrupted'。
// [/maint-note:a661409d8193]
function isFinalToolPhase(phase: ToolMessagePhase | undefined): boolean {
  return phase === 'succeeded' || phase === 'failed' || phase === 'denied' || phase === 'interrupted'
}

// [maint-note:90534aef5f94]
// 维护导航：toolCorrelationKey；下列为本函数源码中的接口与路径索引。
// 输入 runId：number。
// 输入 callId：string。
// 返回契约：string。
// 返回路径：`run:${runId}:call:${callId}`。
// [/maint-note:90534aef5f94]
function toolCorrelationKey(runId: number, callId: string): string {
  return `run:${runId}:call:${callId}`
}

// [maint-note:7918eedff4cf]
// 维护导航：assistantDraftKey；下列为本函数源码中的接口与路径索引。
// 输入 runId：number。
// 返回契约：string。
// 返回路径：`run:${runId}:assistant-draft`。
// [/maint-note:7918eedff4cf]
function assistantDraftKey(runId: number): string {
  return `run:${runId}:assistant-draft`
}

// [maint-note:816c86963b50]
// 维护导航：assistantSegmentKey；下列为本函数源码中的接口与路径索引。
// 输入 runId：number。
// 输入 messages：readonly Message[]。
// 返回契约：string。
// 返回路径：`${prefix}${count + 1}`。
// 直接调用：messages.filter。
// [/maint-note:816c86963b50]
function assistantSegmentKey(runId: number, messages: readonly Message[]): string {
  const prefix = `run:${runId}:assistant-segment:`
  const count = messages.filter((item) => item.correlationKey?.startsWith(prefix)).length
  return `${prefix}${count + 1}`
}

// [maint-note:b313ac53f425]
// 维护导航：turnTerminalKey；下列为本函数源码中的接口与路径索引。
// 输入 runId：number。
// 返回契约：string。
// 返回路径：`run:${runId}:terminal`。
// [/maint-note:b313ac53f425]
export function turnTerminalKey(runId: number): string {
  return `run:${runId}:terminal`
}

// [maint-note:3f62c68a0f7b]
// 维护导航：turnUserKey；下列为本函数源码中的接口与路径索引。
// 输入 runId：number。
// 返回契约：string。
// 返回路径：`run:${runId}:user`。
// [/maint-note:3f62c68a0f7b]
function turnUserKey(runId: number): string {
  return `run:${runId}:user`
}

// [maint-note:cc50d1ba2095]
// 维护导航：formatToolOutput；下列为本函数源码中的接口与路径索引。
// 输入 output：string。
// 返回契约：string。
// 分支条件：bounded.length <= MAX_TOOL_DISPLAY_CHARS。
// 返回路径：bounded。
// 返回路径：`${bounded.slice(0, MAX_TOOL_DISPLAY_CHARS - 32)}… [output truncated]`。
// 直接调用：bounded.slice。
// [/maint-note:cc50d1ba2095]
export function formatToolOutput(output: string): string {
  const bounded = output || 'completed'
  if (bounded.length <= MAX_TOOL_DISPLAY_CHARS) return bounded
  return `${bounded.slice(0, MAX_TOOL_DISPLAY_CHARS - 32)}… [output truncated]`
}

const MAX_TOOL_DISPLAY_CHARS = 1_200
