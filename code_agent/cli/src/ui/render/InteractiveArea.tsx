import { Box, Text } from 'ink'
import { InputLine, projectInputLine } from '../components/Input.js'
import type { InputLineHandle } from '../components/Input.js'
import { renderSessionFooter, SessionFooter } from '../components/TimingProjection.js'
import {
  DEFAULT_SESSION_STATUS,
  projectSessionStatus,
  projectSessionStatusWarning,
  SessionStatusSlot,
} from '../components/SessionStatusSlot.js'
import type { SessionStatusMetadata } from '../components/SessionStatusSlot.js'
import type { SessionFooterDescriptor } from '../state/sessionTiming.js'
import type { ReplCtrlCAction } from '../state/replCtrlC.js'
import type { RefObject } from 'react'
import { calculateContentWidth, fillDisplayWidth, sanitizeTerminalText, truncateDisplayText } from '../terminal/index.js'

const HELP_HINT = 'enter /help for help'

// [maint-note:a8169764cc2a]
// 维护导航：projectHelpHint；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 返回契约：string。
// 返回路径：truncateDisplayText(HELP_HINT, Math.max(1, width))。
// 直接调用：truncateDisplayText。
// 直接调用：Math.max。
// [/maint-note:a8169764cc2a]
function projectHelpHint(width: number): string {
  return truncateDisplayText(HELP_HINT, Math.max(1, width))
}

// [maint-note:8689103226c8]
// 数据契约导航：InteractiveAreaProps；字段含义以消费端校验为准。
// 成员 footer：SessionFooterDescriptor | undefined。
// 成员 inputVisible：boolean。
// 成员 showHelpHint（可省略）：boolean。
// 成员 keepInputMounted（可省略）：boolean。
// 成员 input：string。
// 成员 onInputChange：(value: string) => void。
// 成员 onSubmit：(value: string) => void。
// 成员 placeholder：string。
// 成员 onDropdownChange：(open: boolean) => void。
// 成员 resetGeneration：number。
// 成员 onControlC：(inputLength: number, exitArmed: boolean) => ReplCtrlCAction。
// 成员 exitArmed：boolean。
// 成员 clearing：boolean。
// 成员 width：number。
// 成员 onInputSubscriptionChange（可省略）：(attached: boolean) => void。
// 成员 initialEditingState（可省略）：{ value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number }。
// 成员 inputRef（可省略）：RefObject<InputLineHandle | null>。
// 成员 statusMetadata（可省略）：SessionStatusMetadata。
// 成员 resizeGeneration（可省略）：number。
// [/maint-note:8689103226c8]
export interface InteractiveAreaProps {
  footer: SessionFooterDescriptor | undefined
  inputVisible: boolean
  /** Current session generation has not accepted its first run yet. */
  showHelpHint?: boolean
  /** Keep InputLine's editing state alive behind a transient overlay. */
  keepInputMounted?: boolean
  input: string
  onInputChange: (value: string) => void
  onSubmit: (value: string) => void
  placeholder: string
  onDropdownChange: (open: boolean) => void
  resetGeneration: number
  onControlC: (inputLength: number, exitArmed: boolean) => ReplCtrlCAction
  exitArmed: boolean
  clearing: boolean
  width: number
  onInputSubscriptionChange?: (attached: boolean) => void
  /** editing state restored when the line remounts after the viewer. */
  initialEditingState?: { value: string; cursor: number; dropdownOpen: boolean; selectedIndex: number }
  inputRef?: RefObject<InputLineHandle | null>
  statusMetadata?: SessionStatusMetadata
  /** Internal generation used to force a complete incremental-frame rewrite. */
  resizeGeneration?: number
}

/**
 * Text-only projection used by the resize handoff to measure the previous
 * mutable frame after the terminal reflows it. It deliberately mirrors only
 * the interactive chrome; startup resize is disabled once a turn or a batch
 * exists, so no model/tool content is guessed here.
 */
// [maint-note:c1a1be624ff5]
// 维护导航：projectInteractiveAreaForResize；下列为本函数源码中的接口与路径索引。
// 输入 input：Pick<InteractiveAreaProps, 'footer' | 'inputVisible' | 'showHelpHint' | 'input' | 'placeholder' | 'clearing' | 'exitArmed' | 'width' | 'statusMetadata'>。
// 返回契约：string。
// 分支条件：input.footer。
// 分支条件：input.clearing。
// 分支条件：input.inputVisible。
// 分支条件：input.showHelpHint !== false && input.width > 4。
// 分支条件：input.width <= 4。
// 返回路径：rows.join('\n')。
// 直接调用：rows.push。
// 直接调用：renderSessionFooter。
// 直接调用：truncateDisplayText。
// 直接调用：Math.max。
// 直接调用：projectHelpHint。
// 直接调用：sanitizeTerminalText。
// 直接调用：calculateContentWidth。
// 直接调用：projectInputLine。
// 直接调用：fillDisplayWidth。
// 直接调用：projectSessionStatusWarning。
// 直接调用：projectSessionStatus。
// 直接调用：rows.join。
// [/maint-note:c1a1be624ff5]
export function projectInteractiveAreaForResize(input: Pick<InteractiveAreaProps, 'footer' | 'inputVisible' | 'showHelpHint' | 'input' | 'placeholder' | 'clearing' | 'exitArmed' | 'width' | 'statusMetadata'>): string {
  const rows: string[] = []
  if (input.footer) rows.push(renderSessionFooter(input.footer.durationMs, input.width))
  if (input.clearing) rows.push(truncateDisplayText('clearing conversation...', Math.max(1, input.width)))
  if (input.inputVisible) {
    if (input.showHelpHint !== false && input.width > 4) rows.push(projectHelpHint(input.width))
    if (input.width <= 4) {
      rows.push(truncateDisplayText(sanitizeTerminalText(input.input || input.placeholder), Math.max(1, input.width)))
    } else {
      const dividerWidth = calculateContentWidth({ terminalWidth: input.width, paddingLeft: 2, minWidth: 0 })
      const projection = projectInputLine(input.input, input.input.length, input.width, input.placeholder)
      rows.push(fillDisplayWidth('─', dividerWidth))
      rows.push(`> ${projection.before}${projection.at}${projection.after}`)
      rows.push(fillDisplayWidth('─', dividerWidth))
    }
  }
  rows.push(input.exitArmed
    ? projectSessionStatusWarning(input.width)
    : projectSessionStatus(input.statusMetadata ?? DEFAULT_SESSION_STATUS, input.width))
  return rows.join('\n')
}

/**
 * interactive region: the single session footer (idle only, one footer ID
 * at the very bottom), the command input, and the fixed session status slot.
 * The footer/status never live inside a frozen turn batch, so they update in
 * place without adding another stable history owner.
 */
// [maint-note:dbc19d431600]
// 维护导航：InteractiveArea；下列为本函数源码中的接口与路径索引。
// 输入 { footer, inputVisible, showHelpHint = true, input, onInputChange, onSubmit, placeholder, onDropdownChange, resetGeneration, onControlC, exitArmed, clearing, width, onInputSubscriptionChange, initialEditingState, inputRef, statusMetadata = ：InteractiveAreaProps。
// 返回路径：( <Box flexDirection="column" width={Math.max(1, width)}> {/* The footer follows the session timing owner, not input visibility: it appears the moment the run's timing closes (which may precede the executor's iterator drain) and hides as so。
// 直接调用：Math.max。
// 直接调用：truncateDisplayText。
// 直接调用：projectHelpHint。
// [/maint-note:dbc19d431600]
export function InteractiveArea({
  footer, inputVisible, showHelpHint = true, input, onInputChange, onSubmit, placeholder,
  onDropdownChange, resetGeneration, onControlC, exitArmed, clearing, width,
  onInputSubscriptionChange, initialEditingState,
  inputRef,
  statusMetadata = DEFAULT_SESSION_STATUS,
  resizeGeneration = 0,
  keepInputMounted = false,
}: InteractiveAreaProps) {
  const frameMarker = resizeGeneration > 0
    ? `\u001b[${30 + (resizeGeneration % 8)}m\u001b[39m`
    : ''
  return (
    <Box flexDirection="column" width={Math.max(1, width)}>
      {/* The footer follows the session timing owner, not input visibility: it
          appears the moment the run's timing closes (which may precede the
          executor's iterator drain) and hides as soon as a new run begins. */}
      {footer && <SessionFooter footer={footer} width={width} />}
      {clearing && <Text color="gray">{frameMarker}{truncateDisplayText('clearing conversation...', Math.max(1, width))}</Text>}
      {(inputVisible || keepInputMounted) && (
        <Box flexDirection="column">
          {inputVisible && showHelpHint && width > 4 && <Text color="gray">{frameMarker}{projectHelpHint(width)}</Text>}
          <InputLine
            ref={inputRef}
            value={input}
            onChange={onInputChange}
            onSubmit={onSubmit}
            placeholder={placeholder}
            onDropdownChange={onDropdownChange}
            resetGeneration={resetGeneration}
            onControlC={onControlC}
            exitArmed={exitArmed}
            onInputSubscriptionChange={onInputSubscriptionChange}
            initialEditingState={initialEditingState}
            renderMarker={frameMarker}
            hidden={!inputVisible}
            width={width} />
        </Box>
      )}
      {/* One fixed bottom row: warning is an in-place replacement, never an
          additional overlay. It remains visible while input is hidden. */}
      <SessionStatusSlot metadata={statusMetadata} exitArmed={exitArmed} width={width} renderMarker={frameMarker} />
    </Box>
  )
}
