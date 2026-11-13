import { Box, Text } from 'ink'

import { UI_PALETTE } from '../theme.js'
import { sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'

/**
 * shared "user input surface" renderer for committed user messages.
 *
 * One component serves every consumer (MessageColumn is shared by the active
 * region, completed-prefix handoff and final Static; MessageList is the
 * legacy entry), so a given message renders with identical colors, wrapping
 * and width before and after it freezes.
 *
 * Geometry contract: the wrapped text projection is byte-for-byte
 * the previous one — `> ${content}` sanitized and hard-wrapped at the same
 * content width — so row counts and wrap thresholds cannot change. Only the
 * presentation changes: each body line sits in a row Box of the message's
 * safe drawable width filled with the surface background (the fill is UI
 * padding from Ink's box layout, never spaces written into the message), the
 * framework-generated `> ` prompt keeps green, and the body uses the paired
 * light foreground instead of the old whole-first-line green. Timestamps,
 * message margins and other roles stay outside the surface.
 */
// [maint-note:73a2f4040d5c]
// 维护导航：UserBodyLines；下列为本函数源码中的接口与路径索引。
// 输入 { content, width }：{ content: string; width: number }。
// 返回路径：( <Box flexDirection="column"> {lines.map((line, index) => ( // height=1 pins one physical row per wrapped line: an internal blank // line of the user's content stays a surfaced body row instead of // collapsing, and no row can ever wrap di。
// 直接调用：Math.max。
// 直接调用：wrapDisplayLines。
// 直接调用：sanitizeTerminalText。
// 直接调用：lines.map。
// [/maint-note:73a2f4040d5c]
export function UserBodyLines({ content, width }: { content: string; width: number }) {
  const safeWidth = Math.max(1, width)
  const lines = wrapDisplayLines(`> ${sanitizeTerminalText(content)}`, safeWidth, { hard: true, wordWrap: false })
  return (
    <Box flexDirection="column">
      {lines.map((line, index) => (
        // height=1 pins one physical row per wrapped line: an internal blank
        // line of the user's content stays a surfaced body row instead of
        // collapsing, and no row can ever wrap differently than projected.
        <Box key={index} width={safeWidth} height={1} backgroundColor={UI_PALETTE.userSurfaceBackground}>
          {index === 0 ? (
            <>
              <Text color="green">{'> '}</Text>
              <Text color={UI_PALETTE.userSurfaceForeground}>{line.slice(2)}</Text>
            </>
          ) : (
            <Text color={UI_PALETTE.userSurfaceForeground}>{line}</Text>
          )}
        </Box>
      ))}
    </Box>
  )
}
