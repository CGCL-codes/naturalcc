import { Box } from 'ink'
import type { Message } from '../types/message.js'
import { MessageColumn } from '../components/MessageColumn.js'
import { TurnSeparator } from '../components/TimingProjection.js'

export interface TurnRecordsProps {
  messages: readonly Message[]
  width: number
  /** Whether a pure separator boundary follows this group. */
  separator: 'after' | 'none'
}

/**
 * Shared immutable-record projection used by both the stable region and the
 * active-turn region, so a record renders identically before and after it
 * freezes.
 */
export function TurnRecords({ messages, width, separator }: TurnRecordsProps) {
  return (
    <Box flexDirection="column" width={Math.max(1, width)}>
      {messages.map((message) => (
        <MessageColumn key={message.id} message={message} width={width} />
      ))}
      {separator === 'after' && <TurnSeparator width={width} />}
    </Box>
  )
}
