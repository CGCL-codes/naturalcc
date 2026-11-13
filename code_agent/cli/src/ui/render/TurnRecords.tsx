import { Box } from 'ink'
import type { Message } from '../types/message.js'
import { MessageColumn } from '../components/MessageColumn.js'
import { TurnSeparator } from '../components/TimingProjection.js'

// [maint-note:8523aeae5bf1]
// 数据契约导航：TurnRecordsProps；字段含义以消费端校验为准。
// 成员 messages：readonly Message[]。
// 成员 width：number。
// 成员 separator：'after' | 'none'。
// [/maint-note:8523aeae5bf1]
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
// [maint-note:a38fd2c6d485]
// 维护导航：TurnRecords；下列为本函数源码中的接口与路径索引。
// 输入 { messages, width, separator }：TurnRecordsProps。
// 返回路径：( <Box flexDirection="column" width={Math.max(1, width)}> {messages.map((message) => ( <MessageColumn key={message.id} message={message} width={width} /> ))} {separator === 'after' && <TurnSeparator width={width} />} </Box> )。
// 直接调用：Math.max。
// 直接调用：messages.map。
// [/maint-note:a38fd2c6d485]
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
