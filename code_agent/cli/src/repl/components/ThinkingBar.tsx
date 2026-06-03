import { Box, Text } from 'ink'
import type React from 'react'
import { now } from '../utils.js'

interface ThinkingBarProps {
  loading: boolean
  isStreaming: React.RefObject<boolean>
  streamingContent: string
  thinkTimeRef: React.RefObject<number | null>
  elapsed: number
  dots: string
  width: number
}

export function ThinkingBar({ loading, isStreaming, streamingContent, thinkTimeRef, elapsed, dots, width }: ThinkingBarProps) {
  if (!loading) return null

  return (
    <>
      {!isStreaming.current && (
        <Box flexDirection="column" marginBottom={1}>
          <Box flexDirection="row">
            <Text color="blue">{'● '}</Text>
            <Text color='gray'>{now()}</Text>
          </Box>
          <Text>thinking for <Text color='yellow'>{(thinkTimeRef.current ?? elapsed).toFixed(1)}</Text> s {dots}</Text>
        </Box>
      )}
      {isStreaming.current && (
        <Box flexDirection="column" marginBottom={1}>
          <Box flexDirection="row">
            <Text color="blue">{'● '}</Text>
            <Text color='gray'>{now()}</Text>
          </Box>
          <Text>{streamingContent}</Text>
          <Text>{"\n"}work for <Text color='yellow'>{(thinkTimeRef.current ?? elapsed).toFixed(1)}</Text> s</Text>
        </Box>
      )}
      <Box flexDirection="column">
        <Text>{'─'.repeat(width - 2)}</Text>
        <Text color='gray'>Esc to interrupted</Text>
      </Box>
    </>
  )
}
