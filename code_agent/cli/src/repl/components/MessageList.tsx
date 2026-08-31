import { Box, Text } from 'ink'
import type { Message } from '../../types/message.js'

interface MessageListProps {
  messages: Message[]
}

export function MessageList({ messages }: MessageListProps) {
  return (
    <>
      {messages.map((msg, i) => (
        <Box key={i} flexDirection="column" marginBottom={1}>
          {msg.role === 'user' && (
            <Box flexDirection="row">
              <Text color='green'>{'> '}</Text>
              <Text color='gray'>{msg.time} </Text>
              <Text>{msg.content}</Text>
            </Box>
          )}
          {msg.role === 'assistant' && (
            <Box flexDirection="column">
              <Box flexDirection="row">
                <Text color='blue'>{'● '}</Text>
                <Text color='gray'>{msg.time}</Text>
              </Box>
              <Text>{msg.content}</Text>
            </Box>
          )}
          {msg.role === 'error' && (
            <Box flexDirection="row">
              <Text color='red'>{'⚠ '}</Text>
              <Text color='gray'>{msg.time} </Text>
              <Text color='red'>{msg.content}</Text>
            </Box>
          )}
        </Box>
      ))}
    </>
  )
}
