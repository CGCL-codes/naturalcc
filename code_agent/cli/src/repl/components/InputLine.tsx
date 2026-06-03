import { Box, Text } from 'ink'
import TextInput from 'ink-text-input'

interface InputLineProps {
  input: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
}

export function InputLine({ input, onChange, onSubmit }: InputLineProps) {
  return (
    <Box>
      <Text color="green">{'> '}</Text>
      <TextInput
        value={input}
        onChange={onChange}
        onSubmit={onSubmit}
        placeholder="Enter command..."
      />
    </Box>
  )
}
