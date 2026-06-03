import { Box, Text } from 'ink'

interface SettingsPanelProps {
  visible: boolean
  files: string[]
  model: string
  apiKey: string | null
  projectDir: string
  symbol: string | null
  completionType: string | null
  prefix: string
  preview: boolean
}

export function SettingsPanel({ visible, files, model, apiKey, projectDir, symbol, completionType, prefix, preview }: SettingsPanelProps) {
  if (!visible) return null

  return (
    <Box flexDirection='column' borderStyle='round'>
      <Text color="green">files:{files.length > 0 ? files.join(' ') : '-'}</Text>
      <Text color="green">model:{model.trim() ? model : '-'}</Text>
      <Text color="green">apiKey:{apiKey ? apiKey : '-'}</Text>
      <Text color="green">projectDir:{projectDir.trim() ? projectDir : '-'}</Text>
      <Text color="green">symbol:{symbol ? symbol : '-'}</Text>
      <Text color="green">completionType:{completionType ? completionType : '-'}</Text>
      <Text color="green">prefix:{prefix.trim() ? prefix : '-'}</Text>
      <Text color="green">preview:{preview ? 'True' : 'False'}</Text>
    </Box>
  )
}
