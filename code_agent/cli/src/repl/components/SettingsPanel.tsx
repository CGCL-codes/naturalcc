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
  feature: string
  featureConfig: Record<string, unknown>
}

function maskSecret(secret: string | null): string {
  const value = secret?.trim()
  if (!value) return '-'
  if (value.length <= 7) return `${value.slice(0,3)}***`
  return `${value.slice(0,3)}***${value.slice(-4)}`
}

export function SettingsPanel({ visible, files, model, apiKey, projectDir, symbol, completionType, prefix, preview, feature, featureConfig }: SettingsPanelProps) {
  if (!visible) return null
  const configText = Object.keys(featureConfig).length > 0 ? JSON.stringify(featureConfig) : '-'

  return (
    <Box flexDirection='column' borderStyle='round'>
      <Text color="green">files:{files.length > 0 ? files.join(' ') : '-'}</Text>
      <Text color="green">model:{model.trim() ? model : '-'}</Text>
      <Text color="green">apiKey:{maskSecret(apiKey)}</Text>
      <Text color="green">projectDir:{projectDir.trim() ? projectDir : '-'}</Text>
      <Text color="green">symbol:{symbol ? symbol : '-'}</Text>
      <Text color="green">completionType:{completionType ? completionType : '-'}</Text>
      <Text color="green">prefix:{prefix.trim() ? prefix : '-'}</Text>
      <Text color="green">feature:{feature.trim() ? feature : '-'}</Text>
      <Text color="green">featureConfig:{configText}</Text>
      <Text color="green">preview:{preview ? 'True' : 'False'}</Text>
    </Box>
  )
}
