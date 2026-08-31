import { Box, Text } from 'ink'
import { VERSION } from '../../version.js'

interface HeaderProps {
  width: number
}

export function Header({ width }: HeaderProps) {
  const outerWidth = Math.max(1, width - 2)
  if (width < 40) {
    return (
      <Box borderStyle="double" width={outerWidth}>
        <Text bold>naturalcc CLI v{VERSION}</Text>
      </Box>
    )
  }

  const artWidth = 35
  const infoWidth = Math.max(1, outerWidth - artWidth)

  return (
    <Box flexDirection="row" borderStyle="double" width={outerWidth}>
      <Box flexDirection='column' width={artWidth}>
        <Text color="#8B4513">{"      |\\      _,,,---,,_"}</Text>
        <Text color="#8B4513">{" ZZZzz /,`.-'`'    -.  ;-;;,_"}</Text>
        <Text color="#8B4513">{"      |,4-  ) )-,_. ,\\ (  `'-'"}</Text>
        <Text color="#8B4513">{"       '---''(_/--'  `-'\\_) "}</Text>
      </Box>
      <Box flexDirection="column" width={infoWidth}>
        <Text bold>Welcome to naturalcc CLI</Text>
        <Text>REPL mode{'\x1b[0K'}</Text>
        <Text>v{VERSION}</Text>
        <Text>a code agent</Text>
      </Box>
    </Box>
  )
}
