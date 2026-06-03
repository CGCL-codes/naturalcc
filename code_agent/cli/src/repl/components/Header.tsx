import { Box, Text } from 'ink'

interface HeaderProps {
  width: number
}

export function Header({ width }: HeaderProps) {
  return (
    <Box flexDirection="row" borderStyle="double" width={width - 2}>
      <Box flexDirection='column' width={35}>
        <Text color="#8B4513">{"      |\\      _,,,---,,_"}</Text>
        <Text color="#8B4513">{" ZZZzz /,`.-'`'    -.  ;-;;,_"}</Text>
        <Text color="#8B4513">{"      |,4-  ) )-,_. ,\\ (  `'-'"}</Text>
        <Text color="#8B4513">{"       '---''(_/--'  `-'\\_) "}</Text>
      </Box>
      <Box flexDirection="column" width={width - 35}>
        <Text bold>naturalcc CLI</Text>
        <Text>REPL mode{'\x1b[0K'}</Text>
        <Text>v0.0.0</Text>
        <Text>a code agent</Text>
      </Box>
    </Box>
  )
}
