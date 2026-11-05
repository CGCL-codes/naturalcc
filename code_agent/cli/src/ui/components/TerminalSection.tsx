import { Box, Text } from 'ink'
import { sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'

/** A business-agnostic section consumed by terminal presentation components. */
export interface TerminalSectionData {
  readonly id: string
  readonly title: string
  readonly body: string
}

const MAX_SECTION_TEXT_CHARS = 2400

export function TerminalSectionStack({
  sections,
  width,
}: {
  sections: readonly TerminalSectionData[]
  width: number
}) {
  return (
    <Box flexDirection="column" width={width}>
      {sections.map((section, index) => (
        <Box flexDirection="column" width={width} key={section.id}>
          {index > 0 && <Box height={1} />}
          <TerminalSection section={section} width={width} />
        </Box>
      ))}
    </Box>
  )
}

function TerminalSection({ section, width }: { section: TerminalSectionData; width: number }) {
  const titleLines = wrapSectionText(`${section.title}`, width)
  const bodyLines = wrapSectionText(section.body, width)
  return (
    <Box flexDirection="column" width={width}>
      {titleLines.map((line, index) => <Text color="blue" key={`title-${index}`}>{line}</Text>)}
      {bodyLines.map((line, index) => <Text key={`body-${index}`}>{line}</Text>)}
    </Box>
  )
}

function wrapSectionText(value: string, width: number): string[] {
  const safe = [...sanitizeTerminalText(value)].slice(0, MAX_SECTION_TEXT_CHARS).join('')
  const lines = wrapDisplayLines(safe, width, { hard: true, trim: false, wordWrap: true })
  return lines.length > 0 ? lines : ['']
}
