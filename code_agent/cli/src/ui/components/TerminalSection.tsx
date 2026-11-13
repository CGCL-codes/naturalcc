import { Box, Text } from 'ink'
import { sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'

/** A business-agnostic section consumed by terminal presentation components. */
// [maint-note:38626767b435]
// 数据契约导航：TerminalSectionData；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 title：string。
// 成员 body：string。
// [/maint-note:38626767b435]
export interface TerminalSectionData {
  readonly id: string
  readonly title: string
  readonly body: string
}

const MAX_SECTION_TEXT_CHARS = 2400

// [maint-note:5d7a278b6926]
// 维护导航：TerminalSectionStack；下列为本函数源码中的接口与路径索引。
// 输入 { sections, width, }：{ sections: readonly TerminalSectionData[] width: number }。
// 返回路径：( <Box flexDirection="column" width={width}> {sections.map((section, index) => ( <Box flexDirection="column" width={width} key={section.id}> {index > 0 && <Box height={1} />} <TerminalSection section={section} width={width} /> </Box> ))} </。
// 直接调用：sections.map。
// [/maint-note:5d7a278b6926]
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

// [maint-note:397a2012e159]
// 维护导航：TerminalSection；下列为本函数源码中的接口与路径索引。
// 输入 { section, width }：{ section: TerminalSectionData; width: number }。
// 返回路径：( <Box flexDirection="column" width={width}> {titleLines.map((line, index) => <Text color="blue" key={`title-${index}`}>{line}</Text>)} {bodyLines.map((line, index) => <Text key={`body-${index}`}>{line}</Text>)} </Box> )。
// 直接调用：wrapSectionText。
// 直接调用：titleLines.map。
// 直接调用：bodyLines.map。
// [/maint-note:397a2012e159]
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

// [maint-note:c56c1fc3319c]
// 维护导航：wrapSectionText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 width：number。
// 返回契约：string[]。
// 返回路径：lines.length > 0 ? lines : ['']。
// 直接调用：[...sanitizeTerminalText(value)].slice(0, MAX_SECTION_TEXT_CHARS).join。
// 直接调用：[...sanitizeTerminalText(value)].slice。
// 直接调用：sanitizeTerminalText。
// 直接调用：wrapDisplayLines。
// [/maint-note:c56c1fc3319c]
function wrapSectionText(value: string, width: number): string[] {
  const safe = [...sanitizeTerminalText(value)].slice(0, MAX_SECTION_TEXT_CHARS).join('')
  const lines = wrapDisplayLines(safe, width, { hard: true, trim: false, wordWrap: true })
  return lines.length > 0 ? lines : ['']
}
