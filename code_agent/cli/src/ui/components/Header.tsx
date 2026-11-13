import { Box, Text } from 'ink'
import { VERSION } from '../../version.js'
import { displayLineWidth, fillDisplayWidth, normalizeTerminalWidth, truncateDisplayText, wrapDisplayLines } from '../terminal/index.js'

// [maint-note:f622bb880e63]
// 数据契约导航：HeaderProjectionVariant；字段含义以消费端校验为准。
// 类型组合：'tiny' | 'compact' | 'wide'。
// [/maint-note:f622bb880e63]
export type HeaderProjectionVariant = 'tiny' | 'compact' | 'wide'

// [maint-note:6d124746f29b]
// 数据契约导航：HeaderProjection；字段含义以消费端校验为准。
// 成员 width：number。
// 成员 variant：HeaderProjectionVariant。
// 成员 physicalRows：number。
// 成员 outerWidth：number。
// 成员 contentRows：readonly string[]。
// 成员 artRows：readonly string[]。
// 成员 infoRows：readonly string[]。
// 成员 renderedRows：readonly string[]。
// [/maint-note:6d124746f29b]
export interface HeaderProjection {
  readonly width: number
  readonly variant: HeaderProjectionVariant
  /** Physical rows occupied by the actual Header component, including borders. */
  readonly physicalRows: number
  readonly outerWidth: number
  readonly contentRows: readonly string[]
  readonly artRows: readonly string[]
  readonly infoRows: readonly string[]
  /** Exact plain rows emitted by the Header projection before terminal wrap. */
  readonly renderedRows: readonly string[]
}

const HEADER_ART_ROWS = [
  '      |\\      _,,,---,,_',
  " ZZZzz /,`.-'`'    -.  ;-;;,_",
  '      |,4-  ) )-,_. ,\\ (  `\'-\'',
  "       '---''(_/--'  `-'\\_) ",
] as const

const HEADER_INFO_ROWS = [
  'Welcome to naturalcc CLI',
  'REPL mode',
  `v${VERSION}`,
  'a code agent',
] as const

/**
 * The one Header layout fact consumed by both JSX and the frame ledger.
 * `physicalRows` is derived from the same variant/rows that Header renders;
 * callers must not re-create an estimate from a magic number.
 */
// [maint-note:2594904aade9]
// 维护导航：buildHeaderProjection；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 返回契约：HeaderProjection。
// 分支条件：safeWidth <= 6。
// 分支条件：safeWidth < 80。
// 返回路径：{ ...projection, renderedRows: renderHeaderProjectionRows(projection) }。
// 直接调用：normalizeTerminalWidth。
// 直接调用：truncateDisplayText。
// 直接调用：renderHeaderProjectionRows。
// 直接调用：Math.max。
// [/maint-note:2594904aade9]
export function buildHeaderProjection(width: number): HeaderProjection {
  const safeWidth = normalizeTerminalWidth(width, 1)
  if (safeWidth <= 6) {
    const projection: HeaderProjection = {
      width: safeWidth,
      variant: 'tiny',
      physicalRows: 1,
      outerWidth: safeWidth,
      contentRows: [truncateDisplayText(`naturalcc v${VERSION}`, safeWidth)],
      artRows: [],
      infoRows: [],
      renderedRows: [],
    }
    return { ...projection, renderedRows: renderHeaderProjectionRows(projection) }
  }

  const outerWidth = Math.max(1, safeWidth - 2)
  if (safeWidth < 80) {
    const projection: HeaderProjection = {
      width: safeWidth,
      variant: 'compact',
      // One content row plus the top and bottom double borders.
      physicalRows: 3,
      outerWidth,
      contentRows: [truncateDisplayText(`naturalcc CLI v${VERSION}`, Math.max(1, safeWidth - 4))],
      artRows: [],
      infoRows: [],
      renderedRows: [],
    }
    return { ...projection, renderedRows: renderHeaderProjectionRows(projection) }
  }

  const projection: HeaderProjection = {
    width: safeWidth,
    variant: 'wide',
    // Four content rows plus the top and bottom double borders.
    physicalRows: HEADER_ART_ROWS.length + 2,
    outerWidth,
    contentRows: [],
    artRows: HEADER_ART_ROWS,
    infoRows: HEADER_INFO_ROWS,
    renderedRows: [],
  }
  return { ...projection, renderedRows: renderHeaderProjectionRows(projection) }
}

// [maint-note:dfbcac551238]
// 维护导航：measureHeaderRows；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 返回契约：number。
// 返回路径：buildHeaderProjection(width).physicalRows。
// 直接调用：buildHeaderProjection。
// [/maint-note:dfbcac551238]
export function measureHeaderRows(width: number): number {
  return buildHeaderProjection(width).physicalRows
}

/**
 * Return the rows the matching Header JSX emits before a real terminal wraps
 * them. Keeping this beside the JSX projection lets the frame ledger account
 * for a frozen wide Header after a resize without constructing a second,
 * compact Header for the new width.
 */
// [maint-note:9e30f25f7d82]
// 维护导航：renderHeaderProjectionRows；下列为本函数源码中的接口与路径索引。
// 输入 projection：HeaderProjection。
// 返回契约：readonly string[]。
// 分支条件：projection.variant === 'tiny'。
// 分支条件：projection.variant === 'compact'。
// 返回路径：projection.contentRows。
// 返回路径：[ border('╔', '╗'), `║${content}${fillDisplayWidth(' ', Math.max(0, projection.outerWidth - 2 - displayLineWidth(content)))}║`, border('╚', '╝'), ]。
// 返回路径：[border('╔', '╗'), ...contentRows, border('╚', '╝')]。
// 直接调用：border。
// 直接调用：fillDisplayWidth。
// 直接调用：Math.max。
// 直接调用：displayLineWidth。
// 直接调用：wideHeaderColumnWidths。
// 直接调用：projection.artRows.map。
// [/maint-note:9e30f25f7d82]
export function renderHeaderProjectionRows(projection: HeaderProjection): readonly string[] {
  if (projection.variant === 'tiny') return projection.contentRows

  // [maint-note:dda319212830]
  // 维护导航：border；下列为本函数源码中的接口与路径索引。
  // 输入 left：string。
  // 输入 right：string。
  // 直接调用：fillDisplayWidth。
  // 直接调用：Math.max。
  // [/maint-note:dda319212830]
  const border = (left: string, right: string) =>
    `${left}${fillDisplayWidth('═', Math.max(0, projection.outerWidth - 2))}${right}`
  if (projection.variant === 'compact') {
    const content = projection.contentRows[0] ?? ''
    return [
      border('╔', '╗'),
      `║${content}${fillDisplayWidth(' ', Math.max(0, projection.outerWidth - 2 - displayLineWidth(content)))}║`,
      border('╚', '╝'),
    ]
  }

  const { artWidth, infoWidth } = wideHeaderColumnWidths(projection)
  const contentRows = projection.artRows.map((art, index) => {
    const info = projection.infoRows[index] ?? ''
    return `║${art}${fillDisplayWidth(' ', Math.max(0, artWidth - displayLineWidth(art)))}${info}${fillDisplayWidth(' ', Math.max(0, infoWidth - displayLineWidth(info)))}║`
  })
  return [border('╔', '╗'), ...contentRows, border('╚', '╝')]
}

// [maint-note:a614481cafd1]
// 维护导航：wideHeaderColumnWidths；下列为本函数源码中的接口与路径索引。
// 输入 projection：HeaderProjection。
// 返回契约：{ artWidth: number; infoWidth: number }。
// 返回路径：{ artWidth, infoWidth: Math.max(1, innerWidth - artWidth) }。
// 直接调用：Math.max。
// 直接调用：Math.min。
// [/maint-note:a614481cafd1]
function wideHeaderColumnWidths(projection: HeaderProjection): { artWidth: number; infoWidth: number } {
  const innerWidth = Math.max(1, projection.outerWidth - 2)
  const artWidth = Math.min(35, innerWidth)
  return { artWidth, infoWidth: Math.max(1, innerWidth - artWidth) }
}

/**
 * Measure a static Header owned by `projection` at the current terminal
 * width. The source rows remain frozen; only their physical wrapping changes
 * with the terminal. This is intentionally different from measureHeaderRows,
 * which measures a newly constructed Header at its own width.
 */
// [maint-note:0a0a786e6c78]
// 维护导航：measureFrozenHeaderRows；下列为本函数源码中的接口与路径索引。
// 输入 projection：HeaderProjection。
// 输入 terminalWidth：number。
// 返回契约：number。
// 返回路径：projection.renderedRows.reduce((total, row) => total + Math.max(1, wrapDisplayLines(row, width, { hard: true, trim: false, wordWrap: false, }).length), 0)。
// 直接调用：normalizeTerminalWidth。
// 直接调用：projection.renderedRows.reduce。
// [/maint-note:0a0a786e6c78]
export function measureFrozenHeaderRows(projection: HeaderProjection, terminalWidth: number): number {
  const width = normalizeTerminalWidth(terminalWidth, 1)
  return projection.renderedRows.reduce((total, row) => total + Math.max(1, wrapDisplayLines(row, width, {
    hard: true,
    trim: false,
    wordWrap: false,
  }).length), 0)
}

// [maint-note:1b3aff0a6fab]
// 数据契约导航：HeaderProps；字段含义以消费端校验为准。
// 成员 width：number。
// [/maint-note:1b3aff0a6fab]
interface HeaderProps {
  width: number
}

// [maint-note:c0b6807cef2c]
// 维护导航：Header；下列为本函数源码中的接口与路径索引。
// 输入 { width }：HeaderProps。
// 分支条件：projection.variant === 'tiny'。
// 分支条件：projection.variant === 'compact'。
// 返回路径：( <Box width={projection.outerWidth}> <Text>{projection.contentRows[0]}</Text> </Box> )。
// 返回路径：( <Box borderStyle="double" width={projection.outerWidth}> <Text bold>{projection.contentRows[0]}</Text> </Box> )。
// 返回路径：( <Box flexDirection="row" borderStyle="double" width={projection.outerWidth}> <Box flexDirection='column' width={artWidth}> {projection.artRows.map((row) => <Text key={row} color="#8B4513">{row}</Text>)} </Box> <Box flexDirection="column" 。
// 直接调用：buildHeaderProjection。
// 直接调用：wideHeaderColumnWidths。
// 直接调用：projection.artRows.map。
// 直接调用：projection.infoRows.map。
// [/maint-note:c0b6807cef2c]
export function Header({ width }: HeaderProps) {
  const projection = buildHeaderProjection(width)
  if (projection.variant === 'tiny') {
    return (
      <Box width={projection.outerWidth}>
        <Text>{projection.contentRows[0]}</Text>
      </Box>
    )
  }
  if (projection.variant === 'compact') {
    return (
      <Box borderStyle="double" width={projection.outerWidth}>
        <Text bold>{projection.contentRows[0]}</Text>
      </Box>
    )
  }

  const { artWidth, infoWidth } = wideHeaderColumnWidths(projection)

  return (
    <Box flexDirection="row" borderStyle="double" width={projection.outerWidth}>
      <Box flexDirection='column' width={artWidth}>
        {projection.artRows.map((row) => <Text key={row} color="#8B4513">{row}</Text>)}
      </Box>
      <Box flexDirection="column" width={infoWidth}>
        {projection.infoRows.map((row, index) => <Text key={row} bold={index === 0}>{row}</Text>)}
      </Box>
    </Box>
  )
}
