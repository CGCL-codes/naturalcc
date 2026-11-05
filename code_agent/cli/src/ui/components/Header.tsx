import { Box, Text } from 'ink'
import { VERSION } from '../../version.js'
import { displayLineWidth, fillDisplayWidth, normalizeTerminalWidth, truncateDisplayText, wrapDisplayLines } from '../terminal/index.js'

export type HeaderProjectionVariant = 'tiny' | 'compact' | 'wide'

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

export function measureHeaderRows(width: number): number {
  return buildHeaderProjection(width).physicalRows
}

/**
 * Return the rows the matching Header JSX emits before a real terminal wraps
 * them. Keeping this beside the JSX projection lets the frame ledger account
 * for a frozen wide Header after a resize without constructing a second,
 * compact Header for the new width.
 */
export function renderHeaderProjectionRows(projection: HeaderProjection): readonly string[] {
  if (projection.variant === 'tiny') return projection.contentRows

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
export function measureFrozenHeaderRows(projection: HeaderProjection, terminalWidth: number): number {
  const width = normalizeTerminalWidth(terminalWidth, 1)
  return projection.renderedRows.reduce((total, row) => total + Math.max(1, wrapDisplayLines(row, width, {
    hard: true,
    trim: false,
    wordWrap: false,
  }).length), 0)
}

interface HeaderProps {
  width: number
}

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
