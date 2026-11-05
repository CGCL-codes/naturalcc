const ALWAYS_EXCLUDED_GLOBS = [
  '!.git',
  '!.git/**',
  '!**/.git',
  '!**/.git/**',
]

const HEAVY_DIRECTORY_NAMES = [
  'node_modules',
  '.venv',
  'venv',
  '__pycache__',
  'build',
  'dist',
  'out',
]

export interface RipgrepGlobOptions {
  pattern?: string
  rootRelative: string
}

/** User globs come first; safety exclusions are always the final rules. */
export function buildSafeGlobArgs(options: RipgrepGlobOptions): string[] {
  const args: string[] = []
  if (options.pattern) args.push('--glob', options.pattern)

  args.push(...ALWAYS_EXCLUDED_GLOBS.flatMap((glob) => ['--glob', glob]))

  for (const name of HEAVY_DIRECTORY_NAMES) {
    if (isExplicitHeavyDirectory(options.rootRelative, name)) continue
    args.push('--glob', `!${name}`)
    args.push('--glob', `!${name}/**`)
    args.push('--glob', `!**/${name}`)
    args.push('--glob', `!**/${name}/**`)
  }

  return args
}

export function isHeavyDirectoryPath(rootRelative: string): boolean {
  return pathSegments(rootRelative).some((segment) => HEAVY_DIRECTORY_NAMES.includes(segment))
}

function isExplicitHeavyDirectory(rootRelative: string, name: string): boolean {
  return pathSegments(rootRelative).includes(name)
}

function pathSegments(value: string): string[] {
  return normalizeRelative(value)
    .split('/')
    .filter((segment) => segment && segment !== '.')
}

function normalizeRelative(value: string): string {
  const normalized = value.split('\\').join('/').replace(/^\.\//, '')
  return normalized || '.'
}
