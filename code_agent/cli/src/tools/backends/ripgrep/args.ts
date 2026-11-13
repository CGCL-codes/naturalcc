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

// [maint-note:2a4df6c78546]
// 数据契约导航：RipgrepGlobOptions；字段含义以消费端校验为准。
// 成员 pattern（可省略）：string。
// 成员 rootRelative：string。
// [/maint-note:2a4df6c78546]
export interface RipgrepGlobOptions {
  pattern?: string
  rootRelative: string
}

/** User globs come first; safety exclusions are always the final rules. */
// [maint-note:779781a30b6e]
// 维护导航：buildSafeGlobArgs；下列为本函数源码中的接口与路径索引。
// 输入 options：RipgrepGlobOptions。
// 返回契约：string[]。
// 分支条件：options.pattern。
// 分支条件：isExplicitHeavyDirectory(options.rootRelative, name)。
// 遍历边界：const name ← HEAVY_DIRECTORY_NAMES。
// 返回路径：args。
// 直接调用：args.push。
// 直接调用：ALWAYS_EXCLUDED_GLOBS.flatMap。
// 直接调用：isExplicitHeavyDirectory。
// [/maint-note:779781a30b6e]
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

// [maint-note:b18f1a004a32]
// 维护导航：isHeavyDirectoryPath；下列为本函数源码中的接口与路径索引。
// 输入 rootRelative：string。
// 返回契约：boolean。
// 返回路径：pathSegments(rootRelative).some((segment) => HEAVY_DIRECTORY_NAMES.includes(segment))。
// 直接调用：pathSegments(rootRelative).some。
// 直接调用：pathSegments。
// [/maint-note:b18f1a004a32]
export function isHeavyDirectoryPath(rootRelative: string): boolean {
  return pathSegments(rootRelative).some((segment) => HEAVY_DIRECTORY_NAMES.includes(segment))
}

// [maint-note:3ae8e2af5a4d]
// 维护导航：isExplicitHeavyDirectory；下列为本函数源码中的接口与路径索引。
// 输入 rootRelative：string。
// 输入 name：string。
// 返回契约：boolean。
// 返回路径：pathSegments(rootRelative).includes(name)。
// 直接调用：pathSegments(rootRelative).includes。
// 直接调用：pathSegments。
// [/maint-note:3ae8e2af5a4d]
function isExplicitHeavyDirectory(rootRelative: string, name: string): boolean {
  return pathSegments(rootRelative).includes(name)
}

// [maint-note:fe7e9f3f6441]
// 维护导航：pathSegments；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string[]。
// 返回路径：normalizeRelative(value) .split('/') .filter((segment) => segment && segment !== '.')。
// 直接调用：normalizeRelative(value) .split('/') .filter。
// 直接调用：normalizeRelative(value) .split。
// 直接调用：normalizeRelative。
// [/maint-note:fe7e9f3f6441]
function pathSegments(value: string): string[] {
  return normalizeRelative(value)
    .split('/')
    .filter((segment) => segment && segment !== '.')
}

// [maint-note:6fd8430d73f1]
// 维护导航：normalizeRelative；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：normalized || '.'。
// 直接调用：value.split('\\').join('/').replace。
// 直接调用：value.split('\\').join。
// 直接调用：value.split。
// [/maint-note:6fd8430d73f1]
function normalizeRelative(value: string): string {
  const normalized = value.split('\\').join('/').replace(/^\.\//, '')
  return normalized || '.'
}
