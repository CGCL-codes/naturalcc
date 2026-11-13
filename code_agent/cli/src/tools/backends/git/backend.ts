import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto'
import { constants } from 'node:fs'
import { lstat, open, realpath } from 'node:fs/promises'
import { dirname, join, parse, relative, resolve } from 'node:path'

import {
  decodePath,
  parseCheckAttr,
  parseIndexEntries,
  parseNumstat,
  parsePatch,
  parsePorcelainV2,
  parseTreeEntries,
  parseUntracked,
  type GitIndexEntry,
  type GitTreeEntry,
  type RetainParsedValue,
} from './parsers.js'
import {
  GitCommandSession,
  type GitCommandLimits,
  type GitRetainedBudgetEvent,
  type GitSpawn,
} from './runner.js'
import {
  GitChangesError,
  type GitChangeLayer,
  type GitChangesBackend,
  type GitChangesObservation,
  type GitChangesObservationRequest,
  type GitFileRecord,
  type GitPatchLayer,
} from './types.js'

const CONTRACT_VERSION = 'git_changes.v1'
const MAX_CONFIG_BYTES = 1024 * 1024
const MAX_FILES = 10_000
const MAX_HUNKS = 20_000
const MAX_ATTRIBUTE_ARG_BYTES = 512 * 1024
const MAX_PATH_BYTES = 4_096
const MAX_OUTPUT_RESERVE_BYTES = 32 * 1024 * 1024

// [maint-note:51928a5ea7f6]
// 数据契约导航：RepositoryLayout；字段含义以消费端校验为准。
// 成员 workspaceRoot：string。
// 成员 repoRoot：string。
// 成员 gitDir：string。
// 成员 commonDir：string。
// 成员 workspacePrefix：string。
// 成员 workspaceId：string。
// 成员 configFiles：string[]。
// [/maint-note:51928a5ea7f6]
interface RepositoryLayout {
  workspaceRoot: string
  repoRoot: string
  gitDir: string
  commonDir: string
  workspacePrefix: string
  workspaceId: string
  configFiles: string[]
}

// [maint-note:e64ed9156f6e]
// 数据契约导航：Capture；字段含义以消费端校验为准。
// 成员 records：GitFileRecord[]。
// 成员 patches：GitPatchLayer[]。
// 成员 guidance（可省略）：string。
// 成员 coverage：NonNullable<GitChangesObservation['coverage']>。
// 成员 fingerprint：string。
// 成员 retainedBytes：number。
// [/maint-note:e64ed9156f6e]
interface Capture {
  records: GitFileRecord[]
  patches: GitPatchLayer[]
  guidance?: string
  coverage: NonNullable<GitChangesObservation['coverage']>
  fingerprint: string
  retainedBytes: number
}

// [maint-note:8501c37f74f1]
// 数据契约导航：GitChangesBackendOptions；字段含义以消费端校验为准。
// 成员 executable（可省略）：string。
// 成员 path（可省略）：string。
// 成员 spawnProcess（可省略）：GitSpawn。
// 成员 cursorSecret（可省略）：Uint8Array。
// 成员 now（可省略）：() => number。
// 成员 onCommand（可省略）：(request: { args: readonly string[]; cwd: string }) => void。
// 成员 onMetadataRead（可省略）：(path: string) => void。
// 成员 limits（可省略）：Partial<GitCommandLimits>。
// 成员 onRetainedBudget（可省略）：(event: GitRetainedBudgetEvent) => void。
// [/maint-note:8501c37f74f1]
export interface GitChangesBackendOptions {
  executable?: string
  path?: string
  spawnProcess?: GitSpawn
  cursorSecret?: Uint8Array
  now?: () => number
  /** Read-only diagnostic hook used by deterministic command-shape tests. */
  onCommand?: (request: { args: readonly string[]; cwd: string }) => void
  /** Deterministic cancellation/mutation seam before bounded metadata reads. */
  onMetadataRead?: (path: string) => void
  /** Narrow deterministic seam for exact/over budget tests. */
  limits?: Partial<GitCommandLimits>
  /** Read-only retained-budget trace for deterministic high-water tests. */
  onRetainedBudget?: (event: GitRetainedBudgetEvent) => void
}

// [maint-note:ed50bdeb617c]
// 维护导航：createGitChangesBackend；下列为本函数源码中的接口与路径索引。
// 输入 options：GitChangesBackendOptions；缺省表达式 {}。
// 返回契约：GitChangesBackend。
// 分支条件：secret.length < 16。
// 返回路径：{ async observe(request) { const session = new GitCommandSession({ executable: options.executable, path: options.path, spawnProcess: options.spawnProcess, now: options.now, signal: request.signal, timeoutMs: request.timeoutMs, onCommand: op。
// 显式抛出：new Error('git_changes cursor secret must contain at least 16 bytes')。
// 直接调用：Buffer.from。
// 直接调用：randomBytes。
// [/maint-note:ed50bdeb617c]
export function createGitChangesBackend(options: GitChangesBackendOptions = {}): GitChangesBackend {
  const secret = Buffer.from(options.cursorSecret ?? randomBytes(32))
  if (secret.length < 16) throw new Error('git_changes cursor secret must contain at least 16 bytes')

  return {
    // [maint-note:ea9cc301b277]
    // 维护导航：observe；下列为本函数源码中的接口与路径索引。
    // 输入 request：类型由调用方与函数体约束。
    // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
    // 分支条件：outputReserveBytes > 0。
    // 分支条件：!repository。
    // 分支条件：!verifiedRepository || verifiedRepository.workspaceId !== repository.workspaceId。
    // 分支条件：!equalBuffers(configBytes, verifiedConfig)。
    // 分支条件：initial.fingerprint !== verification.fingerprint。
    // 返回路径：{ applicable: false, reason: 'not_git_repository' }。
    // 返回路径：{ applicable: true, workspaceId: repository.workspaceId, revision: `sha256:${verification.fingerprint}`, coverage: verification.coverage, records: verification.records, patches: verification.patches, ...(verification.guidance ? { guidance: 。
    // 直接调用：Math.min。
    // 直接调用：Math.max。
    // 直接调用：Math.floor。
    // 直接调用：session.consumeRetainedBytes。
    // 直接调用：discoverRepository。
    // 直接调用：session.throwIfStopped。
    // 直接调用：checkGitCapability。
    // 直接调用：inspectConfiguration。
    // 直接调用：captureObservation。
    // 直接调用：observationChanged。
    // 直接调用：readConfigurationBytes。
    // 直接调用：equalBuffers。
    // 直接调用：session.releaseRetainedBytes。
    // 直接调用：totalBytes。
    // [/maint-note:ea9cc301b277]
    async observe(request) {
      const session = new GitCommandSession({
        executable: options.executable,
        path: options.path,
        spawnProcess: options.spawnProcess,
        now: options.now,
        signal: request.signal,
        timeoutMs: request.timeoutMs,
        onCommand: options.onCommand,
        onMetadataRead: options.onMetadataRead,
        onRetainedBudget: options.onRetainedBudget,
        limits: options.limits,
      })
      const outputReserveBytes = Math.min(
        MAX_OUTPUT_RESERVE_BYTES,
        Math.max(0, Math.floor(request.outputReserveBytes ?? 0)),
      )
      if (outputReserveBytes > 0) session.consumeRetainedBytes(outputReserveBytes, 'output_reserve')
      const repository = await discoverRepository(session, request.projectDir)
      if (!repository) {
        session.throwIfStopped()
        return { applicable: false, reason: 'not_git_repository' }
      }

      await checkGitCapability(session, repository.repoRoot)
      const configBytes = await inspectConfiguration(session, repository)
      const initial = await captureObservation(session, repository, request, configBytes, false)
      // Re-establish filesystem identity and compare trusted configuration
      // before dispatching any verification command. A changed config is not
      // safe merely because the final content fingerprint would be stale.
      const verifiedRepository = await discoverRepository(session, request.projectDir)
      if (!verifiedRepository || verifiedRepository.workspaceId !== repository.workspaceId) {
        observationChanged('repository_layout_changed')
      }
      const verifiedConfig = await readConfigurationBytes(session, verifiedRepository.configFiles)
      if (!equalBuffers(configBytes, verifiedConfig)) observationChanged('repository_config_changed')
      session.releaseRetainedBytes(totalBytes(configBytes))
      const verification = await captureObservation(session, verifiedRepository, request, verifiedConfig, true)
      session.releaseRetainedBytes(totalBytes(verifiedConfig))
      session.throwIfStopped()
      if (initial.fingerprint !== verification.fingerprint) {
        observationChanged('verification_mismatch')
      }
      session.throwIfStopped()
      return {
        applicable: true,
        workspaceId: repository.workspaceId,
        revision: `sha256:${verification.fingerprint}`,
        coverage: verification.coverage,
        records: verification.records,
        patches: verification.patches,
        ...(verification.guidance ? { guidance: verification.guidance } : {}),
      }
    },
    // [maint-note:c525c00c6ff9]
    // 维护导航：sealCursor；下列为本函数源码中的接口与路径索引。
    // 输入 payload：类型由调用方与函数体约束。
    // 返回路径：`gcv1.${encoded}.${signature}`。
    // 直接调用：Buffer.from(payload, 'utf8').toString。
    // 直接调用：Buffer.from。
    // 直接调用：createHmac('sha256', secret).update(encoded).digest。
    // 直接调用：createHmac('sha256', secret).update。
    // 直接调用：createHmac。
    // [/maint-note:c525c00c6ff9]
    sealCursor(payload) {
      const encoded = Buffer.from(payload, 'utf8').toString('base64url')
      const signature = createHmac('sha256', secret).update(encoded).digest('base64url')
      return `gcv1.${encoded}.${signature}`
    },
    // [maint-note:11dd5a854aa8]
    // 维护导航：openCursor；下列为本函数源码中的接口与路径索引。
    // 输入 cursor：类型由调用方与函数体约束。
    // 分支条件：!match。
    // 分支条件：actual.length !== expected.length || !timingSafeEqual(actual, expected)。
    // 返回路径：undefined。
    // 返回路径：Buffer.from(match[1]!, 'base64url').toString('utf8')。
    // 写入目标：actual。
    // 直接调用：/^gcv1\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)$/u.exec。
    // 直接调用：createHmac('sha256', secret).update(match[1]!).digest。
    // 直接调用：createHmac('sha256', secret).update。
    // 直接调用：createHmac。
    // 直接调用：Buffer.from。
    // 直接调用：timingSafeEqual。
    // 直接调用：Buffer.from(match[1]!, 'base64url').toString。
    // [/maint-note:11dd5a854aa8]
    openCursor(cursor) {
      const match = /^gcv1\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)$/u.exec(cursor)
      if (!match) return undefined
      const expected = createHmac('sha256', secret).update(match[1]!).digest()
      let actual: Buffer
      try { actual = Buffer.from(match[2]!, 'base64url') } catch { return undefined }
      if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) return undefined
      try { return Buffer.from(match[1]!, 'base64url').toString('utf8') } catch { return undefined }
    },
  }
}

// [maint-note:1d99d2d899f9]
// 维护导航：discoverRepository；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 projectDir：string。
// 返回契约：Promise<RepositoryLayout | undefined>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：markerInfo。
// 分支条件：markerInfo.isSymbolicLink()。
// 分支条件：markerInfo.isDirectory()。
// 分支条件：markerInfo.isFile()。
// 分支条件：!match。
// 分支条件：await existsRegularFile(session, worktreeConfig)。
// 分支条件：candidate === filesystemRoot。
// 遍历边界：; ; 。
// 返回路径：{ workspaceRoot, repoRoot: candidate, gitDir, commonDir, workspacePrefix, workspaceId, configFiles }。
// 返回路径：undefined。
// 写入目标：gitDir。
// 写入目标：markerBytes。
// 写入目标：candidate。
// 直接调用：session.throwIfStopped。
// 直接调用：realpath(projectDir).catch。
// 直接调用：realpath。
// 直接调用：parse。
// 直接调用：join。
// 直接调用：lstat(marker).catch。
// 直接调用：lstat。
// 直接调用：markerInfo.isSymbolicLink。
// 直接调用：unsupportedLayout。
// 直接调用：Buffer.alloc。
// 直接调用：markerInfo.isDirectory。
// 直接调用：markerInfo.isFile。
// 直接调用：readBoundedFile。
// 直接调用：/^gitdir: ([^\r\n]+)\r?\n?$/u.exec。
// 直接调用：decodeMetadataText。
// 直接调用：realpath(resolve(candidate, match[1]!)).catch。
// 直接调用：resolve。
// 直接调用：readOptionalBoundedFile。
// 直接调用：realpath(resolve(gitDir, decodeRepositoryPointer(commonText, 'invalid_commondir'))).catch。
// 直接调用：decodeRepositoryPointer。
// 直接调用：rejectAlternates。
// 直接调用：existsRegularFile。
// 直接调用：configFiles.push。
// 直接调用：slash。
// 直接调用：relative。
// 直接调用：createHash('sha256').update(JSON.stringify({ contract: CONTRACT_VERSION, workspaceRoot, workspaceStat: await statIdentity(session, workspaceRoot), repoRoot: candidate, markerStat: await statIdentity(session, marker), markerDigest: digest([m。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// 直接调用：statIdentity。
// 直接调用：digest。
// 直接调用：session.releaseRetainedBytes。
// 直接调用：dirname。
// [/maint-note:1d99d2d899f9]
async function discoverRepository(
  session: GitCommandSession,
  projectDir: string,
): Promise<RepositoryLayout | undefined> {
  session.throwIfStopped()
  const workspaceRoot = await realpath(projectDir).catch(() => {
    throw new GitChangesError('git_changes_unsupported_repository_layout', 'Workspace is unavailable', 'workspace_unavailable')
  })
  session.throwIfStopped()
  let candidate = workspaceRoot
  const filesystemRoot = parse(candidate).root
  for (;;) {
    const marker = join(candidate, '.git')
    session.throwIfStopped()
    const markerInfo = await lstat(marker).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return undefined
      throw error
    })
    session.throwIfStopped()
    if (markerInfo) {
      if (markerInfo.isSymbolicLink()) unsupportedLayout('git_marker_symlink')
      let gitDir: string
      let markerBytes: Buffer = Buffer.alloc(0)
      if (markerInfo.isDirectory()) {
        gitDir = await realpath(marker)
        session.throwIfStopped()
      } else if (markerInfo.isFile()) {
        const markerText = await readBoundedFile(session, marker, 4096)
        markerBytes = markerText
        const match = /^gitdir: ([^\r\n]+)\r?\n?$/u.exec(decodeMetadataText(markerText, 'invalid_gitdir_file'))
        if (!match) unsupportedLayout('invalid_gitdir_file')
        gitDir = await realpath(resolve(candidate, match[1]!)).catch(() => unsupportedLayout('invalid_gitdir_target'))
        session.throwIfStopped()
      } else unsupportedLayout('invalid_git_marker')

      const commonMarker = join(gitDir, 'commondir')
      const commonText = await readOptionalBoundedFile(session, commonMarker, 4096)
      const commonDir = commonText
        ? await realpath(resolve(gitDir, decodeRepositoryPointer(commonText, 'invalid_commondir'))).catch(() => unsupportedLayout('invalid_commondir'))
        : gitDir
      session.throwIfStopped()
      await rejectAlternates(session, commonDir)
      const configFiles = [join(commonDir, 'config')]
      const worktreeConfig = join(gitDir, 'config.worktree')
      if (await existsRegularFile(session, worktreeConfig)) configFiles.push(worktreeConfig)
      const workspacePrefix = slash(relative(candidate, workspaceRoot)) || '.'
      const workspaceId = createHash('sha256').update(JSON.stringify({
        contract: CONTRACT_VERSION,
        workspaceRoot,
        workspaceStat: await statIdentity(session, workspaceRoot),
        repoRoot: candidate,
        markerStat: await statIdentity(session, marker),
        markerDigest: digest([markerBytes]),
        gitDir,
        gitDirStat: await statIdentity(session, gitDir),
        commonMarkerDigest: digest([commonText ?? Buffer.alloc(0)]),
        commonDir,
        commonDirStat: await statIdentity(session, commonDir),
        workspacePrefix,
        configFiles,
      })).digest('hex')
      session.releaseRetainedBytes(markerBytes.length + (commonText?.length ?? 0))
      session.throwIfStopped()
      return { workspaceRoot, repoRoot: candidate, gitDir, commonDir, workspacePrefix, workspaceId, configFiles }
    }
    if (candidate === filesystemRoot) return undefined
    candidate = dirname(candidate)
  }
}

// [maint-note:abade6f092d6]
// 维护导航：rejectAlternates；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 commonDir：string。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：info。
// 直接调用：join。
// 直接调用：session.throwIfStopped。
// 直接调用：lstat(path).catch。
// 直接调用：lstat。
// 直接调用：unsupportedLayout。
// [/maint-note:abade6f092d6]
async function rejectAlternates(session: GitCommandSession, commonDir: string): Promise<void> {
  const path = join(commonDir, 'objects', 'info', 'alternates')
  session.throwIfStopped()
  const info = await lstat(path).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT') return undefined
    throw error
  })
  session.throwIfStopped()
  if (info) unsupportedLayout('alternate_object_store')
}

// [maint-note:43dc68601037]
// 维护导航：checkGitCapability；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 cwd：string。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!supported。
// 显式抛出：new GitChangesError('git_changes_unsupported_git', 'Git version check failed', 'unsupported_git_version')。
// 直接调用：session.run。
// 直接调用：/^git version \d+\.\d+(?:\.\d+)?(?:\s|$)/u.test。
// 直接调用：result.stdout.toString('utf8').trim。
// 直接调用：result.stdout.toString。
// 直接调用：session.releaseRetainedBytes。
// [/maint-note:43dc68601037]
async function checkGitCapability(session: GitCommandSession, cwd: string): Promise<void> {
  const result = await session.run(['--version'], cwd)
  const supported = result.exitCode === 0 && /^git version \d+\.\d+(?:\.\d+)?(?:\s|$)/u.test(result.stdout.toString('utf8').trim())
  session.releaseRetainedBytes(result.stdout.length)
  if (!supported) {
    throw new GitChangesError('git_changes_unsupported_git', 'Git version check failed', 'unsupported_git_version')
  }
}

// [maint-note:80dfe67806d6]
// 维护导航：inspectConfiguration；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 repository：RepositoryLayout。
// 返回契约：Promise<Buffer[]>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 遍历边界：const configFile ← repository.configFiles。
// 返回路径：bytes。
// 直接调用：readConfigurationBytes。
// 直接调用：session.run。
// 直接调用：assertGitSuccess。
// 直接调用：inspectConfigEntries。
// 直接调用：session.releaseRetainedBytes。
// [/maint-note:80dfe67806d6]
async function inspectConfiguration(session: GitCommandSession, repository: RepositoryLayout): Promise<Buffer[]> {
  const bytes = await readConfigurationBytes(session, repository.configFiles)
  for (const configFile of repository.configFiles) {
    const result = await session.run(
      ['config', '--null', '--list', '--no-includes', '--file', configFile],
      repository.repoRoot,
    )
    assertGitSuccess(result, 'config')
    inspectConfigEntries(result.stdout)
    session.releaseRetainedBytes(result.stdout.length)
  }
  return bytes
}

// [maint-note:c44835ad0b51]
// 维护导航：readConfigurationBytes；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 paths：readonly string[]。
// 返回契约：Promise<Buffer[]>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 遍历边界：const path ← paths。
// 返回路径：result。
// 直接调用：readBoundedFile(session, path, MAX_CONFIG_BYTES).catch。
// 直接调用：readBoundedFile。
// 直接调用：result.push。
// [/maint-note:c44835ad0b51]
async function readConfigurationBytes(session: GitCommandSession, paths: readonly string[]): Promise<Buffer[]> {
  const result: Buffer[] = []
  for (const path of paths) {
    const content = await readBoundedFile(session, path, MAX_CONFIG_BYTES).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') unsupportedLayout('missing_repository_config')
      throw error
    })
    result.push(content)
  }
  return result
}

// [maint-note:644a4ac2e925]
// 维护导航：inspectConfigEntries；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 返回契约：void。
// 分支条件：!raw。
// 分支条件：newline < 1。
// 分支条件：key === 'core.bare' && value !== 'false' && value !== 'no' && value !== '0'。
// 分支条件：key === 'core.worktree'。
// 分支条件：key.startsWith('include.') || key.startsWith('includeif.') || /^filter\.[^.]+\./u.test(key) || key === 'diff.external' || (/^diff\.[^.]+\.(?:command|textconv)$/u.test(key)) || key === 'core.attributesfile' || key === 'core.excludesfile' || 。
// 分支条件：key === 'core.fsmonitor' && !['', 'false', 'no', 'off', '0'].includes(value)。
// 遍历边界：const raw ← output.toString('utf8').split('\0')。
// 直接调用：output.toString('utf8').split。
// 直接调用：output.toString。
// 直接调用：raw.indexOf。
// 直接调用：unsupportedConfig。
// 直接调用：raw.slice(0, newline).toLowerCase。
// 直接调用：raw.slice。
// 直接调用：raw.slice(newline + 1).trim().toLowerCase。
// 直接调用：raw.slice(newline + 1).trim。
// 直接调用：unsupportedLayout。
// 直接调用：key.startsWith。
// 直接调用：/^filter\.[^.]+\./u.test。
// 直接调用：/^diff\.[^.]+\.(?:command|textconv)$/u.test。
// 直接调用：configReason。
// 直接调用：['', 'false', 'no', 'off', '0'].includes。
// [/maint-note:644a4ac2e925]
function inspectConfigEntries(output: Buffer): void {
  for (const raw of output.toString('utf8').split('\0')) {
    if (!raw) continue
    const newline = raw.indexOf('\n')
    if (newline < 1) unsupportedConfig('malformed_config')
    const key = raw.slice(0, newline).toLowerCase()
    const value = raw.slice(newline + 1).trim().toLowerCase()
    if (key === 'core.bare' && value !== 'false' && value !== 'no' && value !== '0') unsupportedLayout('bare_repository')
    if (key === 'core.worktree') unsupportedConfig('worktree_redirection')
    if (
      key.startsWith('include.') || key.startsWith('includeif.') ||
      /^filter\.[^.]+\./u.test(key) || key === 'diff.external' ||
      (/^diff\.[^.]+\.(?:command|textconv)$/u.test(key)) ||
      key === 'core.attributesfile' || key === 'core.excludesfile' ||
      key === 'core.pager' || key.startsWith('pager.') || key === 'interactive.difffilter'
    ) unsupportedConfig(configReason(key))
    if (key === 'core.fsmonitor' && !['', 'false', 'no', 'off', '0'].includes(value)) {
      unsupportedConfig('fsmonitor_config')
    }
  }
}

// [maint-note:c3865c2c47e0]
// 维护导航：captureObservation；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 repository：RepositoryLayout。
// 输入 request：GitChangesObservationRequest。
// 输入 configBytes：readonly Buffer[]。
// 输入 materialize：boolean。
// 返回契约：Promise<Capture>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：attributePaths.length > MAX_FILES。
// 分支条件：attributeArgBytes > MAX_ATTRIBUTE_ARG_BYTES。
// 分支条件：treeResult.exitCode !== 0。
// 分支条件：records.length > MAX_FILES。
// 分支条件：records.some((record) => record.layer === 'unmerged') && request.scope !== 'both'。
// 分支条件：request.mode === 'patch'。
// 分支条件：selected.some((record) => record.action === 'unmerged')。
// 分支条件：selected.some((record) => record.boundaryTruncated)。
// 分支条件：untracked.length > 0。
// 分支条件：record.action === 'untracked'。
// 分支条件：record.objectType !== 'file'。
// 分支条件：record.layer !== 'staged' && record.layer !== 'unstaged'。
// 分支条件：numstatPaths.length === 0 && request.path。
// 分支条件：observedHunks > MAX_HUNKS。
// 分支条件：record.oldPath。
// 分支条件：materializedBytes > 0。
// 遍历边界：const record ← selected。
// 遍历边界：const layer ← requestedLayers(request.scope)。
// 遍历边界：const record ← records。
// 返回路径：{ records: materialize ? records : [], patches: materialize ? patches : [], ...(materialize && guidance ? { guidance } : {}), coverage, fingerprint, retainedBytes: materializedBytes, }。
// 显式抛出：new GitChangesError('git_changes_unsupported_conflict', 'Conflicts require scope "both" metadata mode', 'conflict_scope')。
// 显式抛出：new GitChangesError('git_changes_unsupported_conflict', 'Conflict patches are not supported', 'conflict_patch')。
// 显式抛出：new GitChangesError('git_changes_unsupported_cross_boundary_rename', 'Cross-workspace rename patch is not supported', 'cross_boundary_rename')。
// 显式抛出：new GitChangesError('git_changes_unsupported_path_type', 'Only regular-file patches are supported', record.objectType)。
// 写入目标：records。
// 写入目标：guidance。
// 写入目标：parsedRetainedBytes。
// 写入目标：observedHunks。
// 直接调用：session.throwIfStopped。
// 直接调用：runRepoGit。
// 直接调用：parseResult。
// 直接调用：indexEntries.map。
// 直接调用：observationLimit。
// 直接调用：inspectAttributeSources。
// 直接调用：attributePaths.filter。
// 直接调用：attributePaths.reduce。
// 直接调用：Buffer.alloc。
// 直接调用：inspectAttributes。
// 直接调用：runRepoGitAllowUnborn。
// 直接调用：session.releaseRetainedBytes。
// 直接调用：applyConflictStages。
// 直接调用：rejectSubmoduleDescendantRequest。
// 直接调用：addGitlinkRecords。
// 直接调用：filterRecords。
// 直接调用：records.some。
// 直接调用：selectPatchRecords。
// 直接调用：selected.some。
// 直接调用：selected.filter。
// 直接调用：toRepoPath。
// 直接调用：diffArgs。
// 直接调用：Buffer.from。
// 直接调用：digest。
// 直接调用：session.consumeRetainedBytes。
// 直接调用：payloads.push。
// 直接调用：patches.push。
// 直接调用：records.flatMap((record) => [record.oldPath, record.path]).filter((path): path is string => Boolean(path)).map。
// 直接调用：records.flatMap((record) => [record.oldPath, record.path]).filter。
// 直接调用：records.flatMap。
// 直接调用：numstatPaths.push。
// 直接调用：requestedLayers。
// 直接调用：applyNumstat。
// 直接调用：relevantPaths.add。
// 直接调用：collectPathStats。
// 直接调用：indexEntries.filter。
// 直接调用：untrackedPaths.filter。
// 直接调用：attributeFacts.filter。
// 直接调用：treeEntries.filter。
// 直接调用：status.facts.filter。
// 直接调用：observationFingerprint。
// 直接调用：estimateMaterializedBytes。
// 直接调用：totalBytes。
// [/maint-note:c3865c2c47e0]
async function captureObservation(
  session: GitCommandSession,
  repository: RepositoryLayout,
  request: GitChangesObservationRequest,
  configBytes: readonly Buffer[],
  materialize: boolean,
): Promise<Capture> {
  session.throwIfStopped()
  let parsedRetainedBytes = 0
  // [maint-note:761039f1bc67]
  // 维护导航：retainParsedValue；下列为本函数源码中的接口与路径索引。
  // 输入 value：类型由调用方与函数体约束。
  // 写入目标：parsedRetainedBytes。
  // 直接调用：retainedValueBytes。
  // 直接调用：session.consumeRetainedBytes。
  // [/maint-note:761039f1bc67]
  const retainParsedValue: RetainParsedValue = (value) => {
    const bytes = retainedValueBytes(value)
    session.consumeRetainedBytes(bytes, 'parsed_unit')
    parsedRetainedBytes += bytes
  }
  // [maint-note:bde29d1d5e95]
  // 维护导航：parseResult；下列为本函数源码中的接口与路径索引。
  // 输入 result：{ stdout: Buffer }。
  // 输入 parser：(output: Buffer, retain: RetainParsedValue) => T。
  // 返回契约：T。
  // 返回路径：parser(result.stdout, retainParsedValue)。
  // 直接调用：session.consumeRetainedBytes。
  // 直接调用：parser。
  // 直接调用：session.releaseRetainedBytes。
  // [/maint-note:bde29d1d5e95]
  const parseResult = <T>(
    result: { stdout: Buffer },
    parser: (output: Buffer, retain: RetainParsedValue) => T,
  ): T => {
    // Reserve parser scratch before decoding/splitting the raw buffer. Parsed
    // records are then charged one bounded unit at a time before retention, so
    // a low cap cannot be crossed by first constructing a complete AST.
    const scratchBytes = result.stdout.length
    session.consumeRetainedBytes(scratchBytes, 'parse_scratch')
    try {
      return parser(result.stdout, retainParsedValue)
    } finally {
      session.releaseRetainedBytes(scratchBytes, 'parse_scratch')
      session.releaseRetainedBytes(result.stdout.length, 'git_stdout')
    }
  }
  const pathspec = repository.workspacePrefix
  const indexResult = await runRepoGit(session, repository, ['ls-files', '--stage', '-z', '--', pathspec])
  const untrackedResult = await runRepoGit(session, repository, ['ls-files', '--others', '--exclude-standard', '-z', '--', pathspec])
  const indexEntries = parseResult(indexResult, (output, retain) => parseIndexEntries(output, MAX_FILES * 3, retain))
  const untrackedPaths = parseResult(untrackedResult, (output, retain) => parseUntracked(output, MAX_FILES, retain))
  const attributePaths = [...new Set([
    ...indexEntries.map((entry) => entry.path),
    ...untrackedPaths,
  ])]
  if (attributePaths.length > MAX_FILES) observationLimit('file_count_exceeded')
  const attributeSourceBytes = await inspectAttributeSources(
    session,
    repository,
    attributePaths.filter((path) => path === '.gitattributes' || path.endsWith('/.gitattributes')),
  )
  const attributeArgBytes = attributePaths.reduce((total, path) => total + Buffer.byteLength(path, 'utf8') + 1, 0)
  if (attributeArgBytes > MAX_ATTRIBUTE_ARG_BYTES) observationLimit('attribute_path_arguments_exceeded')
  const attrResult = attributePaths.length === 0
    ? { stdout: Buffer.alloc(0), stderr: Buffer.alloc(0), exitCode: 0 }
    : await runRepoGit(
      session,
      repository,
      ['check-attr', '-z', 'filter', 'diff', 'working-tree-encoding', '--', ...attributePaths],
    )
  const attributeFacts = parseResult(attrResult, (output, retain) => parseCheckAttr(output, attributePaths.length, retain))
  inspectAttributes(attributeFacts)

  const treeResult = await runRepoGitAllowUnborn(
    session,
    repository,
    ['ls-tree', '-rdz', '--full-tree', 'HEAD', '--', pathspec],
  )
  const treeEntries = treeResult.exitCode === 0
    ? parseResult(treeResult, (output, retain) => parseTreeEntries(output, MAX_FILES, retain))
    : []
  if (treeResult.exitCode !== 0) session.releaseRetainedBytes(treeResult.stdout.length)
  const statusResult = await runRepoGit(session, repository, [
    'status', '--porcelain=v2', '-z', '--branch', '--find-renames=50%',
    '--untracked-files=all', '--ignore-submodules=all', '--', pathspec,
  ])
  // [maint-note:ef1147345501]
  // 维护导航：mapPath；下列为本函数源码中的接口与路径索引。
  // 输入 path：string。
  // 直接调用：mapRepositoryPath。
  // [/maint-note:ef1147345501]
  const mapPath = (path: string) => mapRepositoryPath(repository, path)
  const status = parseResult(statusResult, (output, retain) => parsePorcelainV2(output, mapPath, MAX_FILES, retain))
  let records = status.records
  applyConflictStages(records, indexEntries, repository)
  rejectSubmoduleDescendantRequest(indexEntries, treeEntries, repository, request)
  const gitlink = addGitlinkRecords(records, indexEntries, treeEntries, repository, request)
  records = filterRecords(records, request)
  if (records.length > MAX_FILES) observationLimit('file_count_exceeded')
  if (records.some((record) => record.layer === 'unmerged') && request.scope !== 'both') {
    throw new GitChangesError('git_changes_unsupported_conflict', 'Conflicts require scope "both" metadata mode', 'conflict_scope')
  }

  const payloads: Buffer[] = []
  let patches: GitPatchLayer[] = []
  let observedHunks = 0
  let guidance: string | undefined
  if (request.mode === 'patch') {
    const selected = selectPatchRecords(records, request.path!)
    if (selected.some((record) => record.action === 'unmerged')) {
      throw new GitChangesError('git_changes_unsupported_conflict', 'Conflict patches are not supported', 'conflict_patch')
    }
    if (selected.some((record) => record.boundaryTruncated)) {
      throw new GitChangesError('git_changes_unsupported_cross_boundary_rename', 'Cross-workspace rename patch is not supported', 'cross_boundary_rename')
    }
    const untracked = selected.filter((record) => record.action === 'untracked')
    if (untracked.length > 0) guidance = 'Untracked files have no Git baseline patch; use read_file for current content.'
    for (const record of selected) {
      if (record.action === 'untracked') continue
      if (record.objectType !== 'file') {
        throw new GitChangesError('git_changes_unsupported_path_type', 'Only regular-file patches are supported', record.objectType)
      }
      if (record.layer !== 'staged' && record.layer !== 'unstaged') {
        throw new GitChangesError('git_changes_unsupported_conflict', 'Conflict patches are not supported', 'conflict_patch')
      }
      const endpoints = record.action === 'renamed' && record.oldPath
        ? [toRepoPath(repository, record.oldPath), toRepoPath(repository, record.path)]
        : [toRepoPath(repository, record.path)]
      const patchResult = await runRepoGit(session, repository, diffArgs(record.layer, 'patch', endpoints))
      const payload = Buffer.from(digest([patchResult.stdout]))
      session.consumeRetainedBytes(payload.length, 'digest_payload')
      parsedRetainedBytes += payload.length
      payloads.push(payload)
      const patch = parseResult(patchResult, (output, retain) => parsePatch(
        output,
        record.layer as 'staged' | 'unstaged',
        record,
        mapPath,
        MAX_HUNKS - observedHunks,
        retain,
      ))
      observedHunks += patch.hunks.length
      patches.push(patch)
    }
  } else {
    const numstatPaths = request.path
      ? [...new Set(records.flatMap((record) => [record.oldPath, record.path]).filter((path): path is string => Boolean(path)).map((path) => toRepoPath(repository, path)))]
      : [pathspec]
    if (numstatPaths.length === 0 && request.path) numstatPaths.push(toRepoPath(repository, request.path))
    for (const layer of requestedLayers(request.scope)) {
      const numstatResult = await runRepoGit(session, repository, diffArgs(layer, 'numstat', numstatPaths))
      const payload = Buffer.from(digest([numstatResult.stdout]))
      session.consumeRetainedBytes(payload.length, 'digest_payload')
      parsedRetainedBytes += payload.length
      payloads.push(payload)
      applyNumstat(
        records,
        layer,
        parseResult(numstatResult, (output, retain) => parseNumstat(output, MAX_FILES, retain)),
        repository,
      )
    }
  }
  if (observedHunks > MAX_HUNKS) observationLimit('hunk_count_exceeded')

  const relevantPaths = new Set<string>()
  for (const record of records) {
    relevantPaths.add(record.path)
    if (record.oldPath) relevantPaths.add(record.oldPath)
  }
  const stats = await collectPathStats(session, repository.workspaceRoot, relevantPaths, retainParsedValue)
  const relevantIndex = indexEntries.filter((entry) => isRelevantRepositoryPath(repository, entry.path, request, records))
  const relevantUntracked = untrackedPaths.filter((path) => isRelevantRepositoryPath(repository, path, request, records))
  const relevantAttributes = attributeFacts.filter((fact) => isRelevantRepositoryPath(repository, fact.path, request, records))
  const relevantTree = treeEntries.filter((entry) => isRelevantRepositoryPath(repository, entry.path, request, records))
  const relevantStatus = status.facts.filter((fact) => fact.paths.some((path) => isRelevantWorkspacePath(path, request, records)))
  const fingerprint = observationFingerprint({
    workspaceId: repository.workspaceId,
    request,
    configBytes,
    relevantIndex,
    relevantUntracked,
    relevantAttributes,
    relevantTree,
    relevantStatus,
    attributeSourceBytes,
    headOid: status.headOid,
    records,
    payloads,
    patches,
    stats,
  })
  const coverage: NonNullable<GitChangesObservation['coverage']> = {
    parentRepositoryChanges: 'complete',
    submoduleWorktrees: gitlink.count === 0 ? 'not_present' : 'not_observed',
    gitlinkCount: gitlink.count,
  }
  const materializedBytes = materialize
    ? estimateMaterializedBytes(records, patches, guidance, coverage)
    : 0
  if (materializedBytes > 0) session.consumeRetainedBytes(materializedBytes, 'materialized_candidate')
  session.releaseRetainedBytes(parsedRetainedBytes + totalBytes(attributeSourceBytes))
  session.throwIfStopped()
  return {
    records: materialize ? records : [],
    patches: materialize ? patches : [],
    ...(materialize && guidance ? { guidance } : {}),
    coverage,
    fingerprint,
    retainedBytes: materializedBytes,
  }
}

// [maint-note:02ca47682887]
// 维护导航：inspectAttributeSources；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 repository：RepositoryLayout。
// 输入 discoveredRepoPaths：readonly string[]。
// 返回契约：Promise<Buffer[]>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!info。
// 分支条件：!info.isFile() || info.isSymbolicLink()。
// 分支条件：!line || line.startsWith('#')。
// 分支条件：/^(?:filter(?:=.+)?|working-tree-encoding(?:=.+)?)$/iu.test(field)。
// 分支条件：/^diff=.+$/iu.test(field)。
// 遍历边界：let count = 0; count <= prefixParts.length; count += 1。
// 遍历边界：const path ← discoveredRepoPaths。
// 遍历边界：const path ← [...sources].sort()。
// 遍历边界：const rawLine ← logical.split(/\r?\n/gu)。
// 遍历边界：const field ← fields。
// 返回路径：retained。
// 写入目标：count。
// 写入目标：text。
// 直接调用：join。
// 直接调用：repository.workspacePrefix.split。
// 直接调用：sources.add。
// 直接调用：prefixParts.slice。
// 直接调用：path.split。
// 直接调用：[...sources].sort。
// 直接调用：lstat(path).catch。
// 直接调用：lstat。
// 直接调用：info.isFile。
// 直接调用：info.isSymbolicLink。
// 直接调用：unsupportedAttributes。
// 直接调用：readBoundedFile。
// 直接调用：new TextDecoder('utf-8', { fatal: true }).decode。
// 直接调用：text.replace。
// 直接调用：logical.split。
// 直接调用：rawLine.trim。
// 直接调用：line.startsWith。
// 直接调用：line.split(/\s+/gu).slice。
// 直接调用：line.split。
// 直接调用：/^(?:filter(?:=.+)?|working-tree-encoding(?:=.+)?)$/iu.test。
// 直接调用：field.toLowerCase().startsWith。
// 直接调用：field.toLowerCase。
// 直接调用：/^diff=.+$/iu.test。
// 直接调用：Buffer.from。
// 直接调用：session.consumeRetainedBytes。
// 直接调用：retained.push。
// [/maint-note:02ca47682887]
async function inspectAttributeSources(
  session: GitCommandSession,
  repository: RepositoryLayout,
  discoveredRepoPaths: readonly string[],
): Promise<Buffer[]> {
  const sources = new Set<string>([join(repository.commonDir, 'info', 'attributes')])
  const prefixParts = repository.workspacePrefix === '.' ? [] : repository.workspacePrefix.split('/')
  for (let count = 0; count <= prefixParts.length; count += 1) {
    sources.add(join(repository.repoRoot, ...prefixParts.slice(0, count), '.gitattributes'))
  }
  for (const path of discoveredRepoPaths) sources.add(join(repository.repoRoot, ...path.split('/')))

  const retained: Buffer[] = []
  for (const path of [...sources].sort()) {
    const info = await lstat(path).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return undefined
      throw error
    })
    if (!info) continue
    if (!info.isFile() || info.isSymbolicLink()) unsupportedAttributes('unsafe_attributes_file')
    const content = await readBoundedFile(session, path, MAX_CONFIG_BYTES)
    let text: string
    try { text = new TextDecoder('utf-8', { fatal: true }).decode(content) } catch {
      unsupportedAttributes('invalid_attributes_encoding')
    }
    const logical = text.replace(/\\\r?\n/gu, '')
    for (const rawLine of logical.split(/\r?\n/gu)) {
      const line = rawLine.trim()
      if (!line || line.startsWith('#')) continue
      const fields = line.split(/\s+/gu).slice(1)
      for (const field of fields) {
        if (/^(?:filter(?:=.+)?|working-tree-encoding(?:=.+)?)$/iu.test(field)) {
          unsupportedAttributes(field.toLowerCase().startsWith('filter') ? 'filter_attribute' : 'working_tree_encoding')
        }
        if (/^diff=.+$/iu.test(field)) unsupportedAttributes('custom_diff_driver')
      }
    }
    const pathBytes = Buffer.from(path)
    session.consumeRetainedBytes(pathBytes.length)
    retained.push(pathBytes, content)
  }
  return retained
}

// [maint-note:2bf902142662]
// 维护导航：runRepoGit；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 repository：RepositoryLayout。
// 输入 command：readonly string[]。
// 输入 stdin：Buffer（可省略）。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：result。
// 直接调用：session.run。
// 直接调用：assertGitSuccess。
// [/maint-note:2bf902142662]
async function runRepoGit(
  session: GitCommandSession,
  repository: RepositoryLayout,
  command: readonly string[],
  stdin?: Buffer,
) {
  const result = await session.run([
    '--no-pager', '--literal-pathspecs', '--no-optional-locks',
    '-c', 'core.fsmonitor=false', '-c', 'core.quotePath=true',
    `--git-dir=${repository.gitDir}`, `--work-tree=${repository.repoRoot}`,
    '-C', repository.repoRoot,
    ...command,
  ], repository.repoRoot, stdin)
  assertGitSuccess(result, command[0] ?? 'git')
  return result
}

// [maint-note:fea40e207193]
// 维护导航：runRepoGitAllowUnborn；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 repository：RepositoryLayout。
// 输入 command：readonly string[]。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：result.exitCode === 0。
// 分支条件：/not a valid object name HEAD|bad revision 'HEAD'|ambiguous argument 'HEAD'/iu.test(stderr)。
// 返回路径：result。
// 直接调用：session.run。
// 直接调用：result.stderr.toString。
// 直接调用：/not a valid object name HEAD|bad revision 'HEAD'|ambiguous argument 'HEAD'/iu.test。
// 直接调用：assertGitSuccess。
// [/maint-note:fea40e207193]
async function runRepoGitAllowUnborn(
  session: GitCommandSession,
  repository: RepositoryLayout,
  command: readonly string[],
) {
  const result = await session.run([
    '--no-pager', '--literal-pathspecs', '--no-optional-locks',
    '-c', 'core.fsmonitor=false', '-c', 'core.quotePath=true',
    `--git-dir=${repository.gitDir}`, `--work-tree=${repository.repoRoot}`,
    '-C', repository.repoRoot,
    ...command,
  ], repository.repoRoot)
  if (result.exitCode === 0) return result
  const stderr = result.stderr.toString('utf8')
  if (/not a valid object name HEAD|bad revision 'HEAD'|ambiguous argument 'HEAD'/iu.test(stderr)) return result
  assertGitSuccess(result, command[0] ?? 'git')
  return result
}

// [maint-note:cbbd69504090]
// 维护导航：assertGitSuccess；下列为本函数源码中的接口与路径索引。
// 输入 result：{ exitCode: number; stderr: Buffer }。
// 输入 operation：string。
// 返回契约：void。
// 分支条件：result.exitCode === 0。
// 分支条件：/missing blob|missing tree|unable to read|could not fetch|promisor|bad object/iu.test(diagnostic)。
// 返回路径：无返回值。
// 显式抛出：new GitChangesError('git_changes_missing_object', 'Required Git object is unavailable locally', 'missing_object')。
// 显式抛出：new GitChangesError('git_changes_git_failed', `Git ${operation} failed`, `exit_${result.exitCode}`)。
// 直接调用：result.stderr.toString。
// 直接调用：/missing blob|missing tree|unable to read|could not fetch|promisor|bad object/iu.test。
// [/maint-note:cbbd69504090]
function assertGitSuccess(result: { exitCode: number; stderr: Buffer }, operation: string): void {
  if (result.exitCode === 0) return
  const diagnostic = result.stderr.toString('utf8')
  if (/missing blob|missing tree|unable to read|could not fetch|promisor|bad object/iu.test(diagnostic)) {
    throw new GitChangesError('git_changes_missing_object', 'Required Git object is unavailable locally', 'missing_object')
  }
  throw new GitChangesError('git_changes_git_failed', `Git ${operation} failed`, `exit_${result.exitCode}`)
}

// [maint-note:dc238c835465]
// 维护导航：diffArgs；下列为本函数源码中的接口与路径索引。
// 输入 layer：'staged' | 'unstaged'。
// 输入 mode：'numstat' | 'patch'。
// 输入 paths：readonly string[]。
// 返回契约：string[]。
// 分支条件：layer === 'staged'。
// 分支条件：mode === 'numstat'。
// 返回路径：result。
// 直接调用：result.push。
// [/maint-note:dc238c835465]
function diffArgs(
  layer: 'staged' | 'unstaged',
  mode: 'numstat' | 'patch',
  paths: readonly string[],
): string[] {
  const result = ['diff']
  if (layer === 'staged') result.push('--cached')
  result.push('--ignore-submodules=all', '--no-ext-diff', '--no-textconv', '--no-color', '--find-renames=50%')
  if (mode === 'numstat') result.push('--numstat', '-z')
  else result.push('--patch', '--unified=3', '--full-index', '--src-prefix=a/', '--dst-prefix=b/')
  result.push('--', ...paths)
  return result
}

// [maint-note:27d47c943896]
// 维护导航：inspectAttributes；下列为本函数源码中的接口与路径索引。
// 输入 values：readonly { path: string; attribute: string; value: string }[]。
// 返回契约：void。
// 分支条件：item.attribute === 'filter' && !['unspecified', 'unset'].includes(value)。
// 分支条件：item.attribute === 'working-tree-encoding' && !['unspecified', 'unset'].includes(value)。
// 分支条件：item.attribute === 'diff' && !['unspecified', 'unset', 'set'].includes(value)。
// 遍历边界：const item ← values。
// 直接调用：item.value.toLowerCase。
// 直接调用：['unspecified', 'unset'].includes。
// 直接调用：unsupportedAttributes。
// 直接调用：['unspecified', 'unset', 'set'].includes。
// [/maint-note:27d47c943896]
function inspectAttributes(values: readonly { path: string; attribute: string; value: string }[]): void {
  for (const item of values) {
    const value = item.value.toLowerCase()
    if (item.attribute === 'filter' && !['unspecified', 'unset'].includes(value)) unsupportedAttributes('filter_attribute')
    if (item.attribute === 'working-tree-encoding' && !['unspecified', 'unset'].includes(value)) unsupportedAttributes('working_tree_encoding')
    if (item.attribute === 'diff' && !['unspecified', 'unset', 'set'].includes(value)) unsupportedAttributes('custom_diff_driver')
  }
}

// [maint-note:7e7a09886db2]
// 维护导航：addGitlinkRecords；下列为本函数源码中的接口与路径索引。
// 输入 records：GitFileRecord[]。
// 输入 indexEntries：readonly GitIndexEntry[]。
// 输入 treeEntries：readonly GitTreeEntry[]。
// 输入 repository：RepositoryLayout。
// 输入 request：GitChangesObservationRequest。
// 返回契约：{ count: number }。
// 分支条件：request.scope !== 'unstaged'。
// 分支条件：localPath === undefined || records.some((record) => record.layer === 'staged' && record.path === localPath)。
// 分支条件：before?.oid === after?.oid && before?.mode === after?.mode。
// 遍历边界：const repoPath ← paths。
// 返回路径：{ count: paths.filter((path) => isRelevantRepositoryPath(repository, path, request, records)).length }。
// 直接调用：indexEntries.filter((entry) => entry.stage === 0 && entry.mode === '160000').map。
// 直接调用：indexEntries.filter。
// 直接调用：treeEntries.filter((entry) => entry.mode === '160000').map。
// 直接调用：treeEntries.filter。
// 直接调用：index.keys。
// 直接调用：head.keys。
// 直接调用：mapRepositoryPath。
// 直接调用：records.some。
// 直接调用：head.get。
// 直接调用：index.get。
// 直接调用：records.push。
// 直接调用：paths.filter。
// [/maint-note:7e7a09886db2]
function addGitlinkRecords(
  records: GitFileRecord[],
  indexEntries: readonly GitIndexEntry[],
  treeEntries: readonly GitTreeEntry[],
  repository: RepositoryLayout,
  request: GitChangesObservationRequest,
): { count: number } {
  const index = new Map(indexEntries.filter((entry) => entry.stage === 0 && entry.mode === '160000').map((entry) => [entry.path, entry]))
  const head = new Map(treeEntries.filter((entry) => entry.mode === '160000').map((entry) => [entry.path, entry]))
  const paths = [...new Set([...index.keys(), ...head.keys()])]
  if (request.scope !== 'unstaged') {
    for (const repoPath of paths) {
      const localPath = mapRepositoryPath(repository, repoPath)
      if (localPath === undefined || records.some((record) => record.layer === 'staged' && record.path === localPath)) continue
      const before = head.get(repoPath)
      const after = index.get(repoPath)
      if (before?.oid === after?.oid && before?.mode === after?.mode) continue
      records.push({
        layer: 'staged',
        action: !before ? 'added' : !after ? 'deleted' : 'modified',
        path: localPath,
        contentChanged: before?.oid !== after?.oid,
        binary: 'unknown',
        modeChanged: before?.mode !== after?.mode,
        ...(before ? { oldMode: before.mode } : {}),
        ...(after ? { newMode: after.mode } : {}),
        objectType: 'gitlink',
      })
    }
  }
  return { count: paths.filter((path) => isRelevantRepositoryPath(repository, path, request, records)).length }
}

// [maint-note:0ccfed6807b9]
// 维护导航：rejectSubmoduleDescendantRequest；下列为本函数源码中的接口与路径索引。
// 输入 indexEntries：readonly GitIndexEntry[]。
// 输入 treeEntries：readonly GitTreeEntry[]。
// 输入 repository：RepositoryLayout。
// 输入 request：GitChangesObservationRequest。
// 返回契约：void。
// 分支条件：request.mode !== 'patch' || !request.path。
// 分支条件：localPath !== undefined && request.path.startsWith(`${localPath}/`)。
// 遍历边界：const repoPath ← roots。
// 返回路径：无返回值。
// 显式抛出：new GitChangesError( 'git_changes_unsupported_submodule_path', 'Paths inside submodule worktrees are not observed', 'submodule_descendant_path', )。
// 直接调用：indexEntries.filter((entry) => entry.mode === '160000').map。
// 直接调用：indexEntries.filter。
// 直接调用：treeEntries.filter((entry) => entry.mode === '160000').map。
// 直接调用：treeEntries.filter。
// 直接调用：mapRepositoryPath。
// 直接调用：request.path.startsWith。
// [/maint-note:0ccfed6807b9]
function rejectSubmoduleDescendantRequest(
  indexEntries: readonly GitIndexEntry[],
  treeEntries: readonly GitTreeEntry[],
  repository: RepositoryLayout,
  request: GitChangesObservationRequest,
): void {
  if (request.mode !== 'patch' || !request.path) return
  const roots = new Set([
    ...indexEntries.filter((entry) => entry.mode === '160000').map((entry) => entry.path),
    ...treeEntries.filter((entry) => entry.mode === '160000').map((entry) => entry.path),
  ])
  for (const repoPath of roots) {
    const localPath = mapRepositoryPath(repository, repoPath)
    if (localPath !== undefined && request.path.startsWith(`${localPath}/`)) {
      throw new GitChangesError(
        'git_changes_unsupported_submodule_path',
        'Paths inside submodule worktrees are not observed',
        'submodule_descendant_path',
      )
    }
  }
}

// [maint-note:6914b89c5688]
// 维护导航：applyConflictStages；下列为本函数源码中的接口与路径索引。
// 输入 records：GitFileRecord[]。
// 输入 indexEntries：readonly GitIndexEntry[]。
// 输入 repository：RepositoryLayout。
// 返回契约：void。
// 分支条件：record.layer !== 'unmerged'。
// 分支条件：stages.length === 0。
// 遍历边界：const record ← records。
// 显式抛出：new GitChangesError('git_changes_patch_protocol_error', 'Conflict record omitted index stages', 'missing_conflict_stages')。
// 写入目标：record.conflictStages。
// 直接调用：[...new Set(indexEntries .filter((entry) => entry.stage !== 0 && mapRepositoryPath(repository, entry.path) === record.path) .map((entry) => entry.stage))].sort。
// 直接调用：indexEntries .filter((entry) => entry.stage !== 0 && mapRepositoryPath(repository, entry.path) === record.path) .map。
// 直接调用：indexEntries .filter。
// [/maint-note:6914b89c5688]
function applyConflictStages(
  records: GitFileRecord[],
  indexEntries: readonly GitIndexEntry[],
  repository: RepositoryLayout,
): void {
  for (const record of records) {
    if (record.layer !== 'unmerged') continue
    const stages = [...new Set(indexEntries
      .filter((entry) => entry.stage !== 0 && mapRepositoryPath(repository, entry.path) === record.path)
      .map((entry) => entry.stage))].sort() as Array<1 | 2 | 3>
    if (stages.length === 0) {
      throw new GitChangesError('git_changes_patch_protocol_error', 'Conflict record omitted index stages', 'missing_conflict_stages')
    }
    record.conflictStages = stages
  }
}

// [maint-note:ed727406c2a1]
// 维护导航：filterRecords；下列为本函数源码中的接口与路径索引。
// 输入 records：readonly GitFileRecord[]。
// 输入 request：GitChangesObservationRequest。
// 返回契约：GitFileRecord[]。
// 分支条件：!request.path。
// 分支条件：request.mode === 'patch'。
// 分支条件：!aliases.has(record.path) && (!record.oldPath || !aliases.has(record.oldPath))。
// 分支条件：alias && !aliases.has(alias)。
// 遍历边界：changed。
// 遍历边界：const record ← inScope。
// 遍历边界：const alias ← [record.path, record.oldPath]。
// 返回路径：inScope。
// 返回路径：inScope.filter((record) => aliases.has(record.path) || (record.oldPath ? aliases.has(record.oldPath) : false))。
// 返回路径：inScope.filter((record) => { const prefix = `${request.path}/` return record.path === request.path || record.oldPath === request.path || record.path.startsWith(prefix) || record.oldPath?.startsWith(prefix) === true })。
// 写入目标：changed。
// 直接调用：records.filter。
// 直接调用：aliases.has。
// 直接调用：aliases.add。
// 直接调用：inScope.filter。
// [/maint-note:ed727406c2a1]
function filterRecords(records: readonly GitFileRecord[], request: GitChangesObservationRequest): GitFileRecord[] {
  const layers = request.scope === 'both' ? new Set(['staged', 'unstaged', 'unmerged']) : new Set([request.scope, 'unmerged'])
  const inScope = records.filter((record) => layers.has(record.layer))
  if (!request.path) return inScope
  if (request.mode === 'patch') {
    const aliases = new Set([request.path])
    let changed = true
    while (changed) {
      changed = false
      for (const record of inScope) {
        if (!aliases.has(record.path) && (!record.oldPath || !aliases.has(record.oldPath))) continue
        for (const alias of [record.path, record.oldPath]) {
          if (alias && !aliases.has(alias)) {
            aliases.add(alias)
            changed = true
          }
        }
      }
    }
    return inScope.filter((record) => aliases.has(record.path) || (record.oldPath ? aliases.has(record.oldPath) : false))
  }
  return inScope.filter((record) => {
    const prefix = `${request.path}/`
    return record.path === request.path || record.oldPath === request.path ||
      record.path.startsWith(prefix) || record.oldPath?.startsWith(prefix) === true
  })
}

// [maint-note:65c3d5315443]
// 维护导航：selectPatchRecords；下列为本函数源码中的接口与路径索引。
// 输入 records：readonly GitFileRecord[]。
// 输入 path：string。
// 返回契约：GitFileRecord[]。
// 分支条件：!aliases.has(record.path) && (!record.oldPath || !aliases.has(record.oldPath))。
// 分支条件：alias && !aliases.has(alias)。
// 分支条件：[...perLayer.values()].some((count) => count > 1)。
// 遍历边界：changed。
// 遍历边界：const record ← records。
// 遍历边界：const alias ← [record.path, record.oldPath]。
// 遍历边界：const record ← selected。
// 返回路径：selected.sort((left, right) => layerRank(left.layer) - layerRank(right.layer))。
// 显式抛出：new GitChangesError('git_changes_ambiguous_change', 'Path maps to multiple Git changes in one layer', 'ambiguous_alias')。
// 写入目标：changed。
// 直接调用：aliases.has。
// 直接调用：aliases.add。
// 直接调用：records.filter。
// 直接调用：perLayer.set。
// 直接调用：perLayer.get。
// 直接调用：[...perLayer.values()].some。
// 直接调用：perLayer.values。
// 直接调用：selected.sort。
// [/maint-note:65c3d5315443]
function selectPatchRecords(records: readonly GitFileRecord[], path: string): GitFileRecord[] {
  const aliases = new Set([path])
  let changed = true
  while (changed) {
    changed = false
    for (const record of records) {
      if (!aliases.has(record.path) && (!record.oldPath || !aliases.has(record.oldPath))) continue
      for (const alias of [record.path, record.oldPath]) {
        if (alias && !aliases.has(alias)) {
          aliases.add(alias)
          changed = true
        }
      }
    }
  }
  const selected = records.filter((record) => aliases.has(record.path) || (record.oldPath ? aliases.has(record.oldPath) : false))
  const perLayer = new Map<GitChangeLayer, number>()
  for (const record of selected) perLayer.set(record.layer, (perLayer.get(record.layer) ?? 0) + 1)
  if ([...perLayer.values()].some((count) => count > 1)) {
    throw new GitChangesError('git_changes_ambiguous_change', 'Path maps to multiple Git changes in one layer', 'ambiguous_alias')
  }
  return selected.sort((left, right) => layerRank(left.layer) - layerRank(right.layer))
}

// [maint-note:7315477537f0]
// 维护导航：applyNumstat；下列为本函数源码中的接口与路径索引。
// 输入 records：GitFileRecord[]。
// 输入 layer：'staged' | 'unstaged'。
// 输入 entries：ReturnType<typeof parseNumstat>。
// 输入 repository：RepositoryLayout。
// 返回契约：void。
// 分支条件：record.layer === layer && record.action !== 'untracked' && record.objectType !== 'gitlink'。
// 分支条件：path === undefined && oldPath === undefined。
// 分支条件：candidates.length === 0 && records.some((record) => ( record.layer === 'unmerged' && (record.path === path || record.oldPath === oldPath) ))。
// 分支条件：candidates.length !== 1。
// 遍历边界：const record ← records。
// 遍历边界：const entry ← entries。
// 显式抛出：new GitChangesError('git_changes_ambiguous_change', 'numstat does not map to one Git change', 'numstat_mapping')。
// 写入目标：record.binary。
// 写入目标：record.contentChanged。
// 直接调用：mapRepositoryPath。
// 直接调用：records.filter。
// 直接调用：records.some。
// [/maint-note:7315477537f0]
function applyNumstat(
  records: GitFileRecord[],
  layer: 'staged' | 'unstaged',
  entries: ReturnType<typeof parseNumstat>,
  repository: RepositoryLayout,
): void {
  for (const record of records) {
    if (record.layer === layer && record.action !== 'untracked' && record.objectType !== 'gitlink') {
      record.binary = false
      record.contentChanged = false
    }
  }
  for (const entry of entries) {
    const path = mapRepositoryPath(repository, entry.path)
    const oldPath = entry.oldPath ? mapRepositoryPath(repository, entry.oldPath) : undefined
    if (path === undefined && oldPath === undefined) continue
    const candidates = records.filter((record) => record.layer === layer && (
      record.path === path || (oldPath !== undefined && (
        record.oldPath === oldPath || record.oldPath === path || record.path === oldPath
      ))
    ))
    if (candidates.length === 0 && records.some((record) => (
      record.layer === 'unmerged' && (record.path === path || record.oldPath === oldPath)
    ))) continue
    if (candidates.length !== 1) {
      throw new GitChangesError('git_changes_ambiguous_change', 'numstat does not map to one Git change', 'numstat_mapping')
    }
    const record = candidates[0]!
    record.binary = entry.added === 'binary' || entry.deleted === 'binary'
    record.contentChanged = record.binary || (entry.added as number) + (entry.deleted as number) > 0
  }
}

// [maint-note:abd3705a57a0]
// 维护导航：requestedLayers；下列为本函数源码中的接口与路径索引。
// 输入 scope：GitChangesObservationRequest['scope']。
// 返回契约：Array<'staged' | 'unstaged'>。
// 返回路径：scope === 'both' ? ['staged', 'unstaged'] : [scope]。
// [/maint-note:abd3705a57a0]
function requestedLayers(scope: GitChangesObservationRequest['scope']): Array<'staged' | 'unstaged'> {
  return scope === 'both' ? ['staged', 'unstaged'] : [scope]
}

// [maint-note:c2afa5bccaef]
// 维护导航：mapRepositoryPath；下列为本函数源码中的接口与路径索引。
// 输入 repository：RepositoryLayout。
// 输入 path：string。
// 返回契约：string | undefined。
// 分支条件：!normalized || Buffer.byteLength(normalized, 'utf8') > MAX_PATH_BYTES || normalized.startsWith('/') || normalized.split('/').some((part) => part === '..')。
// 分支条件：repository.workspacePrefix === '.'。
// 分支条件：normalized === repository.workspacePrefix。
// 返回路径：normalized。
// 返回路径：'.'。
// 返回路径：normalized.startsWith(prefix) ? normalized.slice(prefix.length) : undefined。
// 显式抛出：new GitChangesError('git_changes_patch_protocol_error', 'Git returned an out-of-bound path', 'invalid_repository_path')。
// 直接调用：Buffer.byteLength。
// 直接调用：normalized.startsWith。
// 直接调用：normalized.split('/').some。
// 直接调用：normalized.split。
// 直接调用：normalized.slice。
// [/maint-note:c2afa5bccaef]
function mapRepositoryPath(repository: RepositoryLayout, path: string): string | undefined {
  const normalized = path
  if (
    !normalized || Buffer.byteLength(normalized, 'utf8') > MAX_PATH_BYTES ||
    normalized.startsWith('/') || normalized.split('/').some((part) => part === '..')
  ) {
    throw new GitChangesError('git_changes_patch_protocol_error', 'Git returned an out-of-bound path', 'invalid_repository_path')
  }
  if (repository.workspacePrefix === '.') return normalized
  if (normalized === repository.workspacePrefix) return '.'
  const prefix = `${repository.workspacePrefix}/`
  return normalized.startsWith(prefix) ? normalized.slice(prefix.length) : undefined
}

// [maint-note:54c799d82dbd]
// 维护导航：isRelevantRepositoryPath；下列为本函数源码中的接口与路径索引。
// 输入 repository：RepositoryLayout。
// 输入 repoPath：string。
// 输入 request：GitChangesObservationRequest。
// 输入 records：readonly GitFileRecord[]。
// 返回契约：boolean。
// 返回路径：local !== undefined && isRelevantWorkspacePath(local, request, records)。
// 直接调用：mapRepositoryPath。
// 直接调用：isRelevantWorkspacePath。
// [/maint-note:54c799d82dbd]
function isRelevantRepositoryPath(
  repository: RepositoryLayout,
  repoPath: string,
  request: GitChangesObservationRequest,
  records: readonly GitFileRecord[],
): boolean {
  const local = mapRepositoryPath(repository, repoPath)
  return local !== undefined && isRelevantWorkspacePath(local, request, records)
}

// [maint-note:4d99e0a5e29a]
// 维护导航：isRelevantWorkspacePath；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 输入 request：GitChangesObservationRequest。
// 输入 records：readonly GitFileRecord[]。
// 返回契约：boolean。
// 分支条件：!request.path。
// 分支条件：request.mode !== 'patch'。
// 分支条件：!aliases.has(record.path) && (!record.oldPath || !aliases.has(record.oldPath))。
// 分支条件：alias && !aliases.has(alias)。
// 遍历边界：changed。
// 遍历边界：const record ← records。
// 遍历边界：const alias ← [record.path, record.oldPath]。
// 返回路径：true。
// 返回路径：path === request.path || path.startsWith(`${request.path}/`)。
// 返回路径：aliases.has(path)。
// 写入目标：changed。
// 直接调用：path.startsWith。
// 直接调用：aliases.has。
// 直接调用：aliases.add。
// [/maint-note:4d99e0a5e29a]
function isRelevantWorkspacePath(
  path: string,
  request: GitChangesObservationRequest,
  records: readonly GitFileRecord[],
): boolean {
  if (!request.path) return true
  if (request.mode !== 'patch') return path === request.path || path.startsWith(`${request.path}/`)
  const aliases = new Set([request.path])
  let changed = true
  while (changed) {
    changed = false
    for (const record of records) {
      if (!aliases.has(record.path) && (!record.oldPath || !aliases.has(record.oldPath))) continue
      for (const alias of [record.path, record.oldPath]) {
        if (alias && !aliases.has(alias)) {
          aliases.add(alias)
          changed = true
        }
      }
    }
  }
  return aliases.has(path)
}

// [maint-note:d04e0d02cdab]
// 维护导航：toRepoPath；下列为本函数源码中的接口与路径索引。
// 输入 repository：RepositoryLayout。
// 输入 workspacePath：string。
// 返回契约：string。
// 返回路径：repository.workspacePrefix === '.' ? workspacePath : `${repository.workspacePrefix}/${workspacePath}`。
// [/maint-note:d04e0d02cdab]
function toRepoPath(repository: RepositoryLayout, workspacePath: string): string {
  return repository.workspacePrefix === '.' ? workspacePath : `${repository.workspacePrefix}/${workspacePath}`
}

// [maint-note:91893d2033a9]
// 维护导航：collectPathStats；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 workspaceRoot：string。
// 输入 paths：ReadonlySet<string>。
// 输入 retain：RetainParsedValue。
// 返回契约：Promise<unknown[]>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 遍历边界：const path ← [...paths].sort()。
// 返回路径：result。
// 直接调用：[...paths].sort。
// 直接调用：session.throwIfStopped。
// 直接调用：lstat(join(workspaceRoot, ...path.split('/')), { bigint: true }).catch。
// 直接调用：lstat。
// 直接调用：join。
// 直接调用：path.split。
// 直接调用：info.isFile。
// 直接调用：info.isDirectory。
// 直接调用：info.isSymbolicLink。
// 直接调用：info.dev.toString。
// 直接调用：info.ino.toString。
// 直接调用：info.size.toString。
// 直接调用：info.mtimeNs.toString。
// 直接调用：info.ctimeNs.toString。
// 直接调用：info.mode.toString。
// 直接调用：retain。
// 直接调用：result.push。
// [/maint-note:91893d2033a9]
async function collectPathStats(
  session: GitCommandSession,
  workspaceRoot: string,
  paths: ReadonlySet<string>,
  retain: RetainParsedValue,
): Promise<unknown[]> {
  const result: unknown[] = []
  for (const path of [...paths].sort()) {
    session.throwIfStopped()
    const info = await lstat(join(workspaceRoot, ...path.split('/')), { bigint: true }).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return undefined
      throw error
    })
    session.throwIfStopped()
    const value = info ? {
      path, exists: true, type: info.isFile() ? 'file' : info.isDirectory() ? 'directory' : info.isSymbolicLink() ? 'symlink' : 'other',
      dev: info.dev.toString(), ino: info.ino.toString(), size: info.size.toString(),
      mtimeNs: info.mtimeNs.toString(), ctimeNs: info.ctimeNs.toString(), mode: info.mode.toString(),
    } : { path, exists: false }
    retain(value)
    result.push(value)
  }
  return result
}

// [maint-note:a2c7444c7d74]
// 维护导航：digest；下列为本函数源码中的接口与路径索引。
// 输入 parts：readonly Buffer[]。
// 返回契约：string。
// 遍历边界：const part ← parts。
// 返回路径：hash.digest('hex')。
// 直接调用：createHash。
// 直接调用：hash.update(Buffer.from(String(part.length))).update('\0').update。
// 直接调用：hash.update(Buffer.from(String(part.length))).update。
// 直接调用：hash.update。
// 直接调用：Buffer.from。
// 直接调用：String。
// 直接调用：hash.digest。
// [/maint-note:a2c7444c7d74]
function digest(parts: readonly Buffer[]): string {
  const hash = createHash('sha256')
  for (const part of parts) hash.update(Buffer.from(String(part.length))).update('\0').update(part)
  return hash.digest('hex')
}

// [maint-note:0b34d556671c]
// 维护导航：observationFingerprint；下列为本函数源码中的接口与路径索引。
// 输入 input：{ workspaceId: string request: GitChangesObservationRequest configBytes: readonly Buffer[] relevantIndex: readonly GitIndexEntry[] relevantUntracked: readonly string[] relevantAttributes: readonly { path: string; attribute: string; value: s。
// 返回契约：string。
// 遍历边界：const patch ← input.patches。
// 遍历边界：const hunk ← patch.hunks。
// 遍历边界：const line ← hunk.lines。
// 返回路径：hash.digest('hex')。
// 直接调用：createHash。
// 直接调用：addBuffer。
// 直接调用：Buffer.from。
// 直接调用：addJson。
// 直接调用：addBufferSequence。
// 直接调用：addSequence。
// 直接调用：hash.digest。
// [/maint-note:0b34d556671c]
function observationFingerprint(input: {
  workspaceId: string
  request: GitChangesObservationRequest
  configBytes: readonly Buffer[]
  relevantIndex: readonly GitIndexEntry[]
  relevantUntracked: readonly string[]
  relevantAttributes: readonly { path: string; attribute: string; value: string }[]
  relevantTree: readonly GitTreeEntry[]
  relevantStatus: readonly { paths: string[]; value: string }[]
  attributeSourceBytes: readonly Buffer[]
  headOid: string
  records: readonly GitFileRecord[]
  payloads: readonly Buffer[]
  patches: readonly GitPatchLayer[]
  stats: readonly unknown[]
}): string {
  const hash = createHash('sha256')
  // [maint-note:b06e02d8ed39]
  // 维护导航：addBuffer；下列为本函数源码中的接口与路径索引。
  // 输入 value：Buffer。
  // 返回契约：void。
  // 直接调用：hash.update(String(value.length)).update('\0').update。
  // 直接调用：hash.update(String(value.length)).update。
  // 直接调用：hash.update。
  // 直接调用：String。
  // [/maint-note:b06e02d8ed39]
  const addBuffer = (value: Buffer): void => {
    hash.update(String(value.length)).update('\0').update(value)
  }
  // [maint-note:aeb98fc62e37]
  // 维护导航：addJson；下列为本函数源码中的接口与路径索引。
  // 输入 value：unknown。
  // 返回契约：void。
  // 直接调用：addBuffer。
  // 直接调用：Buffer.from。
  // 直接调用：JSON.stringify。
  // [/maint-note:aeb98fc62e37]
  const addJson = (value: unknown): void => {
    addBuffer(Buffer.from(JSON.stringify(value), 'utf8'))
  }
  // [maint-note:516fe648ef42]
  // 维护导航：addSequence；下列为本函数源码中的接口与路径索引。
  // 输入 name：string。
  // 输入 values：readonly unknown[]。
  // 返回契约：void。
  // 遍历边界：const value ← values。
  // 直接调用：addJson。
  // [/maint-note:516fe648ef42]
  const addSequence = (name: string, values: readonly unknown[]): void => {
    addJson({ sequence: name, length: values.length })
    for (const value of values) addJson(value)
  }
  // [maint-note:77bf550cbd83]
  // 维护导航：addBufferSequence；下列为本函数源码中的接口与路径索引。
  // 输入 name：string。
  // 输入 values：readonly Buffer[]。
  // 返回契约：void。
  // 遍历边界：const value ← values。
  // 直接调用：addJson。
  // 直接调用：addBuffer。
  // [/maint-note:77bf550cbd83]
  const addBufferSequence = (name: string, values: readonly Buffer[]): void => {
    addJson({ sequence: name, length: values.length })
    for (const value of values) addBuffer(value)
  }

  addBuffer(Buffer.from(CONTRACT_VERSION))
  addBuffer(Buffer.from(input.workspaceId))
  addJson({ mode: input.request.mode, scope: input.request.scope, path: input.request.path ?? null })
  addBufferSequence('config', input.configBytes)
  addSequence('index', input.relevantIndex)
  addSequence('untracked', input.relevantUntracked)
  addSequence('attributes', input.relevantAttributes)
  addSequence('tree', input.relevantTree)
  addSequence('status', input.relevantStatus)
  addBufferSequence('attribute_sources', input.attributeSourceBytes)
  addBuffer(Buffer.from(input.headOid))
  addSequence('records', input.records)
  addBufferSequence('payloads', input.payloads)
  addJson({ sequence: 'patches', length: input.patches.length })
  for (const patch of input.patches) {
    addJson({ layer: patch.layer, file: patch.file, hunks: patch.hunks.length })
    for (const hunk of patch.hunks) {
      addJson({
        oldStart: hunk.oldStart,
        oldLines: hunk.oldLines,
        newStart: hunk.newStart,
        newLines: hunk.newLines,
        ...(hunk.section ? { section: hunk.section } : {}),
        lineCount: hunk.lines.length,
      })
      for (const line of hunk.lines) addJson(line)
    }
  }
  addSequence('stats', input.stats)
  return hash.digest('hex')
}

// [maint-note:d6c3b07f3ad3]
// 维护导航：slash；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 返回契约：string。
// 返回路径：path.split('\\').join('/')。
// 直接调用：path.split('\\').join。
// 直接调用：path.split。
// [/maint-note:d6c3b07f3ad3]
function slash(path: string): string {
  return path.split('\\').join('/')
}

// [maint-note:3348f3c53adf]
// 维护导航：readBoundedFile；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 path：string。
// 输入 maxBytes：number。
// 返回契约：Promise<Buffer>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!info.isFile()。
// 分支条件：info.size > maxBytes。
// 分支条件：remainingFileBytes <= 0。
// 分支条件：retainedCapacity <= 0。
// 分支条件：bytesRead === 0。
// 分支条件：total > maxBytes。
// 分支条件：chunks.length === 0。
// 分支条件：chunks.length === 1。
// 分支条件：primaryError === undefined。
// 遍历边界：; ; 。
// 返回路径：Buffer.alloc(0)。
// 返回路径：chunks[0]!。
// 返回路径：content。
// 显式抛出：error。
// 显式抛出：closeError。
// 写入目标：total。
// 写入目标：chunks.length。
// 写入目标：primaryError。
// 直接调用：session.beforeMetadataRead。
// 直接调用：open(path, flags).catch。
// 直接调用：open。
// 直接调用：session.throwIfStopped。
// 直接调用：handle.stat。
// 直接调用：info.isFile。
// 直接调用：unsupportedLayout。
// 直接调用：observationLimit。
// 直接调用：session.remainingRetainedBytes。
// 直接调用：session.consumeRetainedBytes。
// 直接调用：Math.min。
// 直接调用：Buffer.allocUnsafe。
// 直接调用：handle.read。
// 直接调用：chunks.push。
// 直接调用：buffer.subarray。
// 直接调用：Buffer.alloc。
// 直接调用：Buffer.concat。
// 直接调用：session.releaseRetainedBytes。
// 直接调用：handle.close。
// [/maint-note:3348f3c53adf]
async function readBoundedFile(
  session: GitCommandSession,
  path: string,
  maxBytes: number,
): Promise<Buffer> {
  session.beforeMetadataRead(path)
  const flags = constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0)
  const handle = await open(path, flags).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') throw error
    unsupportedLayout('unsafe_metadata_file')
  })
  let primaryError: unknown
  try {
    session.throwIfStopped()
    const info = await handle.stat()
    session.throwIfStopped()
    if (!info.isFile()) unsupportedLayout('unsafe_metadata_file')
    if (info.size > maxBytes) observationLimit('metadata_file_too_large')
    const chunks: Buffer[] = []
    let total = 0
    for (;;) {
      session.throwIfStopped()
      const remainingFileBytes = maxBytes + 1 - total
      if (remainingFileBytes <= 0) observationLimit('metadata_file_too_large')
      const retainedCapacity = session.remainingRetainedBytes()
      if (retainedCapacity <= 0) session.consumeRetainedBytes(1, 'metadata_probe')
      const expectedRemaining = info.size >= total ? info.size + 1 - total : 4_096
      const readSize = Math.min(4_096, remainingFileBytes, retainedCapacity, expectedRemaining)
      const buffer = Buffer.allocUnsafe(readSize)
      const { bytesRead } = await handle.read(buffer, 0, readSize, null)
      session.throwIfStopped()
      if (bytesRead === 0) break
      session.consumeRetainedBytes(bytesRead, 'metadata_chunk')
      total += bytesRead
      if (total > maxBytes) observationLimit('metadata_file_too_large')
      chunks.push(bytesRead === buffer.length ? buffer : buffer.subarray(0, bytesRead))
    }
    if (chunks.length === 0) return Buffer.alloc(0)
    if (chunks.length === 1) return chunks[0]!
    session.consumeRetainedBytes(total, 'metadata_concat')
    const content = Buffer.concat(chunks, total)
    chunks.length = 0
    session.releaseRetainedBytes(total, 'metadata_chunks')
    return content
  } catch (error) {
    primaryError = error
    throw error
  } finally {
    try {
      await handle.close()
    } catch (closeError) {
      if (primaryError === undefined) throw closeError
    }
    if (primaryError === undefined) session.throwIfStopped()
  }
}

// [maint-note:539f1bdd48d3]
// 维护导航：readOptionalBoundedFile；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 path：string。
// 输入 maxBytes：number。
// 返回契约：Promise<Buffer | undefined>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：(error as NodeJS.ErrnoException).code === 'ENOENT'。
// 返回路径：await readBoundedFile(session, path, maxBytes)。
// 返回路径：undefined。
// 显式抛出：error。
// 直接调用：readBoundedFile。
// [/maint-note:539f1bdd48d3]
async function readOptionalBoundedFile(
  session: GitCommandSession,
  path: string,
  maxBytes: number,
): Promise<Buffer | undefined> {
  try {
    return await readBoundedFile(session, path, maxBytes)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}

// [maint-note:b7a9285bf46d]
// 维护导航：existsRegularFile；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 path：string。
// 返回契约：Promise<boolean>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!info。
// 分支条件：!info.isFile() || info.isSymbolicLink()。
// 返回路径：false。
// 返回路径：true。
// 直接调用：session.throwIfStopped。
// 直接调用：lstat(path).catch。
// 直接调用：lstat。
// 直接调用：info.isFile。
// 直接调用：info.isSymbolicLink。
// 直接调用：unsupportedLayout。
// [/maint-note:b7a9285bf46d]
async function existsRegularFile(session: GitCommandSession, path: string): Promise<boolean> {
  session.throwIfStopped()
  const info = await lstat(path).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT') return undefined
    throw error
  })
  session.throwIfStopped()
  if (!info) return false
  if (!info.isFile() || info.isSymbolicLink()) unsupportedLayout('unsafe_worktree_config')
  return true
}

// [maint-note:8cc852523163]
// 维护导航：statIdentity；下列为本函数源码中的接口与路径索引。
// 输入 session：GitCommandSession。
// 输入 path：string。
// 返回契约：Promise<Record<string, string>>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：{ dev: info.dev.toString(), ino: info.ino.toString(), mode: info.mode.toString() }。
// 直接调用：session.throwIfStopped。
// 直接调用：lstat。
// 直接调用：info.dev.toString。
// 直接调用：info.ino.toString。
// 直接调用：info.mode.toString。
// [/maint-note:8cc852523163]
async function statIdentity(session: GitCommandSession, path: string): Promise<Record<string, string>> {
  session.throwIfStopped()
  const info = await lstat(path, { bigint: true })
  session.throwIfStopped()
  return { dev: info.dev.toString(), ino: info.ino.toString(), mode: info.mode.toString() }
}

// [maint-note:dad0c7add349]
// 维护导航：layerRank；下列为本函数源码中的接口与路径索引。
// 输入 layer：GitChangeLayer。
// 返回契约：number。
// 返回路径：layer === 'staged' ? 0 : layer === 'unstaged' ? 1 : 2。
// [/maint-note:dad0c7add349]
function layerRank(layer: GitChangeLayer): number {
  return layer === 'staged' ? 0 : layer === 'unstaged' ? 1 : 2
}

// [maint-note:dcb7a3735f9c]
// 维护导航：configReason；下列为本函数源码中的接口与路径索引。
// 输入 key：string。
// 返回契约：string。
// 分支条件：key.startsWith('filter.')。
// 分支条件：key.startsWith('include')。
// 分支条件：key.includes('textconv')。
// 分支条件：key.includes('pager')。
// 返回路径：'executable_filter_config'。
// 返回路径：'include_config'。
// 返回路径：'textconv_config'。
// 返回路径：'pager_config'。
// 返回路径：'executable_git_config'。
// 直接调用：key.startsWith。
// 直接调用：key.includes。
// [/maint-note:dcb7a3735f9c]
function configReason(key: string): string {
  if (key.startsWith('filter.')) return 'executable_filter_config'
  if (key.startsWith('include')) return 'include_config'
  if (key.includes('textconv')) return 'textconv_config'
  if (key.includes('pager')) return 'pager_config'
  return 'executable_git_config'
}

// [maint-note:26dbc0825562]
// 维护导航：unsupportedLayout；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：never。
// 显式抛出：new GitChangesError('git_changes_unsupported_repository_layout', 'Unsupported Git repository layout', reason)。
// [/maint-note:26dbc0825562]
function unsupportedLayout(reason: string): never {
  throw new GitChangesError('git_changes_unsupported_repository_layout', 'Unsupported Git repository layout', reason)
}

// [maint-note:fa114d4ce137]
// 维护导航：unsupportedConfig；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：never。
// 显式抛出：new GitChangesError('git_changes_unsupported_repository_config', 'Unsupported Git repository configuration', reason)。
// [/maint-note:fa114d4ce137]
function unsupportedConfig(reason: string): never {
  throw new GitChangesError('git_changes_unsupported_repository_config', 'Unsupported Git repository configuration', reason)
}

// [maint-note:1c55a1e2b13e]
// 维护导航：unsupportedAttributes；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：never。
// 显式抛出：new GitChangesError('git_changes_unsupported_attributes', 'Unsupported Git attributes', reason)。
// [/maint-note:1c55a1e2b13e]
function unsupportedAttributes(reason: string): never {
  throw new GitChangesError('git_changes_unsupported_attributes', 'Unsupported Git attributes', reason)
}

// [maint-note:a5a41b391aed]
// 维护导航：observationLimit；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：never。
// 显式抛出：new GitChangesError('git_changes_observation_limit', 'Git observation exceeded a bounded limit', reason)。
// [/maint-note:a5a41b391aed]
function observationLimit(reason: string): never {
  throw new GitChangesError('git_changes_observation_limit', 'Git observation exceeded a bounded limit', reason)
}

// [maint-note:c07892010734]
// 维护导航：observationChanged；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：never。
// 显式抛出：new GitChangesError( 'git_changes_observation_changed', 'Git observation changed while it was being read', reason, true, )。
// [/maint-note:c07892010734]
function observationChanged(reason: string): never {
  throw new GitChangesError(
    'git_changes_observation_changed',
    'Git observation changed while it was being read',
    reason,
    true,
  )
}

// [maint-note:55f0e686a362]
// 维护导航：equalBuffers；下列为本函数源码中的接口与路径索引。
// 输入 left：readonly Buffer[]。
// 输入 right：readonly Buffer[]。
// 返回契约：boolean。
// 返回路径：left.length === right.length && left.every((value, index) => value.equals(right[index]!))。
// 直接调用：left.every。
// [/maint-note:55f0e686a362]
function equalBuffers(left: readonly Buffer[], right: readonly Buffer[]): boolean {
  return left.length === right.length && left.every((value, index) => value.equals(right[index]!))
}

// [maint-note:f31b3be2a520]
// 维护导航：totalBytes；下列为本函数源码中的接口与路径索引。
// 输入 values：readonly Buffer[]。
// 返回契约：number。
// 返回路径：values.reduce((total, value) => total + value.length, 0)。
// 直接调用：values.reduce。
// [/maint-note:f31b3be2a520]
function totalBytes(values: readonly Buffer[]): number {
  return values.reduce((total, value) => total + value.length, 0)
}

// [maint-note:d1160e154aff]
// 维护导航：retainedValueBytes；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：number。
// 返回路径：jsonEncodedBytes(value) + 16。
// 直接调用：jsonEncodedBytes。
// [/maint-note:d1160e154aff]
function retainedValueBytes(value: unknown): number {
  return jsonEncodedBytes(value) + 16
}

// [maint-note:d986ebbaabfa]
// 维护导航：jsonEncodedBytes；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 seen：类型由调用方与函数体约束；缺省表达式 new WeakSet<object>()。
// 返回契约：number。
// 分支条件：value === null || value === undefined。
// 分支条件：typeof value === 'string'。
// 分支条件：typeof value === 'number'。
// 分支条件：typeof value === 'boolean'。
// 分支条件：typeof value !== 'object'。
// 分支条件：seen.has(value)。
// 分支条件：Array.isArray(value)。
// 分支条件：count++ > 0。
// 分支条件：item === undefined || typeof item === 'function' || typeof item === 'symbol'。
// 遍历边界：const item ← value。
// 遍历边界：const [key, item] ← Object.entries(value)。
// 返回路径：4。
// 返回路径：jsonStringBytes(value)。
// 返回路径：Number.isFinite(value) ? String(value).length : 4。
// 返回路径：value ? 4 : 5。
// 返回路径：0。
// 返回路径：bytes。
// 写入目标：bytes。
// 直接调用：jsonStringBytes。
// 直接调用：Number.isFinite。
// 直接调用：String。
// 直接调用：seen.has。
// 直接调用：observationLimit。
// 直接调用：seen.add。
// 直接调用：Array.isArray。
// 直接调用：jsonEncodedBytes。
// 直接调用：Object.entries。
// 直接调用：seen.delete。
// [/maint-note:d986ebbaabfa]
function jsonEncodedBytes(value: unknown, seen = new WeakSet<object>()): number {
  if (value === null || value === undefined) return 4
  if (typeof value === 'string') return jsonStringBytes(value)
  if (typeof value === 'number') return Number.isFinite(value) ? String(value).length : 4
  if (typeof value === 'boolean') return value ? 4 : 5
  if (typeof value !== 'object') return 0
  if (seen.has(value)) observationLimit('cyclic_retained_value')
  seen.add(value)
  let bytes = 2
  let count = 0
  if (Array.isArray(value)) {
    for (const item of value) {
      if (count++ > 0) bytes += 1
      bytes += jsonEncodedBytes(item, seen)
    }
  } else {
    for (const [key, item] of Object.entries(value)) {
      if (item === undefined || typeof item === 'function' || typeof item === 'symbol') continue
      if (count++ > 0) bytes += 1
      bytes += jsonStringBytes(key) + 1 + jsonEncodedBytes(item, seen)
    }
  }
  seen.delete(value)
  return bytes
}

// [maint-note:2dffe161de2e]
// 维护导航：jsonStringBytes；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 分支条件：character === '"' || character === '\\' || character === '\b' || character === '\f' || character === '\n' || character === '\r' || character === '\t'。
// 分支条件：code < 0x20。
// 遍历边界：const character ← value。
// 返回路径：bytes。
// 写入目标：bytes。
// 直接调用：character.codePointAt。
// 直接调用：Buffer.byteLength。
// [/maint-note:2dffe161de2e]
function jsonStringBytes(value: string): number {
  let bytes = 2
  for (const character of value) {
    const code = character.codePointAt(0)!
    if (character === '"' || character === '\\' || character === '\b' || character === '\f' || character === '\n' || character === '\r' || character === '\t') {
      bytes += 2
    } else if (code < 0x20) {
      bytes += 6
    } else {
      bytes += Buffer.byteLength(character, 'utf8')
    }
  }
  return bytes
}

// [maint-note:1b5002ba2665]
// 维护导航：estimateMaterializedBytes；下列为本函数源码中的接口与路径索引。
// 输入 records：readonly GitFileRecord[]。
// 输入 patches：readonly GitPatchLayer[]。
// 输入 guidance：string | undefined。
// 输入 coverage：NonNullable<GitChangesObservation['coverage']>。
// 返回契约：number。
// 分支条件：guidance。
// 遍历边界：const record ← records。
// 遍历边界：const patch ← patches。
// 遍历边界：const hunk ← patch.hunks。
// 遍历边界：const line ← hunk.lines。
// 返回路径：total。
// 直接调用：retainedValueBytes。
// 直接调用：add。
// [/maint-note:1b5002ba2665]
function estimateMaterializedBytes(
  records: readonly GitFileRecord[],
  patches: readonly GitPatchLayer[],
  guidance: string | undefined,
  coverage: NonNullable<GitChangesObservation['coverage']>,
): number {
  let total = retainedValueBytes(coverage) + 128
  // [maint-note:9b37f1003f1a]
  // 维护导航：add；下列为本函数源码中的接口与路径索引。
  // 输入 value：unknown。
  // 返回契约：void。
  // 分支条件：!Number.isSafeInteger(total + bytes)。
  // 写入目标：total。
  // 直接调用：retainedValueBytes。
  // 直接调用：Number.isSafeInteger。
  // 直接调用：observationLimit。
  // [/maint-note:9b37f1003f1a]
  const add = (value: unknown): void => {
    const bytes = retainedValueBytes(value)
    if (!Number.isSafeInteger(total + bytes)) observationLimit('retained_bytes_exceeded')
    total += bytes
  }
  if (guidance) add(guidance)
  for (const record of records) add(record)
  for (const patch of patches) {
    add({ layer: patch.layer, file: patch.file })
    for (const hunk of patch.hunks) {
      add({
        oldStart: hunk.oldStart,
        oldLines: hunk.oldLines,
        newStart: hunk.newStart,
        newLines: hunk.newLines,
        ...(hunk.section ? { section: hunk.section } : {}),
      })
      for (const line of hunk.lines) add(line)
    }
  }
  return total
}

// [maint-note:aedc60da9a80]
// 维护导航：decodeMetadataText；下列为本函数源码中的接口与路径索引。
// 输入 value：Buffer。
// 输入 reason：string。
// 返回契约：string。
// 返回路径：new TextDecoder('utf-8', { fatal: true }).decode(value)。
// 直接调用：new TextDecoder('utf-8', { fatal: true }).decode。
// 直接调用：unsupportedLayout。
// [/maint-note:aedc60da9a80]
function decodeMetadataText(value: Buffer, reason: string): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(value)
  } catch {
    unsupportedLayout(reason)
  }
}

// [maint-note:3558f0cfbe7b]
// 维护导航：decodeRepositoryPointer；下列为本函数源码中的接口与路径索引。
// 输入 value：Buffer。
// 输入 reason：string。
// 返回契约：string。
// 分支条件：!match || match[1]!.trim().length === 0。
// 返回路径：match[1]!。
// 直接调用：decodeMetadataText。
// 直接调用：/^([^\r\n]+)\r?\n?$/u.exec。
// 直接调用：match[1]!.trim。
// 直接调用：unsupportedLayout。
// [/maint-note:3558f0cfbe7b]
function decodeRepositoryPointer(value: Buffer, reason: string): string {
  const text = decodeMetadataText(value, reason)
  const match = /^([^\r\n]+)\r?\n?$/u.exec(text)
  if (!match || match[1]!.trim().length === 0) unsupportedLayout(reason)
  return match[1]!
}
