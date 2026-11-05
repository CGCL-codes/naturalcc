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

interface RepositoryLayout {
  workspaceRoot: string
  repoRoot: string
  gitDir: string
  commonDir: string
  workspacePrefix: string
  workspaceId: string
  configFiles: string[]
}

interface Capture {
  records: GitFileRecord[]
  patches: GitPatchLayer[]
  guidance?: string
  coverage: NonNullable<GitChangesObservation['coverage']>
  fingerprint: string
  retainedBytes: number
}

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

export function createGitChangesBackend(options: GitChangesBackendOptions = {}): GitChangesBackend {
  const secret = Buffer.from(options.cursorSecret ?? randomBytes(32))
  if (secret.length < 16) throw new Error('git_changes cursor secret must contain at least 16 bytes')

  return {
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
    sealCursor(payload) {
      const encoded = Buffer.from(payload, 'utf8').toString('base64url')
      const signature = createHmac('sha256', secret).update(encoded).digest('base64url')
      return `gcv1.${encoded}.${signature}`
    },
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

async function checkGitCapability(session: GitCommandSession, cwd: string): Promise<void> {
  const result = await session.run(['--version'], cwd)
  const supported = result.exitCode === 0 && /^git version \d+\.\d+(?:\.\d+)?(?:\s|$)/u.test(result.stdout.toString('utf8').trim())
  session.releaseRetainedBytes(result.stdout.length)
  if (!supported) {
    throw new GitChangesError('git_changes_unsupported_git', 'Git version check failed', 'unsupported_git_version')
  }
}

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

async function captureObservation(
  session: GitCommandSession,
  repository: RepositoryLayout,
  request: GitChangesObservationRequest,
  configBytes: readonly Buffer[],
  materialize: boolean,
): Promise<Capture> {
  session.throwIfStopped()
  let parsedRetainedBytes = 0
  const retainParsedValue: RetainParsedValue = (value) => {
    const bytes = retainedValueBytes(value)
    session.consumeRetainedBytes(bytes, 'parsed_unit')
    parsedRetainedBytes += bytes
  }
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

function assertGitSuccess(result: { exitCode: number; stderr: Buffer }, operation: string): void {
  if (result.exitCode === 0) return
  const diagnostic = result.stderr.toString('utf8')
  if (/missing blob|missing tree|unable to read|could not fetch|promisor|bad object/iu.test(diagnostic)) {
    throw new GitChangesError('git_changes_missing_object', 'Required Git object is unavailable locally', 'missing_object')
  }
  throw new GitChangesError('git_changes_git_failed', `Git ${operation} failed`, `exit_${result.exitCode}`)
}

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

function inspectAttributes(values: readonly { path: string; attribute: string; value: string }[]): void {
  for (const item of values) {
    const value = item.value.toLowerCase()
    if (item.attribute === 'filter' && !['unspecified', 'unset'].includes(value)) unsupportedAttributes('filter_attribute')
    if (item.attribute === 'working-tree-encoding' && !['unspecified', 'unset'].includes(value)) unsupportedAttributes('working_tree_encoding')
    if (item.attribute === 'diff' && !['unspecified', 'unset', 'set'].includes(value)) unsupportedAttributes('custom_diff_driver')
  }
}

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

function requestedLayers(scope: GitChangesObservationRequest['scope']): Array<'staged' | 'unstaged'> {
  return scope === 'both' ? ['staged', 'unstaged'] : [scope]
}

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

function isRelevantRepositoryPath(
  repository: RepositoryLayout,
  repoPath: string,
  request: GitChangesObservationRequest,
  records: readonly GitFileRecord[],
): boolean {
  const local = mapRepositoryPath(repository, repoPath)
  return local !== undefined && isRelevantWorkspacePath(local, request, records)
}

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

function toRepoPath(repository: RepositoryLayout, workspacePath: string): string {
  return repository.workspacePrefix === '.' ? workspacePath : `${repository.workspacePrefix}/${workspacePath}`
}

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

function digest(parts: readonly Buffer[]): string {
  const hash = createHash('sha256')
  for (const part of parts) hash.update(Buffer.from(String(part.length))).update('\0').update(part)
  return hash.digest('hex')
}

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
  const addBuffer = (value: Buffer): void => {
    hash.update(String(value.length)).update('\0').update(value)
  }
  const addJson = (value: unknown): void => {
    addBuffer(Buffer.from(JSON.stringify(value), 'utf8'))
  }
  const addSequence = (name: string, values: readonly unknown[]): void => {
    addJson({ sequence: name, length: values.length })
    for (const value of values) addJson(value)
  }
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

function slash(path: string): string {
  return path.split('\\').join('/')
}

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

async function statIdentity(session: GitCommandSession, path: string): Promise<Record<string, string>> {
  session.throwIfStopped()
  const info = await lstat(path, { bigint: true })
  session.throwIfStopped()
  return { dev: info.dev.toString(), ino: info.ino.toString(), mode: info.mode.toString() }
}

function layerRank(layer: GitChangeLayer): number {
  return layer === 'staged' ? 0 : layer === 'unstaged' ? 1 : 2
}

function configReason(key: string): string {
  if (key.startsWith('filter.')) return 'executable_filter_config'
  if (key.startsWith('include')) return 'include_config'
  if (key.includes('textconv')) return 'textconv_config'
  if (key.includes('pager')) return 'pager_config'
  return 'executable_git_config'
}

function unsupportedLayout(reason: string): never {
  throw new GitChangesError('git_changes_unsupported_repository_layout', 'Unsupported Git repository layout', reason)
}

function unsupportedConfig(reason: string): never {
  throw new GitChangesError('git_changes_unsupported_repository_config', 'Unsupported Git repository configuration', reason)
}

function unsupportedAttributes(reason: string): never {
  throw new GitChangesError('git_changes_unsupported_attributes', 'Unsupported Git attributes', reason)
}

function observationLimit(reason: string): never {
  throw new GitChangesError('git_changes_observation_limit', 'Git observation exceeded a bounded limit', reason)
}

function observationChanged(reason: string): never {
  throw new GitChangesError(
    'git_changes_observation_changed',
    'Git observation changed while it was being read',
    reason,
    true,
  )
}

function equalBuffers(left: readonly Buffer[], right: readonly Buffer[]): boolean {
  return left.length === right.length && left.every((value, index) => value.equals(right[index]!))
}

function totalBytes(values: readonly Buffer[]): number {
  return values.reduce((total, value) => total + value.length, 0)
}

function retainedValueBytes(value: unknown): number {
  return jsonEncodedBytes(value) + 16
}

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

function estimateMaterializedBytes(
  records: readonly GitFileRecord[],
  patches: readonly GitPatchLayer[],
  guidance: string | undefined,
  coverage: NonNullable<GitChangesObservation['coverage']>,
): number {
  let total = retainedValueBytes(coverage) + 128
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

function decodeMetadataText(value: Buffer, reason: string): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(value)
  } catch {
    unsupportedLayout(reason)
  }
}

function decodeRepositoryPointer(value: Buffer, reason: string): string {
  const text = decodeMetadataText(value, reason)
  const match = /^([^\r\n]+)\r?\n?$/u.exec(text)
  if (!match || match[1]!.trim().length === 0) unsupportedLayout(reason)
  return match[1]!
}
