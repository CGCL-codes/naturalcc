// [maint-note:46b8cd3acc26]
// 数据契约导航：GitChangeScope；字段含义以消费端校验为准。
// 类型组合：'staged' | 'unstaged' | 'both'。
// [/maint-note:46b8cd3acc26]
export type GitChangeScope = 'staged' | 'unstaged' | 'both'
// [maint-note:4dd57f7d1603]
// 数据契约导航：GitChangeMode；字段含义以消费端校验为准。
// 类型组合：'summary' | 'files' | 'patch'。
// [/maint-note:4dd57f7d1603]
export type GitChangeMode = 'summary' | 'files' | 'patch'
// [maint-note:6686d74ab190]
// 数据契约导航：GitChangeLayer；字段含义以消费端校验为准。
// 类型组合：'staged' | 'unstaged' | 'unmerged'。
// [/maint-note:6686d74ab190]
export type GitChangeLayer = 'staged' | 'unstaged' | 'unmerged'
// [maint-note:d01d80cc7da1]
// 数据契约导航：GitChangeAction；字段含义以消费端校验为准。
// 类型组合：| 'added' | 'modified' | 'deleted' | 'renamed' | 'untracked' | 'type_changed' | 'unmerged'。
// [/maint-note:d01d80cc7da1]
export type GitChangeAction =
  | 'added'
  | 'modified'
  | 'deleted'
  | 'renamed'
  | 'untracked'
  | 'type_changed'
  | 'unmerged'

// [maint-note:66cd1c81dcc9]
// 数据契约导航：GitFileRecord；字段含义以消费端校验为准。
// 成员 layer：GitChangeLayer。
// 成员 action：GitChangeAction。
// 成员 path：string。
// 成员 oldPath（可省略）：string。
// 成员 boundaryTruncated（可省略）：boolean。
// 成员 contentChanged：boolean | 'unknown'。
// 成员 binary：boolean | 'unknown'。
// 成员 modeChanged：boolean。
// 成员 oldMode（可省略）：string。
// 成员 newMode（可省略）：string。
// 成员 objectType：'file' | 'symlink' | 'gitlink' | 'other'。
// 成员 conflictStages（可省略）：Array<1 | 2 | 3>。
// [/maint-note:66cd1c81dcc9]
export interface GitFileRecord {
  layer: GitChangeLayer
  action: GitChangeAction
  path: string
  oldPath?: string
  boundaryTruncated?: boolean
  contentChanged: boolean | 'unknown'
  binary: boolean | 'unknown'
  modeChanged: boolean
  oldMode?: string
  newMode?: string
  objectType: 'file' | 'symlink' | 'gitlink' | 'other'
  conflictStages?: Array<1 | 2 | 3>
}

// [maint-note:b28fcf424af8]
// 数据契约导航：GitPatchLine；字段含义以消费端校验为准。
// 成员 kind：'context' | 'add' | 'delete' | 'no_newline'。
// 成员 text（可省略）：string。
// 成员 oldLine（可省略）：number | null。
// 成员 newLine（可省略）：number | null。
// 成员 appliesTo（可省略）：'old' | 'new'。
// [/maint-note:b28fcf424af8]
export interface GitPatchLine {
  kind: 'context' | 'add' | 'delete' | 'no_newline'
  text?: string
  oldLine?: number | null
  newLine?: number | null
  appliesTo?: 'old' | 'new'
}

// [maint-note:37ca041d8c36]
// 数据契约导航：GitPatchHunk；字段含义以消费端校验为准。
// 成员 oldStart：number。
// 成员 oldLines：number。
// 成员 newStart：number。
// 成员 newLines：number。
// 成员 section（可省略）：string。
// 成员 continuedFromPrevious（可省略）：boolean。
// 成员 continuesNext（可省略）：boolean。
// 成员 lines：GitPatchLine[]。
// [/maint-note:37ca041d8c36]
export interface GitPatchHunk {
  oldStart: number
  oldLines: number
  newStart: number
  newLines: number
  section?: string
  continuedFromPrevious?: boolean
  continuesNext?: boolean
  lines: GitPatchLine[]
}

// [maint-note:2aac4d1e4016]
// 数据契约导航：GitPatchLayer；字段含义以消费端校验为准。
// 成员 layer：Exclude<GitChangeLayer, 'unmerged'>。
// 成员 file：GitFileRecord。
// 成员 hunks：GitPatchHunk[]。
// [/maint-note:2aac4d1e4016]
export interface GitPatchLayer {
  layer: Exclude<GitChangeLayer, 'unmerged'>
  file: GitFileRecord
  hunks: GitPatchHunk[]
}

// [maint-note:a60543f23f38]
// 数据契约导航：GitChangesCoverage；字段含义以消费端校验为准。
// 成员 parentRepositoryChanges：'complete'。
// 成员 submoduleWorktrees：'not_present' | 'not_observed'。
// 成员 gitlinkCount：number。
// [/maint-note:a60543f23f38]
export interface GitChangesCoverage {
  parentRepositoryChanges: 'complete'
  submoduleWorktrees: 'not_present' | 'not_observed'
  gitlinkCount: number
}

// [maint-note:306f79f111be]
// 数据契约导航：GitChangesObservation；字段含义以消费端校验为准。
// 成员 applicable：boolean。
// 成员 reason（可省略）：'not_git_repository'。
// 成员 workspaceId（可省略）：string。
// 成员 revision（可省略）：string。
// 成员 coverage（可省略）：GitChangesCoverage。
// 成员 records（可省略）：GitFileRecord[]。
// 成员 patches（可省略）：GitPatchLayer[]。
// 成员 guidance（可省略）：string。
// [/maint-note:306f79f111be]
export interface GitChangesObservation {
  applicable: boolean
  reason?: 'not_git_repository'
  workspaceId?: string
  revision?: string
  coverage?: GitChangesCoverage
  records?: GitFileRecord[]
  patches?: GitPatchLayer[]
  guidance?: string
}

// [maint-note:7f194d1cef98]
// 数据契约导航：GitChangesObservationRequest；字段含义以消费端校验为准。
// 成员 projectDir：string。
// 成员 mode：GitChangeMode。
// 成员 scope：GitChangeScope。
// 成员 path（可省略）：string。
// 成员 signal（可省略）：AbortSignal。
// 成员 timeoutMs（可省略）：number。
// 成员 outputReserveBytes（可省略）：number。
// [/maint-note:7f194d1cef98]
export interface GitChangesObservationRequest {
  projectDir: string
  mode: GitChangeMode
  scope: GitChangeScope
  path?: string
  signal?: AbortSignal
  timeoutMs?: number
  /** UTF-8 bytes reserved for the canonical page/envelope built by the Tool. */
  outputReserveBytes?: number
}

// [maint-note:52bf45b51c22]
// 数据契约导航：GitChangesBackend；字段含义以消费端校验为准。
// 成员 observe：Promise<GitChangesObservation>。
// 成员 sealCursor：string。
// 成员 openCursor：string | undefined。
// [/maint-note:52bf45b51c22]
export interface GitChangesBackend {
  observe(request: GitChangesObservationRequest): Promise<GitChangesObservation>
  sealCursor(payload: string): string
  openCursor(cursor: string): string | undefined
}

// [maint-note:8c8e9613953d]
// 数据契约导航：GitChangesErrorCode；字段含义以消费端校验为准。
// 类型组合：| 'git_changes_unsupported_git' | 'git_changes_unsupported_repository_layout' | 'git_changes_unsupported_repository_config' | 'git_changes_unsupported_attributes' | 'git_changes_missing_object' | 'git_changes_unsupported_path_type' | 'git_c。
// [/maint-note:8c8e9613953d]
export type GitChangesErrorCode =
  | 'git_changes_unsupported_git'
  | 'git_changes_unsupported_repository_layout'
  | 'git_changes_unsupported_repository_config'
  | 'git_changes_unsupported_attributes'
  | 'git_changes_missing_object'
  | 'git_changes_unsupported_path_type'
  | 'git_changes_unsupported_submodule_path'
  | 'git_changes_unsupported_cross_boundary_rename'
  | 'git_changes_unsupported_conflict'
  | 'git_changes_ambiguous_change'
  | 'git_changes_patch_protocol_error'
  | 'git_changes_invalid_cursor'
  | 'git_changes_stale_cursor'
  | 'git_changes_observation_changed'
  | 'git_changes_observation_limit'
  | 'git_changes_line_too_large'
  | 'git_changes_timeout'
  | 'git_changes_invalid_path_encoding'
  | 'git_changes_invalid_content_encoding'
  | 'git_changes_git_failed'

export class GitChangesError extends Error {
  // [maint-note:b820f99e255a]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 code：GitChangesErrorCode。
  // 输入 message：string。
  // 输入 reason：string。
  // 输入 recoverable：类型由调用方与函数体约束；缺省表达式 false。
  // 写入目标：this.name。
  // 直接调用：super。
  // [/maint-note:b820f99e255a]
  constructor(
    readonly code: GitChangesErrorCode,
    message: string,
    readonly reason: string,
    readonly recoverable = false,
  ) {
    super(message)
    this.name = 'GitChangesError'
  }
}
