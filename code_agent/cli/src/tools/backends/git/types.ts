export type GitChangeScope = 'staged' | 'unstaged' | 'both'
export type GitChangeMode = 'summary' | 'files' | 'patch'
export type GitChangeLayer = 'staged' | 'unstaged' | 'unmerged'
export type GitChangeAction =
  | 'added'
  | 'modified'
  | 'deleted'
  | 'renamed'
  | 'untracked'
  | 'type_changed'
  | 'unmerged'

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

export interface GitPatchLine {
  kind: 'context' | 'add' | 'delete' | 'no_newline'
  text?: string
  oldLine?: number | null
  newLine?: number | null
  appliesTo?: 'old' | 'new'
}

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

export interface GitPatchLayer {
  layer: Exclude<GitChangeLayer, 'unmerged'>
  file: GitFileRecord
  hunks: GitPatchHunk[]
}

export interface GitChangesCoverage {
  parentRepositoryChanges: 'complete'
  submoduleWorktrees: 'not_present' | 'not_observed'
  gitlinkCount: number
}

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

export interface GitChangesBackend {
  observe(request: GitChangesObservationRequest): Promise<GitChangesObservation>
  sealCursor(payload: string): string
  openCursor(cursor: string): string | undefined
}

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
