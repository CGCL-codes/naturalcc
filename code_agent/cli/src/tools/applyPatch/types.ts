export type PatchOperation =
  | { kind: 'add'; path: string; lines: string[] }
  | { kind: 'delete'; path: string }
  | { kind: 'update'; path: string; hunks: PatchHunk[] }

export interface PatchHunk {
  oldStart: number
  lines: string[]
}

export interface ApplyPatchInput {
  patch: string
  reason: string
}

export interface ParsedPatchHunkHeader {
  oldStart: number
}

export interface FileSnapshot {
  exists: boolean
  content?: Buffer
  mode?: number
}

export interface PatchChange {
  operation: PatchOperation
  path: string
  content?: string
  additions: number
  deletions: number
  before: FileSnapshot
  after?: FileSnapshot
  /**
   * display records computed from the same canonical hunk cursor the
   * executor applies. Neutral text only; attached to the ToolResult on
   * success and never re-derived from formatted output.
   */
  records?: ApplyPatchDisplayRecord[]
}

/** One neutral diff record; kind + verbatim text + applied line numbers. */
export interface ApplyPatchDisplayRecord {
  kind: 'context' | 'added' | 'removed'
  text: string
  /** 1-based old-file line for removed records; undefined for added. */
  oldLine?: number
  /** 1-based new-file line for added/context records; undefined for removed. */
  newLine?: number
}

export interface ApplyPatchDisplayFile {
  path: string
  action: 'add' | 'update' | 'delete'
  additions: number
  deletions: number
  records: readonly ApplyPatchDisplayRecord[]
}

export interface ApplyPatchDisplayPayload {
  files: readonly ApplyPatchDisplayFile[]
  /**
   * True when the display-fact builder itself dropped records/characters to
   * stay bounded. Independent of the canonical content budget.
   */
  displayTruncated: boolean
}

export interface ApplyPatchDependencies {
  /** Test-only transaction seam; deliberately private to apply_patch. */
  onChangeApplied?: (path: string) => void | Promise<void>
}
