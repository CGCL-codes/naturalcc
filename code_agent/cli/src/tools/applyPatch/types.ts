// [maint-note:b076dcc57b28]
// 数据契约导航：PatchOperation；字段含义以消费端校验为准。
// 类型组合：| { kind: 'add'; path: string; lines: string[] } | { kind: 'delete'; path: string } | { kind: 'update'; path: string; hunks: PatchHunk[] }。
// [/maint-note:b076dcc57b28]
export type PatchOperation =
  | { kind: 'add'; path: string; lines: string[] }
  | { kind: 'delete'; path: string }
  | { kind: 'update'; path: string; hunks: PatchHunk[] }

// [maint-note:8d02a036671b]
// 数据契约导航：PatchHunk；字段含义以消费端校验为准。
// 成员 oldStart：number。
// 成员 lines：string[]。
// [/maint-note:8d02a036671b]
export interface PatchHunk {
  oldStart: number
  lines: string[]
}

// [maint-note:3b7020c7d946]
// 数据契约导航：ApplyPatchInput；字段含义以消费端校验为准。
// 成员 patch：string。
// 成员 reason：string。
// [/maint-note:3b7020c7d946]
export interface ApplyPatchInput {
  patch: string
  reason: string
}

// [maint-note:c01b3365d505]
// 数据契约导航：ParsedPatchHunkHeader；字段含义以消费端校验为准。
// 成员 oldStart：number。
// [/maint-note:c01b3365d505]
export interface ParsedPatchHunkHeader {
  oldStart: number
}

// [maint-note:1468204b6324]
// 数据契约导航：FileSnapshot；字段含义以消费端校验为准。
// 成员 exists：boolean。
// 成员 content（可省略）：Buffer。
// 成员 mode（可省略）：number。
// [/maint-note:1468204b6324]
export interface FileSnapshot {
  exists: boolean
  content?: Buffer
  mode?: number
}

// [maint-note:93d1cdb83e70]
// 数据契约导航：PatchChange；字段含义以消费端校验为准。
// 成员 operation：PatchOperation。
// 成员 path：string。
// 成员 content（可省略）：string。
// 成员 additions：number。
// 成员 deletions：number。
// 成员 before：FileSnapshot。
// 成员 after（可省略）：FileSnapshot。
// 成员 records（可省略）：ApplyPatchDisplayRecord[]。
// [/maint-note:93d1cdb83e70]
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
// [maint-note:ffc4b4593be6]
// 数据契约导航：ApplyPatchDisplayRecord；字段含义以消费端校验为准。
// 成员 kind：'context' | 'added' | 'removed'。
// 成员 text：string。
// 成员 oldLine（可省略）：number。
// 成员 newLine（可省略）：number。
// [/maint-note:ffc4b4593be6]
export interface ApplyPatchDisplayRecord {
  kind: 'context' | 'added' | 'removed'
  text: string
  /** 1-based old-file line for removed records; undefined for added. */
  oldLine?: number
  /** 1-based new-file line for added/context records; undefined for removed. */
  newLine?: number
}

// [maint-note:b633de229087]
// 数据契约导航：ApplyPatchDisplayFile；字段含义以消费端校验为准。
// 成员 path：string。
// 成员 action：'add' | 'update' | 'delete'。
// 成员 additions：number。
// 成员 deletions：number。
// 成员 records：readonly ApplyPatchDisplayRecord[]。
// [/maint-note:b633de229087]
export interface ApplyPatchDisplayFile {
  path: string
  action: 'add' | 'update' | 'delete'
  additions: number
  deletions: number
  records: readonly ApplyPatchDisplayRecord[]
}

// [maint-note:d61536679575]
// 数据契约导航：ApplyPatchDisplayPayload；字段含义以消费端校验为准。
// 成员 files：readonly ApplyPatchDisplayFile[]。
// 成员 displayTruncated：boolean。
// [/maint-note:d61536679575]
export interface ApplyPatchDisplayPayload {
  files: readonly ApplyPatchDisplayFile[]
  /**
   * True when the display-fact builder itself dropped records/characters to
   * stay bounded. Independent of the canonical content budget.
   */
  displayTruncated: boolean
}

// [maint-note:4b98ab01f6a5]
// 数据契约导航：ApplyPatchDependencies；字段含义以消费端校验为准。
// 成员 onChangeApplied（可省略）：(path: string) => void | Promise<void>。
// [/maint-note:4b98ab01f6a5]
export interface ApplyPatchDependencies {
  /** Test-only transaction seam; deliberately private to apply_patch. */
  onChangeApplied?: (path: string) => void | Promise<void>
}
