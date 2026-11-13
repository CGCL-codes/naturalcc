import type { Tool } from '../types.js'
import { createGitChangesBackend, type GitChangesBackend } from '../backends/git/index.js'
import { executeGitChanges } from './execute.js'
import { gitChangesSchema, parseGitChangesInput } from './schema.js'

// [maint-note:1b9a0a091bc2]
// 维护导航：createGitChangesTool；下列为本函数源码中的接口与路径索引。
// 输入 backend：GitChangesBackend；缺省表达式 createGitChangesBackend()。
// 返回契约：Tool。
// 返回路径：{ ...gitChangesSchema, parse: parseGitChangesInput, execute: (input, context) => executeGitChanges(input, context, backend), }。
// [/maint-note:1b9a0a091bc2]
export function createGitChangesTool(backend: GitChangesBackend = createGitChangesBackend()): Tool {
  return {
    ...gitChangesSchema,
    parse: parseGitChangesInput,
    execute: (input, context) => executeGitChanges(input, context, backend),
  }
}

export {
  MAX_GIT_CHANGES_CURSOR_BYTES,
  MAX_GIT_CHANGES_PATH_BYTES,
  parseGitChangesInput,
} from './schema.js'
export type { GitChangesInput } from './schema.js'
export type {
  GitChangeAction,
  GitChangeLayer,
  GitChangeMode,
  GitChangeScope,
  GitChangesCoverage,
  GitChangesObservation,
  GitFileRecord,
  GitPatchHunk,
  GitPatchLayer,
  GitPatchLine,
} from './types.js'
