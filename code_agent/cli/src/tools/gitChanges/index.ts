import type { Tool } from '../types.js'
import { createGitChangesBackend, type GitChangesBackend } from '../backends/git/index.js'
import { executeGitChanges } from './execute.js'
import { gitChangesSchema, parseGitChangesInput } from './schema.js'

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
