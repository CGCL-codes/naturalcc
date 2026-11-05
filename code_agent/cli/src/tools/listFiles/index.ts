import type { Tool } from '../types.js'
import { createRipgrepRunner, type RipgrepRunner } from '../backends/ripgrep/index.js'
import { executeListFiles } from './execute.js'
import { listFilesSchema, parseListFilesInput } from './schema.js'

export function createListFilesTool(runner: RipgrepRunner = createRipgrepRunner()): Tool {
  return {
    ...listFilesSchema,
    parse: parseListFilesInput,
    execute: (input, context) => executeListFiles(input, context, runner),
  }
}

export const listFilesTool: Tool = createListFilesTool()

export { parseListFilesInput } from './schema.js'
