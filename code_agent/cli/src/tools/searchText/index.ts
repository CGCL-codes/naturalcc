import type { Tool } from '../types.js'
import { createRipgrepRunner, type RipgrepRunner } from '../backends/ripgrep/index.js'
import { executeSearchText } from './execute.js'
import { parseSearchTextInput, searchTextSchema } from './schema.js'

export function createSearchTextTool(runner: RipgrepRunner = createRipgrepRunner()): Tool {
  return {
    ...searchTextSchema,
    parse: parseSearchTextInput,
    execute: (input, context) => executeSearchText(input, context, runner),
  }
}

export const searchTextTool: Tool = createSearchTextTool()

export {
  MAX_SEARCH_QUERY_UTF8_BYTES,
  parseSearchTextInput,
} from './schema.js'
export type { SearchTextInput } from './schema.js'
