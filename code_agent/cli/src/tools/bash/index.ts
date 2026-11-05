import type { Tool } from '../types.js'
import { executeBashTool, parseBashInput } from './execute.js'
import { bashSchema } from './schema.js'

export const bashTool: Tool = {
  ...bashSchema,
  parse: parseBashInput,
  execute: executeBashTool,
}

export { executeBash } from './execute.js'
