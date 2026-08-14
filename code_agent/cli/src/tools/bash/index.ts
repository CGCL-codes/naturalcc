import type { Tool } from '../types.js'
import { executeBashTool } from './execute.js'
import { bashSchema } from './schema.js'

export const bashTool: Tool = {
  ...bashSchema,
  execute: executeBashTool,
}

export { executeBash } from './execute.js'
