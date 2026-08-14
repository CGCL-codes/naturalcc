import { bashTool } from './bash/index.js'
import {
  ToolInputError,
} from './types.js'
import type {
  Tool,
  ToolCall,
  ToolExecutionContext,
  ToolExecutionResult,
  ToolRegistry,
  ToolSchema,
} from './types.js'

import { applyPatchTool } from './applyPatch.js'
import { listFilesTool } from './listFiles.js'
import { readFileTool } from './readFile.js'
import { searchTextTool } from './searchText.js'
import { createListFilesTool } from './listFiles.js'
import { createSearchTextTool } from './searchText.js'
import type { RipgrepRunner } from './ripgrep/types.js'
import { validateApprovalInput, validateApprovalSchema } from './approval.js'

export function createBuiltinTools(runner?: RipgrepRunner): Tool[] {
  return [
  runner ? createListFilesTool(runner) : listFilesTool,
  readFileTool,
  runner ? createSearchTextTool(runner) : searchTextTool,
  applyPatchTool,
  bashTool,
  ]
}

export const builtinTools: Tool[] = createBuiltinTools()

export class DefaultToolRegistry implements ToolRegistry {
  private readonly tools = new Map<string, Tool>()

  constructor(tools: Tool[] = builtinTools) {
    for (const tool of tools) {
      if (this.tools.has(tool.name)) {
        throw new Error(`Duplicate tool name: ${tool.name}`)
      }
      validateApprovalSchema(tool)
      this.tools.set(tool.name, tool)
    }
  }

  schemas(): ToolSchema[] {
    return Array.from(this.tools.values(), toToolSchema)
  }

  validate(call: ToolCall): void {
    const tool = this.tools.get(call.name)
    if (!tool) throw new ToolInputError(`Unknown tool: ${call.name}`)
    validateApprovalInput(tool, call.arguments)
    tool.parse?.(call.arguments)
  }

  async execute(
    call: ToolCall,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult> {
    const tool = this.tools.get(call.name)
    if (!tool) {
      throw new Error(`Unknown tool: ${call.name}`)
    }
    const parsed = tool.parse?.(call.arguments) ?? call.arguments
    return tool.execute(parsed as Record<string, unknown>, context)
  }
}

export function createToolRegistry(
  tools?: Tool[],
  runner?: RipgrepRunner,
): ToolRegistry {
  return new DefaultToolRegistry(tools ?? createBuiltinTools(runner))
}

function toToolSchema(tool: Tool): ToolSchema {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    readOnly: tool.readOnly,
    requiresApproval: tool.requiresApproval,
  }
}
