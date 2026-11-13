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

import { applyPatchTool } from './applyPatch/index.js'
import { createListFilesTool, listFilesTool } from './listFiles/index.js'
import { readFileTool } from './readFile/index.js'
import { createSearchTextTool, searchTextTool } from './searchText/index.js'
import type { RipgrepRunner } from './backends/ripgrep/index.js'
import type { GitChangesBackend } from './backends/git/index.js'
import { createGitChangesTool } from './gitChanges/index.js'
import { validateApprovalInput, validateApprovalSchema } from './approval.js'
import { updateTaskPlanTool } from './updateTaskPlan/index.js'

// [maint-note:41af189a85aa]
// 维护导航：createBuiltinTools；下列为本函数源码中的接口与路径索引。
// 输入 runner：RipgrepRunner（可省略）。
// 输入 gitChangesBackend：GitChangesBackend（可省略）。
// 返回契约：Tool[]。
// 返回路径：[ runner ? createListFilesTool(runner) : listFilesTool, readFileTool, runner ? createSearchTextTool(runner) : searchTextTool, applyPatchTool, bashTool, updateTaskPlanTool, ...(gitChangesBackend ? [createGitChangesTool(gitChangesBackend)] : 。
// 直接调用：createListFilesTool。
// 直接调用：createSearchTextTool。
// 直接调用：createGitChangesTool。
// [/maint-note:41af189a85aa]
export function createBuiltinTools(
  runner?: RipgrepRunner,
  gitChangesBackend?: GitChangesBackend,
): Tool[] {
  return [
    runner ? createListFilesTool(runner) : listFilesTool,
    readFileTool,
    runner ? createSearchTextTool(runner) : searchTextTool,
    applyPatchTool,
    bashTool,
    updateTaskPlanTool,
    ...(gitChangesBackend ? [createGitChangesTool(gitChangesBackend)] : []),
  ]
}

export const builtinTools: Tool[] = createBuiltinTools()

export class DefaultToolRegistry implements ToolRegistry {
  private readonly tools = new Map<string, Tool>()

  // [maint-note:7cd38599eb1b]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 tools：Tool[]；缺省表达式 builtinTools。
  // 分支条件：this.tools.has(tool.name)。
  // 遍历边界：const tool ← tools。
  // 显式抛出：new Error(`Duplicate tool name: ${tool.name}`)。
  // 直接调用：this.tools.has。
  // 直接调用：validateApprovalSchema。
  // 直接调用：this.tools.set。
  // [/maint-note:7cd38599eb1b]
  constructor(tools: Tool[] = builtinTools) {
    for (const tool of tools) {
      if (this.tools.has(tool.name)) {
        throw new Error(`Duplicate tool name: ${tool.name}`)
      }
      validateApprovalSchema(tool)
      this.tools.set(tool.name, tool)
    }
  }

  // [maint-note:98822b609c3d]
  // 维护导航：schemas；下列为本函数源码中的接口与路径索引。
  // 返回契约：ToolSchema[]。
  // 返回路径：Array.from(this.tools.values(), toToolSchema)。
  // 直接调用：Array.from。
  // 直接调用：this.tools.values。
  // [/maint-note:98822b609c3d]
  schemas(): ToolSchema[] {
    return Array.from(this.tools.values(), toToolSchema)
  }

  // [maint-note:9fbd19c64eba]
  // 维护导航：validate；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 返回契约：void。
  // 分支条件：!tool。
  // 显式抛出：new ToolInputError(`Unknown tool: ${call.name}`)。
  // 直接调用：this.tools.get。
  // 直接调用：validateApprovalInput。
  // 直接调用：tool.parse。
  // [/maint-note:9fbd19c64eba]
  validate(call: ToolCall): void {
    const tool = this.tools.get(call.name)
    if (!tool) throw new ToolInputError(`Unknown tool: ${call.name}`)
    validateApprovalInput(tool, call.arguments)
    tool.parse?.(call.arguments)
  }

  // [maint-note:a1cb500993d2]
  // 维护导航：execute；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 输入 context：ToolExecutionContext。
  // 返回契约：Promise<ToolExecutionResult>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!tool。
  // 返回路径：tool.execute(parsed as Record<string, unknown>, context)。
  // 显式抛出：new Error(`Unknown tool: ${call.name}`)。
  // 直接调用：this.tools.get。
  // 直接调用：tool.parse。
  // 直接调用：tool.execute。
  // [/maint-note:a1cb500993d2]
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

// [maint-note:0804a80bc286]
// 维护导航：createToolRegistry；下列为本函数源码中的接口与路径索引。
// 输入 tools：Tool[]（可省略）。
// 输入 runner：RipgrepRunner（可省略）。
// 输入 gitChangesBackend：GitChangesBackend（可省略）。
// 返回契约：ToolRegistry。
// 返回路径：new DefaultToolRegistry(tools ?? createBuiltinTools(runner, gitChangesBackend))。
// 直接调用：createBuiltinTools。
// [/maint-note:0804a80bc286]
export function createToolRegistry(
  tools?: Tool[],
  runner?: RipgrepRunner,
  gitChangesBackend?: GitChangesBackend,
): ToolRegistry {
  return new DefaultToolRegistry(tools ?? createBuiltinTools(runner, gitChangesBackend))
}

// [maint-note:3cf040371a1d]
// 维护导航：toToolSchema；下列为本函数源码中的接口与路径索引。
// 输入 tool：Tool。
// 返回契约：ToolSchema。
// 返回路径：{ name: tool.name, description: tool.description, parameters: tool.parameters, readOnly: tool.readOnly, requiresApproval: tool.requiresApproval, repeatKind: tool.repeatKind, taskManagement: tool.taskManagement, }。
// [/maint-note:3cf040371a1d]
function toToolSchema(tool: Tool): ToolSchema {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    readOnly: tool.readOnly,
    requiresApproval: tool.requiresApproval,
    repeatKind: tool.repeatKind,
    taskManagement: tool.taskManagement,
  }
}
