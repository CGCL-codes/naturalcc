import type { ToolResultPresentation } from './index.js'

// [maint-note:1879cb43e670]
// 数据契约导航：ToolPresenter；字段含义以消费端校验为准。
// 类型组合：( output: string, options: { maxChars?: number maxLines?: number input?: unknown reserveGlobalNotice?: boolean display?: import('../../tools/types.js').ToolDisplayFacts contentWidth?: number }, ) => ToolResultPresentation。
// [/maint-note:1879cb43e670]
export type ToolPresenter = (
  output: string,
  options: {
    maxChars?: number
    maxLines?: number
    input?: unknown
    reserveGlobalNotice?: boolean
    display?: import('../../tools/types.js').ToolDisplayFacts
    contentWidth?: number
  },
) => ToolResultPresentation

/**
 * presenter dispatch boundary. Tools never import React/ANSI; the UI
 * resolves a presenter by tool name at the application assembly edge.
 * Unknown tools fall back to the bounded literal presenter; a presenter
 * that throws degrades to the same fallback instead of failing the turn —
 * a UI projection bug must never convert a successful tool result into an
 * error. Registering the same name twice fails loudly at assembly time.
 */
export class ToolPresenterRegistry {
  private readonly presenters = new Map<string, ToolPresenter>()

  // [maint-note:7514f02f27f1]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 fallback：(toolName: string, output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation。
  // [/maint-note:7514f02f27f1]
  constructor(
    private readonly fallback: (toolName: string, output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation,
  ) {}

  // [maint-note:422bded78ced]
  // 维护导航：register；下列为本函数源码中的接口与路径索引。
  // 输入 name：string。
  // 输入 presenter：ToolPresenter。
  // 返回契约：void。
  // 分支条件：this.presenters.has(name)。
  // 显式抛出：new Error(`Tool presenter already registered: ${name}`)。
  // 直接调用：this.presenters.has。
  // 直接调用：this.presenters.set。
  // [/maint-note:422bded78ced]
  register(name: string, presenter: ToolPresenter): void {
    if (this.presenters.has(name)) {
      throw new Error(`Tool presenter already registered: ${name}`)
    }
    this.presenters.set(name, presenter)
  }

  // [maint-note:c6d3b39b265c]
  // 维护导航：has；下列为本函数源码中的接口与路径索引。
  // 输入 name：string。
  // 返回契约：boolean。
  // 返回路径：this.presenters.has(name)。
  // 直接调用：this.presenters.has。
  // [/maint-note:c6d3b39b265c]
  has(name: string): boolean {
    return this.presenters.has(name)
  }

  // [maint-note:c4d7b952e907]
  // 维护导航：names；下列为本函数源码中的接口与路径索引。
  // 返回契约：readonly string[]。
  // 返回路径：[...this.presenters.keys()]。
  // 直接调用：this.presenters.keys。
  // [/maint-note:c4d7b952e907]
  names(): readonly string[] {
    return [...this.presenters.keys()]
  }

  // [maint-note:7750b7c8d5b6]
  // 维护导航：present；下列为本函数源码中的接口与路径索引。
  // 输入 toolName：string。
  // 输入 output：string。
  // 输入 options：Parameters<ToolPresenter>[1]；缺省表达式 {}。
  // 返回契约：ToolResultPresentation。
  // 分支条件：presenter !== undefined。
  // 返回路径：presenter(output, options)。
  // 返回路径：this.safeFallback(toolName, output, options)。
  // 直接调用：this.presenters.get。
  // 直接调用：presenter。
  // 直接调用：this.safeFallback。
  // [/maint-note:7750b7c8d5b6]
  present(
    toolName: string,
    output: string,
    options: Parameters<ToolPresenter>[1] = {},
  ): ToolResultPresentation {
    const presenter = this.presenters.get(toolName)
    if (presenter !== undefined) {
      try {
        return presenter(output, options)
      } catch {
        // Degrade, never fail the turn: the canonical ToolResult is a fact
        // the model already received; only this projection is disposable.
        return this.safeFallback(toolName, output, options)
      }
    }
    return this.safeFallback(toolName, output, options)
  }

  // [maint-note:d82615b02539]
  // 维护导航：safeFallback；下列为本函数源码中的接口与路径索引。
  // 输入 toolName：string。
  // 输入 output：string。
  // 输入 options：Parameters<ToolPresenter>[1]。
  // 返回契约：ToolResultPresentation。
  // 返回路径：this.fallback(toolName, output, options)。
  // 返回路径：{ toolName, summary: 'completed', blocks: [{ kind: 'summary', text: 'completed' }], plainText: 'completed', truncated: false, sourceTruncated: false, viewCollapsed: false, canExpand: false, detail: { totalRecords: 1, omittedRecords: 0, sour。
  // 直接调用：this.fallback。
  // [/maint-note:d82615b02539]
  private safeFallback(
    toolName: string,
    output: string,
    options: Parameters<ToolPresenter>[1],
  ): ToolResultPresentation {
    try {
      return this.fallback(toolName, output, options)
    } catch {
      // The literal fallback itself must never throw; if it somehow does,
      // return the most degenerate bounded presentation instead of crashing.
      return {
        toolName,
        summary: 'completed',
        blocks: [{ kind: 'summary', text: 'completed' }],
        plainText: 'completed',
        truncated: false,
        sourceTruncated: false,
        viewCollapsed: false,
        canExpand: false,
        detail: {
          totalRecords: 1,
          omittedRecords: 0,
          sourceTruncated: false,
          viewCollapsed: false,
          canExpand: false,
        },
      }
    }
  }
}
