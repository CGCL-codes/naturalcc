import type { ToolResultPresentation } from './index.js'

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

  constructor(
    private readonly fallback: (toolName: string, output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation,
  ) {}

  register(name: string, presenter: ToolPresenter): void {
    if (this.presenters.has(name)) {
      throw new Error(`Tool presenter already registered: ${name}`)
    }
    this.presenters.set(name, presenter)
  }

  has(name: string): boolean {
    return this.presenters.has(name)
  }

  names(): readonly string[] {
    return [...this.presenters.keys()]
  }

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
