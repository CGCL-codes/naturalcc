import { Command } from 'commander'

import { AgentError, createAbortError, errorMessage, isAbortError } from './agent/errors.js'
import {
  createCodingAgent,
  type CodingAgentApplication,
  type CreateCodingAgentOptions,
} from './app/createCodingAgent.js'
import { OneShotEventRenderer, type EventWriter } from './ui/oneShotEventRenderer.js'
import { VERSION } from './version.js'
import type { ApprovalMode } from './tools/approval.js'
import { InteractiveApprovalProvider } from './tools/approval.js'
import type { SessionStatusMetadata } from './ui/components/SessionStatusSlot.js'
import { createTerminalResizeOwner } from './ui/terminal/terminalResizeOwner.js'

// [maint-note:71e050010901]
// 数据契约导航：CliOptions；字段含义以消费端校验为准。
// 成员 model（可省略）：string。
// 成员 apiKey（可省略）：string。
// 成员 baseUrl（可省略）：string。
// 成员 projectDir（可省略）：string。
// 成员 maxTurns（可省略）：number。
// 成员 toolResults（可省略）：boolean。
// 成员 approval（可省略）：ApprovalMode。
// 成员 yes（可省略）：boolean。
// [/maint-note:71e050010901]
export interface CliOptions {
  model?: string
  apiKey?: string
  baseUrl?: string
  projectDir?: string
  maxTurns?: number
  toolResults?: boolean
  approval?: ApprovalMode
  yes?: boolean
}

// [maint-note:ee3410faf1c0]
// 数据契约导航：CliIO；字段含义以消费端校验为准。
// 成员 stdout：EventWriter。
// 成员 stderr：EventWriter。
// [/maint-note:ee3410faf1c0]
interface CliIO {
  stdout: EventWriter
  stderr: EventWriter
}

// [maint-note:30c440c36d89]
// 数据契约导航：CodingAgentFactory；字段含义以消费端校验为准。
// 类型组合：( options: CreateCodingAgentOptions, ) => Promise<CodingAgentApplication>。
// [/maint-note:30c440c36d89]
export type CodingAgentFactory = (
  options: CreateCodingAgentOptions,
) => Promise<CodingAgentApplication>

export const interactiveRenderOptions = {
  incrementalRendering: false,
  exitOnCtrlC: false,
} as const

// [maint-note:18b97335a9d7]
// 维护导航：runOneShot；下列为本函数源码中的接口与路径索引。
// 输入 prompt：string。
// 输入 options：CreateCodingAgentOptions；缺省表达式 {}。
// 输入 io：CliIO；缺省表达式 { stdout: process.stdout, stderr: process.stderr }。
// 输入 factory：CodingAgentFactory；缺省表达式 createCodingAgent。
// 返回契约：Promise<number>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：controller.signal.aborted。
// 分支条件：!session。
// 分支条件：isAbortError(error, controller.signal)。
// 遍历边界：const event ← session.runTurn({ prompt, signal: controller.signal, })。
// 返回路径：renderer.exitCode。
// 返回路径：130。
// 返回路径：code。
// 显式抛出：createAbortError()。
// 显式抛出：new Error('The coding agent did not create a session')。
// 写入目标：application。
// 直接调用：process.once。
// 直接调用：factory。
// 直接调用：createAbortError。
// 直接调用：application.sessionManager.current。
// 直接调用：session.runTurn。
// 直接调用：renderer.handle。
// 直接调用：isAbortError。
// 直接调用：io.stderr.write。
// 直接调用：errorMessage。
// 直接调用：process.removeListener。
// 直接调用：application?.dispose。
// [/maint-note:18b97335a9d7]
export async function runOneShot(
  prompt: string,
  options: CreateCodingAgentOptions = {},
  io: CliIO = { stdout: process.stdout, stderr: process.stderr },
  factory: CodingAgentFactory = createCodingAgent,
): Promise<number> {
  const controller = new AbortController()
  // [maint-note:85f04e21f7fb]
  // 维护导航：onInterrupt；下列为本函数源码中的接口与路径索引。
  // 直接调用：controller.abort。
  // [/maint-note:85f04e21f7fb]
  const onInterrupt = () => controller.abort()
  process.once('SIGINT', onInterrupt)

  let application: CodingAgentApplication | undefined
  try {
    application = await factory({ ...options, signal: controller.signal })
    if (controller.signal.aborted) throw createAbortError()
    const session = application.sessionManager.current()
    if (!session) throw new Error('The coding agent did not create a session')

    const approvalPresentationOwner = options.approvalProvider?.presentationOwner
      ?? (application.config.codingAgent.approvalMode === 'prompt' ? 'provider' : 'renderer')
    const renderer = new OneShotEventRenderer(io.stdout, io.stderr, {
      approvalPresentationOwner,
    })
    for await (const event of session.runTurn({
      prompt,
      signal: controller.signal,
    })) {
      renderer.handle(event)
    }
    return renderer.exitCode
  } catch (error) {
    if (isAbortError(error, controller.signal)) {
      io.stderr.write('Agent turn interrupted.\n')
      return 130
    }
    const code = error instanceof AgentError && error.code === 'config_error' ? 2 : 1
    io.stderr.write(`${errorMessage(error)}\n`)
    return code
  } finally {
    process.removeListener('SIGINT', onInterrupt)
    await application?.dispose()
  }
}

// [maint-note:95b6181e84c5]
// 维护导航：runCli；下列为本函数源码中的接口与路径索引。
// 输入 args：string[]。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 直接调用：cli .name('naturalcc') .description('NaturalCC coding agent') .version(VERSION, '-v, --version', '显示版本号') .option('--model <name>', '模型名称') .option('--api-key <key>', 'OpenAI API Key') .option('--base-url <url>', 'OpenAI-compatible endpoint。
// 直接调用：cli .name('naturalcc') .description('NaturalCC coding agent') .version(VERSION, '-v, --version', '显示版本号') .option('--model <name>', '模型名称') .option('--api-key <key>', 'OpenAI API Key') .option。
// 直接调用：cli .name('naturalcc') .description('NaturalCC coding agent') .version(VERSION, '-v, --version', '显示版本号') .option('--model <name>', '模型名称') .option。
// 直接调用：cli .name('naturalcc') .description('NaturalCC coding agent') .version(VERSION, '-v, --version', '显示版本号') .option。
// 直接调用：cli .name('naturalcc') .description('NaturalCC coding agent') .version。
// 直接调用：cli .name('naturalcc') .description。
// 直接调用：cli .name。
// 直接调用：cli.parseAsync。
// [/maint-note:95b6181e84c5]
export async function runCli(args: string[]): Promise<void> {
  const cli = new Command()

  cli
    .name('naturalcc')
    .description('NaturalCC coding agent')
    .version(VERSION, '-v, --version', '显示版本号')
    .option('--model <name>', '模型名称')
    .option('--api-key <key>', 'OpenAI API Key')
    .option('--base-url <url>', 'OpenAI-compatible endpoint')
    .option('--project-dir <dir>', '项目目录')
    .option('--max-turns <number>', '单回合模型请求硬上限（默认32，范围1..256）', (value) => Number(value))
    .option('--no-tool-results', '隐藏正常工具输出')
    .option('--approval <mode>', '审批模式：prompt、deny 或 allow', parseApprovalMode)
    .option('--yes', '允许需要审批的工具调用')
    .argument('[prompt...]')
    .action(async (promptParts: string[], options: CliOptions) => {
      const prompt = promptParts.join(' ').trim()
      const config = toAgentOptions(options)

      if (!prompt) {
        if (process.stdin.isTTY !== true) {
          cli.outputHelp()
          process.exitCode = 2
          return
        }
        await runInteractive(config)
        return
      }

      process.exitCode = await runOneShot(prompt, config)
    })

  await cli.parseAsync(['node', 'naturalcc', ...args])
}

// [maint-note:bddf9383726c]
// 维护导航：runInteractive；下列为本函数源码中的接口与路径索引。
// 输入 options：CreateCodingAgentOptions。
// 输入 factory：CodingAgentFactory；缺省表达式 createCodingAgent。
// 输入 onKeyboardReady：() => void（可省略）。
// 输入 onInputChange：(value: string) => void（可省略）。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：controller.signal.aborted。
// 分支条件：isAbortError(error, controller.signal) || controller.signal.aborted。
// 显式抛出：createAbortError()。
// 写入目标：application。
// 写入目标：interactiveInstance。
// 写入目标：process.exitCode。
// 直接调用：createTerminalResizeOwner。
// 直接调用：process.on。
// 直接调用：factory。
// 直接调用：createAbortError。
// 直接调用：Promise.all。
// 直接调用：import。
// 直接调用：render。
// 直接调用：createElement。
// 直接调用：resizeOwner.attachRendererFlush。
// 直接调用：instance.waitUntilExit。
// 直接调用：isAbortError。
// 直接调用：process.stderr.write。
// 直接调用：errorMessage。
// 直接调用：process.removeListener。
// 直接调用：interactiveInstance?.unmount。
// 直接调用：resizeOwner.disposeResizeOwner。
// 直接调用：application?.dispose。
// [/maint-note:bddf9383726c]
export async function runInteractive(
  options: CreateCodingAgentOptions,
  factory: CodingAgentFactory = createCodingAgent,
  onKeyboardReady?: () => void,
  onInputChange?: (value: string) => void,
): Promise<void> {
  let application: CodingAgentApplication | undefined
  let interactiveInstance: {
    unmount: () => void
    waitUntilExit: () => Promise<unknown>
    waitUntilRenderFlush: () => Promise<void>
  } | undefined
  const resizeOwner = createTerminalResizeOwner(process.stdout)
  const controller = new AbortController()
  let resolveKeyboardReady: (() => void) | undefined
  let keyboardReady = false
  const keyboardReadyPromise = new Promise<void>((resolve) => {
    resolveKeyboardReady = resolve
  })
  // [maint-note:818a0c6438b8]
  // 维护导航：onInterrupt；下列为本函数源码中的接口与路径索引。
  // 直接调用：controller.abort。
  // 直接调用：resolveKeyboardReady。
  // [/maint-note:818a0c6438b8]
  const onInterrupt = () => {
    controller.abort()
    // Do not leave the launcher waiting if SIGINT arrives while Ink is being
    // imported/rendered. The listener remains installed until the ready
    // callback or the catch/finally path completes the bounded handoff.
    resolveKeyboardReady?.()
  }
  // Keep one persistent listener until the raw-mode handoff completes. A
  // once-listener can be removed before Bun finishes dispatching SIGINT and
  // re-enable the default process termination on the first-frame race.
  process.on('SIGINT', onInterrupt)
  const approvalProvider = options.approvalMode === 'allow' || options.approvalMode === 'deny'
    ? undefined
    : new InteractiveApprovalProvider()
  try {
    application = await factory({
      ...options,
      approvalProvider,
      signal: controller.signal,
    })
    if (controller.signal.aborted) throw createAbortError()
    const [{ render }, { createElement }, { Repl }] = await Promise.all([
      import('ink'),
      import('react'),
      import('./ui/repl.js'),
    ])
    // [maint-note:a28cd79dfb7f]
    // 维护导航：markKeyboardReady；下列为本函数源码中的接口与路径索引。
    // 分支条件：keyboardReady。
    // 返回路径：无返回值。
    // 写入目标：keyboardReady。
    // 直接调用：process.removeListener。
    // 直接调用：onKeyboardReady。
    // 直接调用：resolveKeyboardReady。
    // [/maint-note:a28cd79dfb7f]
    const markKeyboardReady = () => {
      if (keyboardReady) return
      keyboardReady = true
      process.removeListener('SIGINT', onInterrupt)
      onKeyboardReady?.()
      resolveKeyboardReady?.()
    }
    const statusMetadata: SessionStatusMetadata = {
      model: application.config.model.model,
      workspace: application.config.codingAgent.projectDir,
    }
    const instance = render(createElement(Repl, {
      sessionManager: application.sessionManager,
      approvalProvider,
      statusMetadata,
      onKeyboardReady: markKeyboardReady,
      onInputChange,
    }), {
      ...interactiveRenderOptions,
      // Ink's public type still requires a NodeJS.WriteStream although it
      // only consumes the narrow stdout facade implemented by the owner.
      stdout: resizeOwner as unknown as NodeJS.WriteStream,
      onRender: resizeOwner.beginRendererCommit,
    })
    interactiveInstance = instance
    resizeOwner.attachRendererFlush(instance.waitUntilRenderFlush)
    await keyboardReadyPromise
    if (controller.signal.aborted) throw createAbortError()
    await instance.waitUntilExit()
  } catch (error) {
    if (isAbortError(error, controller.signal) || controller.signal.aborted) {
      process.stderr.write('Agent turn interrupted.\n')
      process.exitCode = 130
    } else {
      process.stderr.write(`${errorMessage(error)}\n`)
      process.exitCode = error instanceof AgentError && error.code === 'config_error' ? 2 : 1
    }
  } finally {
    process.removeListener('SIGINT', onInterrupt)
    interactiveInstance?.unmount()
    resizeOwner.disposeResizeOwner()
    await application?.dispose()
  }
}

// [maint-note:aba64e20581f]
// 维护导航：toAgentOptions；下列为本函数源码中的接口与路径索引。
// 输入 options：CliOptions。
// 返回契约：CreateCodingAgentOptions。
// 返回路径：{ model: options.model, apiKey: options.apiKey, baseURL: options.baseUrl, projectDir: options.projectDir, maxTurns: options.maxTurns, showToolResults: options.toolResults !== false, approvalMode, }。
// [/maint-note:aba64e20581f]
export function toAgentOptions(options: CliOptions): CreateCodingAgentOptions {
  const approvalMode = options.yes ? 'allow' : options.approval
  return {
    model: options.model,
    apiKey: options.apiKey,
    baseURL: options.baseUrl,
    projectDir: options.projectDir,
    maxTurns: options.maxTurns,
    showToolResults: options.toolResults !== false,
    approvalMode,
  }
}

// [maint-note:72f842724e58]
// 维护导航：parseApprovalMode；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：ApprovalMode。
// 分支条件：value === 'prompt' || value === 'deny' || value === 'allow'。
// 返回路径：value。
// 显式抛出：new Error('approval mode must be prompt, deny, or allow')。
// [/maint-note:72f842724e58]
export function parseApprovalMode(value: string): ApprovalMode {
  if (value === 'prompt' || value === 'deny' || value === 'allow') return value
  throw new Error('approval mode must be prompt, deny, or allow')
}
