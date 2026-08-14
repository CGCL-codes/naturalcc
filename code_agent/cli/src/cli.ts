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

interface CliIO {
  stdout: EventWriter
  stderr: EventWriter
}

export type CodingAgentFactory = (
  options: CreateCodingAgentOptions,
) => Promise<CodingAgentApplication>

export async function runOneShot(
  prompt: string,
  options: CreateCodingAgentOptions = {},
  io: CliIO = { stdout: process.stdout, stderr: process.stderr },
  factory: CodingAgentFactory = createCodingAgent,
): Promise<number> {
  const controller = new AbortController()
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
    .option('--max-turns <number>', '单回合最大循环次数', (value) => Number(value))
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

export async function runInteractive(
  options: CreateCodingAgentOptions,
  factory: CodingAgentFactory = createCodingAgent,
): Promise<void> {
  let application: CodingAgentApplication | undefined
  const controller = new AbortController()
  const onInterrupt = () => controller.abort()
  process.once('SIGINT', onInterrupt)
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
    // After initialization, Ink owns interactive SIGINT behavior.
    process.removeListener('SIGINT', onInterrupt)
    const [{ render }, { createElement }, { Repl }] = await Promise.all([
      import('ink'),
      import('react'),
      import('./ui/repl.js'),
    ])
    const instance = render(createElement(Repl, {
      sessionManager: application.sessionManager,
      approvalProvider,
    }), { incrementalRendering: true })
    await instance.waitUntilExit()
  } catch (error) {
    if (isAbortError(error, controller.signal)) {
      process.stderr.write('Agent turn interrupted.\n')
      process.exitCode = 130
    } else {
      process.stderr.write(`${errorMessage(error)}\n`)
      process.exitCode = error instanceof AgentError && error.code === 'config_error' ? 2 : 1
    }
  } finally {
    process.removeListener('SIGINT', onInterrupt)
    await application?.dispose()
  }
}

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

export function parseApprovalMode(value: string): ApprovalMode {
  if (value === 'prompt' || value === 'deny' || value === 'allow') return value
  throw new Error('approval mode must be prompt, deny, or allow')
}
