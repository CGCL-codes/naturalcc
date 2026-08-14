import type {Ctx} from '../types/ctx.js'
import {help_info} from '../utils/help-info.js'

export interface SlashCommandDef {
  name: string
  description: string
}

interface SlashCommand extends SlashCommandDef {
  run: (ctx: Ctx, rawArguments: string) => void | Promise<void>
}

const slashCommandEntries: SlashCommand[] = [
  { name: '/help', description: '显示帮助信息', run: (ctx) => ctx.addMsg('assistant', help_info()) },
  { name: '/exit', description: '退出 REPL', run: (ctx) => { process.stdout.write('\n'); ctx.exit() } },
  {
    name: '/clear',
    description: '清除 UI 与 agent 对话历史',
    run: async (ctx) => { await ctx.clearMessages() },
  },
]

export const slashCommandDefs: SlashCommandDef[] = slashCommandEntries.map(
  ({ name, description }) => ({ name, description }),
)

const slashCommands: Record<string, SlashCommand['run']> = Object.fromEntries(
  slashCommandEntries.map((command) => [command.name, command.run]),
)

export async function dispatch(input: string, ctx: Ctx): Promise<boolean> {
  const trimmed = input.trim()
  const separator = trimmed.search(/\s/)

  const first = separator === -1
    ? trimmed
    : trimmed.slice(0,separator)

  const rawArguments = separator === -1
    ? ''
    : trimmed.slice(separator).trim()

  // slash 命令
  if (first.startsWith('/')) {
    const handler = slashCommands[first]
    if (handler) {
      await handler(ctx, rawArguments)
      return true
    }
  }

  // 不是命令 → instruction 执行
  return ctx.execute(trimmed)
}
