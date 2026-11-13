import type {Ctx} from '../types/ctx.js'
import {help_info} from '../utils/help-info.js'

// [maint-note:6a00938efdb2]
// 数据契约导航：SlashCommandDef；字段含义以消费端校验为准。
// 成员 name：string。
// 成员 description：string。
// [/maint-note:6a00938efdb2]
export interface SlashCommandDef {
  name: string
  description: string
}

// [maint-note:ffbdff42b091]
// 数据契约导航：SlashCommand；字段含义以消费端校验为准。
// 成员 run：(ctx: Ctx, rawArguments: string) => void | Promise<void>。
// [/maint-note:ffbdff42b091]
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

// [maint-note:87f8df5ea151]
// 维护导航：dispatch；下列为本函数源码中的接口与路径索引。
// 输入 input：string。
// 输入 ctx：Ctx。
// 返回契约：Promise<boolean>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：first.startsWith('/')。
// 分支条件：handler。
// 返回路径：true。
// 返回路径：ctx.execute(trimmed)。
// 直接调用：input.trim。
// 直接调用：trimmed.search。
// 直接调用：trimmed.slice。
// 直接调用：trimmed.slice(separator).trim。
// 直接调用：first.startsWith。
// 直接调用：handler。
// 直接调用：ctx.execute。
// [/maint-note:87f8df5ea151]
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
