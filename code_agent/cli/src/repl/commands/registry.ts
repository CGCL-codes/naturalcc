import type {Ctx} from '../../types/ctx.js'
import {help_info} from '../help.js'

const slashCommands: Record<string, (ctx: Ctx) => void> = {
  '/help':     (ctx) => ctx.addMsg('assistant', help_info()),
  '/exit':     (ctx) => { process.stdout.write('\n'); ctx.exit() },
  '/clear':    (ctx) => { ctx.clearMessages() },
  '/settings': (ctx) => { ctx.toggleSettings() },
  '/reset':    (ctx) => { ctx.resetSettings() },
}

interface FlagDef {
  aliases: string[]                        // ['-t', '--completionType']
  arity: number                            // 期望参数个数
  validate?: (value: string[]) => string | null
  apply: (ctx: Ctx, value: string[]) => void
}

const validTypes = new Set(['member', 'variable', 'function', 'function_body', 'type'])

const flagCommands: FlagDef[] = [
  { aliases: ['-f', '--file'],         arity: 1, apply: (ctx, v) => ctx.setFiles(v) },
  { aliases: ['-m', '--model'],        arity: 1, apply: (ctx, [v]) => ctx.setModel(v) },
  { aliases: ['-k', '--apiKey'],       arity: 1, apply: (ctx, [v]) => ctx.setApiKey(v) },
  { aliases: ['-d', '--projectDir'],   arity: 1, apply: (ctx, [v]) => ctx.setProjectDir(v) },
  { aliases: ['-s', '--symbol'],       arity: 1, apply: (ctx, [v]) => ctx.setSymbol(v) },
  {
    aliases: ['-t', '--completionType'],
    arity: 1,
    validate: ([v]) => validTypes.has(v) ? null : `invalid type: ${v}`,
    apply: (ctx, [v]) => ctx.setCompletionType(v),
  },
  { aliases: ['--prefix'], arity: 1, apply: (ctx, [v]) => ctx.setPrefix(v) },
  { aliases: ['--preview'], arity: 0, apply: (ctx) => ctx.togglePreview() },
  { aliases: ['--run'], arity: 0, apply: (ctx) => ctx.rerun() },
]

export function dispatch(input: string, ctx: Ctx): void {
  const tokens = input.trim().split(/\s+/)
  const first = tokens[0]

  // 1) slash 命令
  if (first.startsWith('/')) {
    const handler = slashCommands[first]
    if (!handler) return ctx.error('unknown command')
    handler(ctx)
    return  
  }

  // 2) flag 命令
  const flag = flagCommands.find(f => f.aliases.includes(first))
  if (flag) {
    const value = tokens.slice(1)
    if(value.length < flag.arity) return ctx.error(`missing argument for ${first}`)
    const err = flag.validate?.(value)
    if (err) return ctx.error(err)
    flag.apply(ctx, value)
    return
  }

  // 3) 不是命令 → instruction 执行
  ctx.execute(input.trim())
}