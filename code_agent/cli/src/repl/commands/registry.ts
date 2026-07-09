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
  rawValue?: boolean
  validate?: (value: string[]) => string | null
  apply: (ctx: Ctx, value: string[]) => void
}

const validTypes = new Set(['member', 'variable', 'function', 'function_body', 'type'])

function parseFeatureConfig(raw: string): Record<string, unknown> {
  const parsed = JSON.parse(raw)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('feature config must be a JSON object')
  }
  return parsed as Record<string, unknown>
}

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
	  {
	    aliases: ['--feature'],
	    arity: 1,
	    apply: (ctx, [v]) => ctx.setFeature(v),
	  },
	  {
	    aliases: ['--feature-config'],
	    arity: 1,
      rawValue: true,
	    validate: ([raw]) => {
	      try {
	        parseFeatureConfig(raw)
	        return null
	      } catch (err) {
	        const message = err instanceof Error ? err.message : String(err)
	        return `invalid feature config: ${message}`
	      }
	    },
	    apply: (ctx, [raw]) => ctx.setFeatureConfig(parseFeatureConfig(raw)),
	  },
	  { aliases: ['--preview'], arity: 0, apply: (ctx) => ctx.togglePreview() },
  { aliases: ['--run'], arity: 0, apply: (ctx) => ctx.rerun() },
]

export function dispatch(input: string, ctx: Ctx): void {
  const trimmed = input.trim()
  const separator = trimmed.search(/\s/)

  const first = separator === -1
    ? trimmed
    : trimmed.slice(0,separator)

  const rawArguments = separator === -1
    ? ''
    : trimmed.slice(separator).trim()

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
    const value = flag.rawValue
      ? rawArguments ? [rawArguments] : []
      : rawArguments.split(/\s+/).filter(Boolean)
    if(value.length < flag.arity) return ctx.error(`missing argument for ${first}`)
    const err = flag.validate?.(value)
    if (err) return ctx.error(err)
    flag.apply(ctx, value)
    return
  }

  // 3) 不是命令 → instruction 执行
  ctx.execute(input.trim())
}
