import type {Ctx} from '../../types/ctx.js'
import {help_info} from '../help.js'
import { featureSchemaText, listFeaturesText } from './features.js'

export interface SlashCommandDef {
  name: string
  description: string
}

interface SlashCommand extends SlashCommandDef {
  run: (ctx: Ctx, rawArguments: string) => void
}

const slashCommandEntries: SlashCommand[] = [
  { name: '/help', description: '显示帮助信息', run: (ctx) => ctx.addMsg('assistant', help_info()) },
  { name: '/features', description: '列出可用功能插件', run: (ctx) => ctx.addMsg('assistant', listFeaturesText()) },
  { name: '/feature-schema', description: '查看功能插件配置字段', run: (ctx, rawArguments) => ctx.addMsg('assistant', featureSchemaText(rawArguments)) },
  { name: '/exit', description: '退出 REPL', run: (ctx) => { process.stdout.write('\n'); ctx.exit() } },
  { name: '/clear', description: '清除对话历史', run: (ctx) => { ctx.clearMessages() } },
  { name: '/settings', description: '显示/隐藏当前设置面板', run: (ctx) => { ctx.toggleSettings() } },
  { name: '/reset', description: '重置设置', run: (ctx) => { ctx.resetSettings() } },
]

export const slashCommandDefs: SlashCommandDef[] = slashCommandEntries.map(
  ({ name, description }) => ({ name, description }),
)

const slashCommands: Record<string, SlashCommand['run']> = Object.fromEntries(
  slashCommandEntries.map((command) => [command.name, command.run]),
)

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
    handler(ctx, rawArguments)
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
