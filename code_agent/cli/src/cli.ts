import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { Command, Option } from 'commander'
import { codeAgentDir, pythonPrelude } from './pythonPath.js'
import { VERSION } from './version.js'

async function resolvePrompt(parts: string[]): Promise<string> {
    const direct = parts.join(' ').trim()
    if (direct) return direct
    return ''
}

function parseFeatureConfig(raw: string | undefined): Record<string, unknown> {
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('feature config must be a JSON object')
    }
    return parsed as Record<string, unknown>
}

export async function runCli(args: string[]): Promise<void> {
  const CLIAgent = new Command()

  CLIAgent
    .name('naturalcc')
    .description('naturalcc CLI测试')
    .version(VERSION, '-v, --version', '显示版本号')

  // 默认命令：直接传问题
  CLIAgent
    .argument('[prompt...]')
    .option('-f, --file [file...]', '目标文件列表，如 src/main.c src/utils.c', [])
    .option('-i, --instruction <instruction>', '你的修改需求')
    .option('-m, --model <model>', '使用的模型', 'deepseek/deepseek-chat')
    .option('-k, --apiKey <apiKey>', 'API Key(默认读环境变量)')
    .option('-d, --projectDir <dir>', '项目根目录，默认使用当前运行程序的目录', process.cwd())
    .option('-s, --symbol <symbol>', '目标符号(可选)')
    .addOption(new Option('-t, --completionType <type>', '补全类型(可选)')
      .choices(['member', 'variable', 'function', 'function_body', 'type']))
    .option('--prefix <prefix>', '补全前缀')
    .option('--feature <feature>', '功能插件', 'code_completion')
    .option('--feature-config <json>', '功能插件配置(JSON对象)')
    .option('--preview' ,'仅预览最终 Prompt ，不执行 Aider', false)
    .action(async (promptParts: string[], opts) => {
      const prompt = await resolvePrompt(promptParts)
      const optionInstruction = opts.instruction?.trim()
      const instruction = optionInstruction || prompt
      if (!instruction) {
        if (process.stdin.isTTY) {
          const { startRepl } = await import('./repl/app.js')
          await startRepl()
        } else {
          CLIAgent.help()
        }
        return
      }

      const files: string[] = Array.isArray(opts.file) ? opts.file
        : typeof opts.file === 'string' ? [opts.file]
        : []
      let featureConfig: Record<string, unknown>
      try {
        featureConfig = parseFeatureConfig(opts.featureConfig)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        console.error(`Invalid --feature-config: ${message}`)
        process.exitCode = 1
        return
      }

      const payload = JSON.stringify({
        target_files: files,
        user_instruction: instruction,
        model: opts.model,
        api_key: opts.apiKey ?? null,
        project_dir: resolve(opts.projectDir),
        symbol: opts.symbol ?? null,
        completion_type: opts.completionType ?? null,
        prefix: opts.prefix ?? "",
        feature: opts.feature,
        feature_config: featureConfig,
      })

      const fn = opts.preview ? 'preview_feature' : 'run_feature_stream'
      const script = [
        pythonPrelude,
        `from feature_runner import ${fn}`,
        opts.preview
          ? `print(${fn}(**json.loads(sys.stdin.read())))`
          : [
              'last_log = ""',
              'last_status = "success"',
              `for event_line in ${fn}(**json.loads(sys.stdin.read())):`,
              '  event = json.loads(event_line)',
              '  last_status = event.get("status", last_status)',
              '  last_log = event.get("log") or event.get("report") or last_log',
              'print(last_log, end="" if last_log.endswith("\\n") else "\\n")',
              'sys.exit(1 if last_status == "error" else 0)',
            ].join('\n'),
      ].join('\n')

      const result = spawnSync('python3', ['-c', script], {
        input: payload,
        encoding: 'utf-8',
        cwd: codeAgentDir,
      })

      if (result.error) {
        console.error('Failed to spawn Python:', result.error.message)
        process.exitCode = 1
        return
      }

      if (result.stdout) {
        console.log(result.stdout.trimEnd())
      }
      if (result.stderr) {
        console.error(result.stderr.trimEnd())
      }
      if (result.signal) {
        console.error(`Python process terminated by signal: ${result.signal}`)
        process.exitCode = 1
        return
      }
      if (result.status !== 0) {
        process.exitCode = result.status ?? 1
      }
    })

  await CLIAgent.parseAsync(['node', 'myagent', ...args])
}
