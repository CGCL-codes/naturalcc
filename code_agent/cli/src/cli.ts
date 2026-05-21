import { spawnSync } from 'node:child_process'
import { Command, Option } from 'commander'

async function resolvePrompt(parts: string[]): Promise<string> {
    const direct = parts.join(' ').trim()
    if (direct) return direct
    return ''
}

export async function runCli(args: string[]): Promise<void> {
    const CLIAgent = new Command()
  
    CLIAgent
      .name('naturalcc')
      .description('naturalcc CLI测试')
      .version('0.0.0', '-v, --version', '显示版本号')
  
    // 默认命令：直接传问题
    CLIAgent
      .argument('[prompt...]')
      .option('-f, --file [file...]', '目标文件列表，如 src/main.c src/utils.c', [])
      .option('-i, --instruction <instruction>', '你的修改需求')
      .option('-m, --model <model>', '使用的模型', 'openrouter/deepseek/deepseek-chat')
      .option('-k, --apiKey <apiKey>', 'API Key(默认读环境变量)')
      .option('-d, --projectDir <dir>', '项目根目录，默认使用当前运行程序的目录', process.cwd())
      .option('-s, --symbol <symbol>', '目标符号(可选)')
      .addOption(new Option('-t, --completionType <type>', '补全类型(可选)')
        .choices(['member', 'variable', 'function', 'function_body', 'type']))
      .option('--prefix <prefix>', '补全前缀')
      .option('--preview' ,'仅预览最终 Prompt ，不执行 Aider', false)
      .action(async (promptParts: string[], opts) => {
        const prompt = await resolvePrompt(promptParts)
        if (!prompt) {
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

        const payload = JSON.stringify({
          target_files: files,
          user_instruction: opts.instruction,
          model: opts.model,
          api_key: opts.apiKey ?? null,
          project_dir: opts.projectDir,
          symbol: opts.symbol ?? null,
          completion_type: opts.completionType ?? null,
          prefix: opts.prefix ?? "",
        })

        const fn = opts.preview ? 'preview_prompt' : 'run_aider_cli'

        const script = [
          'import sys, json',
          'sys.path.insert(0, "..")',
          `from aider_runner import ${fn}`,
          `print(${fn}(**json.loads(sys.stdin.read())))`,
        ].join('; ')

        const result = spawnSync('python3', ['-c', script], {
          input: payload,
          encoding: 'utf-8',
        })

        if (result.error) {
          console.error('Failed to spawn Python:', result.error.message)
        } else {
          console.log(result.stdout.trimEnd())
          if (result.stderr) console.error(result.stderr.trimEnd())
        }
        return
      })

    await CLIAgent.parseAsync(['node', 'myagent', ...args])
  }