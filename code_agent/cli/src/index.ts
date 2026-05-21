const VERSION = '0.0.0'

async function main(): Promise<void> {
  const args = process.argv.slice(2)

  // 快速路径 --version / -v 
  if (args.length === 1 && (args[0] === '--version' || args[0] === '-v')) {
    console.log(VERSION)
    return
  }

  // 动态加载完整 CLI
  const { runCli } = await import('./cli.js')
  await runCli(args)
}

void main()
