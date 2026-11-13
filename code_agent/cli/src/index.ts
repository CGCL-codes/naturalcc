import { VERSION } from './version.js'

// [maint-note:feb1fe43db57]
// 维护导航：main；下列为本函数源码中的接口与路径索引。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：args.length === 1 && (args[0] === '--version' || args[0] === '-v')。
// 返回路径：无返回值。
// 直接调用：process.argv.slice。
// 直接调用：console.log。
// 直接调用：import。
// 直接调用：runCli。
// [/maint-note:feb1fe43db57]
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
