import { render } from 'ink'
import { Repl } from './repl.js'

export async function startRepl(): Promise<void> {
  const app = render(<Repl />, { exitOnCtrlC: false })

  // 兜底：吞掉漏过 Ink input 层的 SIGINT，控制权完全交给 useInput
  process.on('SIGINT', () => {})

  await app.waitUntilExit()

  // 清理：手动终止进程，避免子进程变成孤儿
  process.exit(0)
}
