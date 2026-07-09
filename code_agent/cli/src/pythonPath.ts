import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const codeAgentDir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../..',
)

export const pythonPrelude = [
  'import sys, json',
  `sys.path.insert(0, ${JSON.stringify(codeAgentDir)})`,
].join('\n')