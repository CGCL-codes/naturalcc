import { lstat, readFile } from 'node:fs/promises'
import { stat } from 'node:fs/promises'

import type { Tool, ToolExecutionContext } from './types.js'
import { ToolInputError } from './types.js'
import { resolveWorkspacePath } from './workspace.js'
import {
  assertKnownKeys,
  assertObject,
  assertOutputBudget,
  boundedJsonResult,
  boundedText,
  optionalPositiveInteger,
  requiredString,
  throwIfAborted,
} from './toolUtils.js'

const MAX_READ_BYTES = 2_000_000
const MAX_READ_LINES = 5000

interface ReadFileInput {
  path: string
  startLine: number
  endLine?: number
}

export const readFileTool: Tool = {
  name: 'read_file',
  description: '按行读取项目内文本文件；路径必须相对项目根目录。',
  readOnly: true,
  requiresApproval: false,
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string' },
      startLine: { type: 'integer', minimum: 1 },
      endLine: { type: 'integer', minimum: 1 },
    },
    required: ['path'],
    additionalProperties: false,
  },
  parse: parseReadFileInput,
  execute: executeReadFile,
}

export function parseReadFileInput(input: Record<string, unknown>): ReadFileInput {
  assertObject(input)
  assertKnownKeys(input, ['path', 'startLine', 'endLine'])
  const startLine = optionalPositiveInteger(input, 'startLine', 1, MAX_READ_LINES)
  const endLine = input.endLine === undefined
    ? undefined
    : optionalPositiveInteger(input, 'endLine', 1, MAX_READ_LINES)
  if (endLine !== undefined && endLine < startLine) {
    throw new ToolInputError('endLine must be greater than or equal to startLine')
  }
  return {
    path: requiredString(input, 'path', 4096),
    startLine,
    endLine,
  }
}

async function executeReadFile(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
) {
  const options = parseReadFileInput(input)
  assertOutputBudget(context.maxOutputChars)
  const path = await resolveWorkspacePath(context.projectDir, options.path)
  const info = await lstat(path)
  if (!info.isFile()) throw new Error(`Not a regular file: ${options.path}`)
  const fileInfo = await stat(path)
  if (fileInfo.size > MAX_READ_BYTES) throw new Error(`File is larger than ${MAX_READ_BYTES} bytes`)
  throwIfAborted(context.signal)
  const bytes = await readFile(path, { signal: context.signal })
  let raw: string
  try {
    raw = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    throw new Error(`File is not valid UTF-8: ${options.path}`)
  }
  if (raw.includes('\0')) throw new Error(`File is not a text file: ${options.path}`)
  const lines = raw.split(/\r?\n/)
  const start = options.startLine - 1
  const end = options.endLine ?? Math.min(lines.length, start + MAX_READ_LINES)
  const selected = lines.slice(start, end).join('\n')
  const limit = Math.max(0, context.maxOutputChars ?? 8000)
  const bounded = boundedText(selected, limit)
  return {
    content: boundedJsonResult({
      path: options.path,
      startLine: options.startLine,
      endLine: Math.min(end, lines.length),
      content: bounded.text,
      truncated: bounded.truncated || end < lines.length,
    }, context.maxOutputChars, {
      path: '',
      startLine: options.startLine,
      endLine: options.startLine,
      content: '',
      truncated: true,
    }),
  }
}
