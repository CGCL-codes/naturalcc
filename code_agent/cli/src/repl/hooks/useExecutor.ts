import { useState, useRef } from 'react'
import { spawnSync, spawn, type ChildProcess } from 'node:child_process'
import { resolve } from 'node:path'
import type { Ctx } from '../../types/ctx.js'
import { codeAgentDir, pythonPrelude } from '../../pythonPath.js'
import { displayInstruction } from '../../featurePolicy.js'

interface ExecutorDeps {
  files: string[]
  model: string
  apiKey: string | null
  projectDir: string
  symbol: string | null
  completionType: string | null
  prefix: string
  preview: boolean
  feature: string
  featureConfig: Record<string, unknown>
  addMsg: Ctx['addMsg']
}

interface FeatureEvent {
  type?: string
  status?: string
  log?: string
  report?: string
  mode?: string
    artifacts?: {
    html?: string
    html_path?: string
    nodes?: number
    edges?: number
    modules?: number
  }
  files_modified?: string[]
}

function eventText(event: FeatureEvent): string {
  const base = event.log || event.report || ''
  const artifacts = event.artifacts
  if (!artifacts || event.type !== 'done') return base

  const lines = [base.trimEnd()]
  if (typeof artifacts.html_path === 'string') lines.push(`HTML: ${artifacts.html_path}`)
  if (typeof artifacts.modules === 'number') lines.push(`Modules: ${artifacts.modules}`)
  if (typeof artifacts.nodes === 'number') lines.push(`Nodes: ${artifacts.nodes}`)
  if (typeof artifacts.edges === 'number') lines.push(`Edges: ${artifacts.edges}`)

  return lines.filter(Boolean).join('\n') + '\n'
}

function isErrorPreview(text: string): boolean {
  return text.startsWith('❌') || text.startsWith('Unknown feature') || text.startsWith('Feature preview failed')
}

export function useExecutor(deps: ExecutorDeps) {
  const [loading, setLoading] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')

  const startTimeRef = useRef(0)
  const thinkTimeRef = useRef<number | null>(null)
  const isStreaming = useRef(false)
  const lastInstructionRef = useRef('')
  const childRef = useRef<ChildProcess | null>(null)
  const interrupted = useRef(false)
  const hasLastInstructionRef = useRef(false)

  function execute(input: string) {
    const { files, model, apiKey, projectDir, symbol, completionType, prefix, preview, feature, featureConfig } = deps
    const addMsg = deps.addMsg

    if (feature === 'code_completion' && files.length === 0) {
      addMsg('error', 'No file is selected')
      return
    }

    addMsg('user', displayInstruction(feature, input))
    startTimeRef.current = Date.now()
    thinkTimeRef.current = null
    isStreaming.current = false
    setLoading(true)
    setStreamingContent('')
    lastInstructionRef.current = input
    hasLastInstructionRef.current = true

    let fullContent = ''

    const payload = JSON.stringify({
      target_files: files,
      user_instruction: input,
      model,
      api_key: apiKey ?? null,
      project_dir: resolve(projectDir),
      symbol: symbol ?? null,
      completion_type: completionType ?? null,
      prefix: prefix ?? '',
      feature,
      feature_config: featureConfig,
    })

    if (preview) {
      runPreview(payload)
    } else {
      runStream(payload)
    }

    function runPreview(payload: string) {
      const script = [
        pythonPrelude,
        'from feature_runner import preview_feature',
        'print(preview_feature(**json.loads(sys.stdin.read())))',
      ].join('\n')

      try {
        const result = spawnSync('python3', ['-c', script], {
          input: payload,
          encoding: 'utf-8',
          cwd: codeAgentDir,
        })

        isStreaming.current = true
        thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000

        if (result.error) {
          addMsg('error', `[错误] ${result.error.message}`)
          setLoading(false)
          return
        }

        const stdout = result.stdout.trimEnd()
        const stderr = result.stderr.trimEnd()

        if (stdout) {
          fullContent = stdout
        }
        if (stderr) {
          addMsg('error', `[错误] ${stderr}`)
        }
        if (result.signal) {
          addMsg('error', `[错误] Python process terminated by signal: ${result.signal}`)
          setLoading(false)
          return
        }
        if (result.status !== 0) {
          addMsg('error', `[错误] Python process exited with code ${result.status ?? 1}`)
          setLoading(false)
          return
        }

        fullContent += '\nworked for ' + ((Date.now() - startTimeRef.current) / 1000).toFixed(1) + ' s'
        addMsg(isErrorPreview(fullContent) ? 'error' : 'assistant', fullContent)
        setLoading(false)
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        addMsg('error', `[错误] ${message}`)
        setLoading(false)
      }
    }

    function runStream(payload: string) {
      const script = [
        pythonPrelude,
        'from feature_runner import run_feature_stream',
        'for event in run_feature_stream(**json.loads(sys.stdin.read())):',
        '  print("<<NCC>>" + event, end="" if event.endswith("\\n") else "\\n", flush=True)',
      ].join('\n')

      try {
        const child = spawn('python3', ['-c', script], { cwd: codeAgentDir })
        childRef.current = child
        child.stdin.write(payload)
        child.stdin.end()
        let stdoutBuf = ''
        let errBuf = ''
        let hasErrorEvent = false
        let lastEvent: FeatureEvent | null = null

        child.stdout.on('data', (chunk: Buffer) => {
          if (fullContent === '' && !isStreaming.current) {
            isStreaming.current = true
          }
          stdoutBuf += chunk.toString()
          let newlineIndex = stdoutBuf.indexOf('\n')
          while (newlineIndex !== -1) {
            const rawLine = stdoutBuf.slice(0, newlineIndex)
            stdoutBuf = stdoutBuf.slice(newlineIndex + 1)
            newlineIndex = stdoutBuf.indexOf('\n')

            if (!rawLine.trim()) continue
            const line = rawLine.startsWith('<<NCC>>') ? rawLine.slice('<<NCC>>'.length) : rawLine
            try {
              const event = JSON.parse(line) as FeatureEvent
              lastEvent = event
              if (event.status === 'error' || event.type === 'error') hasErrorEvent = true
              const text = eventText(event)
              if (text) {
                fullContent = text
                setStreamingContent(fullContent)
              }
            } catch {
              fullContent = line
              setStreamingContent(fullContent)
            }
          }
        })

        child.on('close', (code, signal) => {
          const stderr = errBuf.trimEnd()
          if (interrupted.current) {
            if (fullContent) addMsg('assistant', fullContent)
            addMsg('error', 'user interrupted')
            interrupted.current = false
          } else if (signal) {
            if (fullContent) addMsg('assistant', fullContent)
            if (stderr) addMsg('error', `[错误] ${stderr}`)
            addMsg('error', `[错误] Python process terminated by signal: ${signal}`)
          } else if (code !== 0) {
            if (fullContent) addMsg('assistant', fullContent)
            if (stderr) addMsg('error', `[错误] ${stderr}`)
            addMsg('error', `[错误] Python process exited with code ${code ?? 1}`)
          } else if (hasErrorEvent || lastEvent?.status === 'error') {
            if (fullContent) addMsg('error', fullContent)
            if (stderr) addMsg('error', `[错误] ${stderr}`)
          } else {
            thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000
            fullContent += '\nworked for ' + thinkTimeRef.current.toFixed(1) + ' s'
            addMsg('assistant', fullContent)
            if (stderr) addMsg('error', `[错误] ${stderr}`)
          }
          childRef.current = null
          setStreamingContent('')
          setLoading(false)
        })

        child.on('error', (err) => {
          addMsg('error', `[错误] ${err.message}`)
          isStreaming.current = true
          thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000
          setLoading(false)
        })

        child.stderr.on('data', (chunk) => {
          errBuf += chunk.toString()
        })
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        addMsg('error', `[错误] ${message}`)
        setStreamingContent('')
        setLoading(false)
      }
    }
  }

  function rerun() {
    if (!hasLastInstructionRef.current) {
      deps.addMsg('error', 'no previous instruction')
      return
    }
    execute(lastInstructionRef.current)
  }

  function interrupt() {
    if (loading && childRef.current) {
      childRef.current.kill()
      interrupted.current = true
      setLoading(false)
    }
  }

  return { loading, streamingContent, thinkTimeRef, isStreaming, execute, rerun, interrupt }
}
