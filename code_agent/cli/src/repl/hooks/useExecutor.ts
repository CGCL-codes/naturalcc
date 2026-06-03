import { useState, useRef } from 'react'
import { spawnSync, spawn, type ChildProcess } from 'node:child_process'
import type { Ctx } from '../../types/ctx.js'

interface ExecutorDeps {
  files: string[]
  model: string
  apiKey: string | null
  projectDir: string
  symbol: string | null
  completionType: string | null
  prefix: string
  preview: boolean
  addMsg: Ctx['addMsg']
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

  function execute(input: string) {
    const { files, model, apiKey, projectDir, symbol, completionType, prefix, preview } = deps
    const addMsg = deps.addMsg

    if (files.length === 0) {
      addMsg('error', 'No file is selected')
      return
    }

    addMsg('user', input)
    startTimeRef.current = Date.now()
    thinkTimeRef.current = null
    isStreaming.current = false
    setLoading(true)
    setStreamingContent('')
    lastInstructionRef.current = input

    let fullContent = ''
    let errorContent = ''

    const payload = JSON.stringify({
      target_files: files,
      user_instruction: input,
      model,
      api_key: apiKey ?? null,
      project_dir: projectDir,
      symbol: symbol ?? null,
      completion_type: completionType ?? null,
      prefix: prefix ?? '',
    })

    if (preview) {
      runPreview(payload)
    } else {
      runStream(payload)
    }

    function runPreview(payload: string) {
      const script = [
        'import sys, json',
        'sys.path.insert(0, "..")',
        'from aider_runner import preview_prompt',
        'print(preview_prompt(**json.loads(sys.stdin.read())))',
      ].join('\n')

      try {
        const result = spawnSync('python3', ['-c', script], {
          input: payload,
          encoding: 'utf-8',
        })

        isStreaming.current = true
        thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000

        if (result.error) {
          addMsg('error', `[错误] ${result.error.message}`)
          setLoading(false)
          return
        }

        fullContent = result.stdout.trimEnd()
        if (result.stderr) errorContent = result.stderr.trimEnd()
        fullContent += '\nworked for ' + ((Date.now() - startTimeRef.current) / 1000).toFixed(1) + ' s'
        addMsg('assistant', fullContent)
        if (errorContent) addMsg('error', `[错误] ${errorContent}`)
        setLoading(false)
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        addMsg('error', `[错误] ${message}`)
        setLoading(false)
      }
    }

    function runStream(payload: string) {
      const script = [
        'import sys, json',
        'sys.path.insert(0, "..")',
        'from aider_runner import run_aider_stream',
        'for chunk in run_aider_stream(**json.loads(sys.stdin.read())):',
        '  print("<<NCC>>" + chunk, flush=True)',
      ].join('\n')

      try {
        const child = spawn('python3', ['-c', script])
        childRef.current = child
        child.stdin.write(payload)
        child.stdin.end()

        child.stdout.on('data', (chunk: Buffer) => {
          if (fullContent === '' && !isStreaming.current) {
            isStreaming.current = true
          }
          const parts = chunk.toString().split('<<NCC>>')
          fullContent = parts[parts.length - 1] ?? ''
          setStreamingContent(fullContent)
        })

        child.on('close', () => {
          if (interrupted.current) {
            if (fullContent) addMsg('assistant', fullContent)
            addMsg('error', 'user interrupted')
            interrupted.current = false
          } else {
            thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000
            fullContent += '\nworked for ' + thinkTimeRef.current.toFixed(1) + ' s'
            addMsg('assistant', fullContent)
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

        let errBuf = ''
        child.stderr.on('data', (chunk) => {
          errBuf += chunk.toString()
        })
        child.stderr.on('end', () => {
          if (errBuf && !interrupted.current) {
            addMsg('error', `[错误] ${errBuf}`)
            isStreaming.current = true
          }
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
    if (!lastInstructionRef.current) {
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
