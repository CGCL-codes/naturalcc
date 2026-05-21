import { useState, useEffect, useRef } from 'react'
import { Box, Text, useApp, useInput, useStdout } from 'ink'
import TextInput from 'ink-text-input'
import { spawnSync,spawn,ChildProcess } from 'node:child_process'

import type { Message } from '../types/message.js'
import { help_info } from './help.js'

function now() : string{
  const d = new Date()
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function useTerminalWidth() {
  const { stdout } = useStdout()
  const [width, setWidth] = useState(stdout?.columns ?? 80)
  const prevWidthRef = useRef(stdout?.columns ?? 80)

  useEffect(() => {
    const onResize = () => {
      const newWidth = stdout.columns ?? 80
      if (newWidth < prevWidthRef.current) {
        stdout.write('\x1b[H\x1b[2J\x1b[H')
      }
      prevWidthRef.current = newWidth
      setWidth(newWidth)
    }
    stdout?.on('resize', onResize)
    return () => {
      stdout?.off('resize', onResize)
    }
  }, [stdout])

  return width
}

export function Repl() {
  const { exit } = useApp();
  const [messages, setMessages] = useState<Message[]>([])
  const [streamingContent, setStreamingContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [dots, setDots] = useState('.')
  const [input, setInput] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [settingsWindow,setSettingsWindow] = useState<boolean>(true)
  const [exitWarning,setExitWarning] = useState<boolean>(false)

  const [files,setFiles] = useState<string[]>([])
  const [model,setModel] = useState<string>('openrouter/deepseek/deepseek-chat')
  const [apiKey,setApiKey] = useState<string | null>(null)
  const [projectDir,setProjectDir] = useState<string>(process.cwd())
  const [symbol,setSymbol] = useState<string | null>(null)
  const [completionType,setCompletionType] = useState<string | null>(null)
  const [prefix,setPrefix] = useState<string>('')
  const [preview,setPreview] = useState<Boolean>(false)

  const startTimeRef = useRef(0)
  const thinkTimeRef = useRef<null | number>(null)
  const isStreaming = useRef<boolean>(false)
  const lastInstructionRef = useRef<string>('')
  const childRef = useRef<ChildProcess | null>(null)
  const interrupted = useRef<boolean>(false)
  const width = useTerminalWidth();

  function msg(role: Message['role'], content: string): Message {
    return { role, content, time: now() }
  }

  function addMsg(role: Message['role'], content: string) {
    setMessages(prev => [...prev, msg(role, content)])
  }

  // 计时
  useEffect(() => {
    if (!loading) return
    const id = setInterval(() => {
      setElapsed((Date.now() - startTimeRef.current) / 1000)
    }, 100)
    return () => clearInterval(id)
  }, [loading])

  // 动态...
  useEffect(() => {
    if (!loading) return
    const id = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '.' : prev + '.')
    }, 400)
    return () => clearInterval(id)
  }, [loading])

  useInput((input, key) => {
    if (input === 'c' && key.ctrl) {
      if (exitWarning) {
        process.stdout.write('\n')
        exit()
        return
      }
      setExitWarning(true)
      setTimeout(() => {setExitWarning(false)} , 3000)
      return
    }
    if(key.escape){
      if(loading && childRef.current){
        childRef.current.kill()
        interrupted.current = true
        setLoading(false)
      }
    }
  })

  const handleSubmit = async (value: string) => {
    const trimmed = value.trim()
    if (!trimmed || loading) return;
    if (trimmed[0] === '/') {
      if (trimmed === '/exit') {
        process.stdout.write('\n')
        exit();
        return;
      }
      if (trimmed === '/clear') {
        setMessages([])
        setStreamingContent('')
        setInput('')
        return;
      }
      if (trimmed === '/help') {
        addMsg('assistant', help_info())
        setInput('')
        return;
      }
      if (trimmed === '/settings') {
        setSettingsWindow(prev => !prev)
        setInput('')
        return
      }
      if(trimmed === '/reset'){
        setFiles([])
        setModel('openrouter/deepseek/deepseek-chat')
        setApiKey(null)
        
      }else {
        addMsg('error', 'unknown command')
        setInput('')
        return
      }
    }
    if (trimmed[0] === '-') {
      const command = trimmed.split(/\s+/)
      const op = command[0]
      if (op === '-f' || op === '--file') {
        if (command.length < 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        setFiles([...command.slice(1)])
        setInput('')
        return
      }
      if (op === '-m' || op === '-model') {
        if (command.length !== 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        if (command[1]) setModel(command[1])
        setInput('')
        return
      }
      if (op === '-k' || op === '--apiKey') {
        if (command.length !== 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        if (command[1]) setApiKey(command[1])
        setInput('')
        return
      }
      if (op === '-d' || op === '--projectDir') {
        if (command.length !== 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        if (command[1]) setProjectDir(command[1])
        setInput('')
        return
      }
      if (op === '-s' || op === '--symbol') {
        if (command.length !== 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        if (command[1]) setSymbol(command[1])
        setInput('')
        return
      }
      if (op === '-t' || op === '--completionType') {
        if (command.length !== 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        const validType = ['member', 'variable', 'function', 'function_body', 'type']
        if (command[1]) {
          if (!validType.includes(command[1])) {
            addMsg('user', trimmed)
            addMsg('error', 'invalid completionType')
            setInput('')
            return
          }
          setCompletionType(command[1])
          setInput('')
        }
        return
      }
      if (op === '--prefix') {
        if (command.length !== 2) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        if (command[1]) {
          setPrefix(command[1])
          setInput('')
        }
        return
      }
      if (op === '--preview') {
        if (command.length > 1) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        setPreview(prev => !prev)
        setInput('')
        return
      }
      if (op === '--run') {
        if (command.length > 1) {
          addMsg('user', trimmed)
          addMsg('error', 'invalid command')
          setInput('')
          return
        }
        if (!lastInstructionRef.current) {
          addMsg('user', trimmed)
          addMsg('error', 'no previous instruction')
          setInput('')
          return
        }
        // --run 处理在下方统一执行
      } else {
        addMsg('user', trimmed)
        addMsg('error', 'unknown command')
        setInput('')
        return
      }
    }

    if (files.length === 0) {
      addMsg('error', 'No file is seleted')
      setInput('')
      return
    }

    const instruction = trimmed === '--run' ? lastInstructionRef.current : trimmed
    addMsg('user', instruction)
    setInput('')
    startTimeRef.current = Date.now()
    thinkTimeRef.current = null
    isStreaming.current = false
    setElapsed(0)
    setLoading(true)
    setStreamingContent('')
    lastInstructionRef.current = instruction

    let fullContent = ''
    let errorContent = ''
    try {
      const payload = JSON.stringify({
        target_files: files,
        user_instruction: instruction,
        model: model,
        api_key: apiKey ?? null,
        project_dir: projectDir,
        symbol: symbol ?? null,
        completion_type: completionType ?? null,
        prefix: prefix ?? "",
      })

      if(preview){//预览输出
        const script = [
        'import sys, json',
        'sys.path.insert(0, "..")',
        `from aider_runner import preview_prompt`,
        `print(preview_prompt(**json.loads(sys.stdin.read())))`,
      ].join('\n')

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
        } else {
          fullContent += result.stdout.trimEnd()
          if (result.stderr)
            errorContent += result.stderr.trimEnd()
          fullContent += "\nworked for "+((Date.now() - startTimeRef.current) / 1000).toFixed(1)+" s"
          addMsg('assistant', fullContent)
          if(errorContent !== '')
            addMsg('error', `[错误] ${errorContent}`)
          setLoading(false)
        }
      } else {//流式输出
        const script = [
          'import sys, json',
          'sys.path.insert(0, "..")',
          `from aider_runner import run_aider_stream`,
          `for chunk in run_aider_stream(**json.loads(sys.stdin.read())):`,
          `  print("<<NCC>>" + chunk, flush=True)`,
        ].join('\n')

        const result = spawn('python3', ['-c', script])
        childRef.current = result
        result.stdin.write(payload)
        result.stdin.end()

        result.stdout.on('data', (chunk: Buffer) => {
          if(fullContent === '' && isStreaming.current === false){
            isStreaming.current = true
          }
          const parts = chunk.toString().split('<<NCC>>')
          fullContent = parts[parts.length - 1] ?? ''
          setStreamingContent(fullContent)
        })

        result.on('close',()=>{
          if(interrupted.current){
            if(fullContent) addMsg('assistant', fullContent)
            addMsg('error','user interrupted')
            interrupted.current = false
          }
          else{
            thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000
            fullContent += "\nworked for "+(thinkTimeRef.current).toFixed(1)+" s"
            addMsg('assistant', fullContent)
          }
          childRef.current = null
          setStreamingContent('')
          setLoading(false)
        })

        result.on('error', (err) => {
          addMsg('error', `[错误] ${err.message}`)
          isStreaming.current = true
          thinkTimeRef.current = (Date.now() - startTimeRef.current) / 1000
          setLoading(false)
        })

        let errBuf = ''
        result.stderr.on('data', (chunk) => {
          errBuf += chunk.toString()
        })
        result.stderr.on('end', () => {
          if (errBuf && !interrupted.current) {
            addMsg('error', `[错误] ${errBuf}`)
            isStreaming.current = true
          }
        })
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      addMsg('error', `[错误] ${message}`)
      setStreamingContent('')
      setLoading(false)
    }
  }

  return (
    <Box flexDirection="column">
      <Box flexDirection="row" borderStyle="double" width={width-2}>
        <Box flexDirection='column' width={35}>
          <Text color="#8B4513">{"      |\\      _,,,---,,_"}</Text>
          <Text color="#8B4513">{" ZZZzz /,`.-'`'    -.  ;-;;,_"}</Text>
          <Text color="#8B4513">{"      |,4-  ) )-,_. ,\\ (  `'-'"}</Text>
          <Text color="#8B4513">{"       '---''(_/--'  `-'\\_) "}</Text>
        </Box>
        <Box flexDirection="column" width={width-35}>
          <Text>naturalcc CLI</Text>
          <Text>REPL mode{'\x1b[0K'}</Text>
          <Text>v0.0.0</Text>
          <Text>a code agent</Text>
        </Box>
      </Box>
      {messages.map((msg, i) => (
        <Box key={i} flexDirection="column" marginBottom={1}>
          {msg.role === 'user' && (
            <Box flexDirection="row">
              <Text color='green'>{'> '}</Text>
              <Text color='gray'>{msg.time} </Text>
              <Text>{msg.content}</Text>
            </Box>
          )}
          {msg.role === 'assistant' && (
            <Box flexDirection="column">
              <Box flexDirection="row">
                <Text color='blue'>{'● '}</Text>
                <Text color='gray'>{msg.time}</Text>
              </Box>
              <Text>{msg.content}</Text>
            </Box>
          )}
          {msg.role === 'error' && (
            <Box flexDirection="row">
              <Text color='red'>{'⚠ '}</Text>
              <Text color='gray'>{msg.time} </Text>
              <Text color='red'>{msg.content}</Text>
            </Box>
          )}
        </Box>
      ))}
      {loading && !isStreaming.current && (
        <Box flexDirection="column" marginBottom={1}>
          <Box flexDirection="row">
            <Text color="blue">{'● '}</Text>
            <Text color='gray'>{now()}</Text>
          </Box>
          <Text>thinking for <Text color='yellow'>{(thinkTimeRef.current ?? elapsed).toFixed(1)}</Text>s {dots}</Text>
        </Box>
      )}
      {loading && isStreaming.current && (
        <Box flexDirection="column" marginBottom={1}>
          <Box flexDirection="row">
            <Text color="blue">{'● '}</Text>
            <Text color='gray'>{now()}</Text>
          </Box>
          <Text>{streamingContent}</Text>
          <Text>{"\n"}work for <Text color='yellow'>{(thinkTimeRef.current ?? elapsed).toFixed(1)}</Text>s</Text>
        </Box>
      )}
      {loading && (
        <Box>
          <Text>{'─'.repeat(width - 2)}</Text>
          <Text color='gray'>Esc to interrupted</Text>
        </Box>
        )}
      {!loading && (
        <Box flexDirection="column">
          {settingsWindow && (<Box flexDirection='column' borderStyle='round'>
            <Text color="green">files:{files.length > 0 ? files.join(' ') : '-'}</Text>
            <Text color="green">model:{model.trim() ? model : '-'}</Text>
            <Text color="green">apiKey:{apiKey ? apiKey : '-'}</Text>
            <Text color="green">projectDir:{projectDir.trim() ? projectDir : '-'}</Text>
            <Text color="green">symbol:{symbol ? symbol : '-'}</Text>
            <Text color="green">completionType:{completionType ? completionType : '-'}</Text>
            <Text color="green">prefix:{prefix.trim() ? prefix : '-'}</Text>
            <Text color="green">preview:{preview ? 'True' : 'False'}</Text>
          </Box>)}
          <Text color="gray">enter{' '}
            <Text color="red">/help</Text> for help{'  '}
            <Text color="red">/settings</Text> to open or close settings window{'  '}
            <Text color="red">--run</Text> to run last instruction{'  '}
          </Text>
          <Text>{'─'.repeat(width - 2)}</Text>
          <Box>
            <Text color="green">{'> '}</Text>
            <TextInput
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              placeholder="Enter command..."
            />
          </Box>
          <Text>{'─'.repeat(width - 2)}</Text>
          {exitWarning && <Text color='gray'>press ctrl+c again to exit</Text>}
        </Box>
      )}
    </Box>
  )
}
