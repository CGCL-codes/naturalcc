import { useEffect, useRef } from 'react'
import { useStdin } from 'ink'
import { createKeyboardInputDecoder, type KeyboardInputBatchHandler } from '../state/keyboardInput.js'

/** The single active keyboard subscription for the REPL. */
// [maint-note:879fb6794818]
// 维护导航：useKeyboardOwner；下列为本函数源码中的接口与路径索引。
// 输入 handler：KeyboardInputBatchHandler。
// 输入 active：类型由调用方与函数体约束；缺省表达式 true。
// 返回契约：void。
// 写入目标：handlerRef.current。
// 直接调用：useStdin。
// 直接调用：useRef。
// 直接调用：useEffect。
// [/maint-note:879fb6794818]
export function useKeyboardOwner(handler: KeyboardInputBatchHandler, active = true): void {
  const { stdin, setRawMode } = useStdin()
  const handlerRef = useRef(handler)
  handlerRef.current = handler

  useEffect(() => {
    if (!active) return
    setRawMode(true)
    const decoder = createKeyboardInputDecoder()
    let escapeTimer: ReturnType<typeof setTimeout> | null = null
    // [maint-note:a7b5204e1bd0]
    // 维护导航：deliver；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 分支条件：decoder.isWaitingForMoreInput()。
    // 分支条件：decoded.length > 0。
    // 返回路径：无返回值。
    // 直接调用：decoder.isWaitingForMoreInput。
    // 直接调用：decoder.flush。
    // 直接调用：handlerRef.current。
    // [/maint-note:a7b5204e1bd0]
    const deliver = (): void => {
      if (decoder.isWaitingForMoreInput()) return
      const decoded = decoder.flush()
      if (decoded.length > 0) handlerRef.current(decoded)
    }
    // [maint-note:194a10db609d]
    // 维护导航：scheduleEscapeFlush；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 分支条件：escapeTimer !== null。
    // 写入目标：escapeTimer。
    // 直接调用：clearTimeout。
    // 直接调用：setTimeout。
    // [/maint-note:194a10db609d]
    const scheduleEscapeFlush = (): void => {
      if (escapeTimer !== null) clearTimeout(escapeTimer)
      escapeTimer = setTimeout(() => {
        escapeTimer = null
        deliver()
      }, 25)
    }
    // [maint-note:a07d4aa604b1]
    // 维护导航：onData；下列为本函数源码中的接口与路径索引。
    // 输入 chunk：Buffer | string。
    // 返回契约：void。
    // 分支条件：decoded.length > 0。
    // 直接调用：decoder.push。
    // 直接调用：handlerRef.current。
    // 直接调用：scheduleEscapeFlush。
    // [/maint-note:a07d4aa604b1]
    const onData = (chunk: Buffer | string): void => {
      const decoded = decoder.push(typeof chunk === 'string' ? chunk : new Uint8Array(chunk))
      if (decoded.length > 0) handlerRef.current(decoded)
      scheduleEscapeFlush()
    }
    stdin.on('data', onData)
    return () => {
      stdin.removeListener('data', onData)
      if (escapeTimer !== null) clearTimeout(escapeTimer)
      // Unmount is ownership revocation, not an input flush. In particular,
      // a pending lone ESC must not be delivered to a component whose handler
      // is already being torn down.
      decoder.discard()
      setRawMode(false)
    }
  }, [active, setRawMode, stdin])
}
