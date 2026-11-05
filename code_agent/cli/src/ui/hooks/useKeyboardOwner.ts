import { useEffect, useRef } from 'react'
import { useStdin } from 'ink'
import { createKeyboardInputDecoder, type KeyboardInputBatchHandler } from '../state/keyboardInput.js'

/** The single active keyboard subscription for the REPL. */
export function useKeyboardOwner(handler: KeyboardInputBatchHandler, active = true): void {
  const { stdin, setRawMode } = useStdin()
  const handlerRef = useRef(handler)
  handlerRef.current = handler

  useEffect(() => {
    if (!active) return
    setRawMode(true)
    const decoder = createKeyboardInputDecoder()
    let escapeTimer: ReturnType<typeof setTimeout> | null = null
    const deliver = (): void => {
      if (decoder.isWaitingForMoreInput()) return
      const decoded = decoder.flush()
      if (decoded.length > 0) handlerRef.current(decoded)
    }
    const scheduleEscapeFlush = (): void => {
      if (escapeTimer !== null) clearTimeout(escapeTimer)
      escapeTimer = setTimeout(() => {
        escapeTimer = null
        deliver()
      }, 25)
    }
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
