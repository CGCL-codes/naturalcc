import { useState, useEffect } from 'react'

export function useTimer(loading: boolean) {
  const [elapsed, setElapsed] = useState(0)
  const [dots, setDots] = useState('.')

  useEffect(() => {
    if (!loading) return
    const startTime = Date.now()
    const id = setInterval(() => {
      setElapsed((Date.now() - startTime) / 1000)
    }, 100)
    return () => clearInterval(id)
  }, [loading])

  useEffect(() => {
    if (!loading) return
    const id = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '.' : prev + '.')
    }, 400)
    return () => clearInterval(id)
  }, [loading])

  return { elapsed, dots } as const
}
