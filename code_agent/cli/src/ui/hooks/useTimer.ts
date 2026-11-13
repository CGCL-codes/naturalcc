import { useState, useEffect } from 'react'

// [maint-note:cb952167b9ee]
// 维护导航：useTimer；下列为本函数源码中的接口与路径索引。
// 输入 loading：boolean。
// 返回路径：{ elapsed, dots } as const。
// 直接调用：useState。
// 直接调用：useEffect。
// [/maint-note:cb952167b9ee]
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