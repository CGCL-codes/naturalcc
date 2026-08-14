export function formatTime(timestampMs: number): string {
  const d = new Date(timestampMs)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

export function now(): string {
  return formatTime(Date.now())
}
