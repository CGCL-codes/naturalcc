// [maint-note:0d633cc92acf]
// 维护导航：formatTime；下列为本函数源码中的接口与路径索引。
// 输入 timestampMs：number。
// 返回契约：string。
// 返回路径：`${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`。
// 直接调用：pad。
// 直接调用：d.getHours。
// 直接调用：d.getMinutes。
// 直接调用：d.getSeconds。
// [/maint-note:0d633cc92acf]
export function formatTime(timestampMs: number): string {
  const d = new Date(timestampMs)
  // [maint-note:3439ca84508e]
  // 维护导航：pad；下列为本函数源码中的接口与路径索引。
  // 输入 n：number。
  // 直接调用：String(n).padStart。
  // 直接调用：String。
  // [/maint-note:3439ca84508e]
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

// [maint-note:35da60a75405]
// 维护导航：now；下列为本函数源码中的接口与路径索引。
// 返回契约：string。
// 返回路径：formatTime(Date.now())。
// 直接调用：formatTime。
// 直接调用：Date.now。
// [/maint-note:35da60a75405]
export function now(): string {
  return formatTime(Date.now())
}
