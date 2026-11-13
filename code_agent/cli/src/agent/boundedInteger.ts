// [maint-note:8ad1ff9334cf]
// 维护导航：boundedInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 输入 fallback：number。
// 输入 minimum：number。
// 输入 maximum：number。
// 输入 name：string。
// 返回契约：number。
// 分支条件：value === undefined。
// 分支条件：!Number.isSafeInteger(value) || value < minimum || value > maximum。
// 返回路径：fallback。
// 返回路径：value。
// 显式抛出：new RangeError(`${name} must be an integer between ${minimum} and ${maximum}`)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:8ad1ff9334cf]
export function boundedInteger(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  name: string,
): number {
  if (value === undefined) return fallback
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer between ${minimum} and ${maximum}`)
  }
  return value
}
