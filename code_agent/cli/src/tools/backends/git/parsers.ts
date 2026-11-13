import { createHash } from 'node:crypto'

import { GitChangesError, type GitFileRecord, type GitPatchHunk, type GitPatchLayer } from './types.js'

const decoder = new TextDecoder('utf-8', { fatal: true })
export const MAX_GIT_DIFF_LINE_BYTES = 1024 * 1024

// [maint-note:2310ba58d9a6]
// 数据契约导航：GitIndexEntry；字段含义以消费端校验为准。
// 成员 mode：string。
// 成员 oid：string。
// 成员 stage：0 | 1 | 2 | 3。
// 成员 path：string。
// [/maint-note:2310ba58d9a6]
export interface GitIndexEntry {
  mode: string
  oid: string
  stage: 0 | 1 | 2 | 3
  path: string
}

// [maint-note:bc10e77033fa]
// 数据契约导航：GitTreeEntry；字段含义以消费端校验为准。
// 成员 mode：string。
// 成员 type：string。
// 成员 oid：string。
// 成员 path：string。
// [/maint-note:bc10e77033fa]
export interface GitTreeEntry {
  mode: string
  type: string
  oid: string
  path: string
}

// [maint-note:758997fa7e59]
// 数据契约导航：ParsedStatus；字段含义以消费端校验为准。
// 成员 headOid：string | 'unborn' | 'unknown'。
// 成员 records：GitFileRecord[]。
// 成员 facts：Array<{ paths: string[]; value: string }>。
// [/maint-note:758997fa7e59]
export interface ParsedStatus {
  headOid: string | 'unborn' | 'unknown'
  records: GitFileRecord[]
  facts: Array<{ paths: string[]; value: string }>
}

// [maint-note:18fcc1e4d5a9]
// 数据契约导航：NumstatEntry；字段含义以消费端校验为准。
// 成员 path：string。
// 成员 oldPath（可省略）：string。
// 成员 added：number | 'binary'。
// 成员 deleted：number | 'binary'。
// [/maint-note:18fcc1e4d5a9]
export interface NumstatEntry {
  path: string
  oldPath?: string
  added: number | 'binary'
  deleted: number | 'binary'
}

// [maint-note:5a1b40e723c0]
// 数据契约导航：RetainParsedValue；字段含义以消费端校验为准。
// 类型组合：(value: unknown) => void。
// [/maint-note:5a1b40e723c0]
export type RetainParsedValue = (value: unknown) => void

// [maint-note:6bbe44af48df]
// 维护导航：decodePath；下列为本函数源码中的接口与路径索引。
// 输入 value：Buffer。
// 返回契约：string。
// 返回路径：decoder.decode(value)。
// 显式抛出：new GitChangesError( 'git_changes_invalid_path_encoding', 'Git returned a path that is not valid UTF-8', 'invalid_utf8_path', )。
// 直接调用：decoder.decode。
// [/maint-note:6bbe44af48df]
export function decodePath(value: Buffer): string {
  try {
    return decoder.decode(value)
  } catch {
    throw new GitChangesError(
      'git_changes_invalid_path_encoding',
      'Git returned a path that is not valid UTF-8',
      'invalid_utf8_path',
    )
  }
}

// [maint-note:c6411a55d17c]
// 维护导航：parseIndexEntries；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 maxEntries：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：GitIndexEntry[]。
// 分支条件：records.length > maxEntries。
// 分支条件：tab < 0。
// 分支条件：!match。
// 遍历边界：const record ← records。
// 返回路径：result。
// 直接调用：splitNul(output, maxParts(maxEntries, 1)).filter。
// 直接调用：splitNul。
// 直接调用：maxParts。
// 直接调用：observationLimit。
// 直接调用：record.indexOf。
// 直接调用：protocol。
// 直接调用：record.subarray(0, tab).toString。
// 直接调用：record.subarray。
// 直接调用：/^([0-7]{6}) ([0-9a-f]+) ([0-3])$/u.exec。
// 直接调用：Number。
// 直接调用：decodePath。
// 直接调用：retain。
// 直接调用：result.push。
// [/maint-note:c6411a55d17c]
export function parseIndexEntries(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): GitIndexEntry[] {
  const records = splitNul(output, maxParts(maxEntries, 1)).filter((record) => record.length > 0)
  if (records.length > maxEntries) observationLimit('file_count_exceeded')
  const result: GitIndexEntry[] = []
  for (const record of records) {
    const tab = record.indexOf(0x09)
    if (tab < 0) protocol('Invalid ls-files record')
    const header = record.subarray(0, tab).toString('ascii')
    const match = /^([0-7]{6}) ([0-9a-f]+) ([0-3])$/u.exec(header)
    if (!match) protocol('Invalid ls-files stage record')
    const value = {
      mode: match[1]!,
      oid: match[2]!,
      stage: Number(match[3]) as 0 | 1 | 2 | 3,
      path: decodePath(record.subarray(tab + 1)),
    }
    retain?.(value)
    result.push(value)
  }
  return result
}

// [maint-note:055aa9146d74]
// 维护导航：parseTreeEntries；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 maxEntries：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：GitTreeEntry[]。
// 分支条件：records.length > maxEntries。
// 分支条件：tab < 0。
// 分支条件：!match。
// 遍历边界：const record ← records。
// 返回路径：result。
// 直接调用：splitNul(output, maxParts(maxEntries, 1)).filter。
// 直接调用：splitNul。
// 直接调用：maxParts。
// 直接调用：observationLimit。
// 直接调用：record.indexOf。
// 直接调用：protocol。
// 直接调用：record.subarray(0, tab).toString。
// 直接调用：record.subarray。
// 直接调用：/^([0-7]{6}) ([a-z]+) ([0-9a-f]+)$/u.exec。
// 直接调用：decodePath。
// 直接调用：retain。
// 直接调用：result.push。
// [/maint-note:055aa9146d74]
export function parseTreeEntries(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): GitTreeEntry[] {
  const records = splitNul(output, maxParts(maxEntries, 1)).filter((record) => record.length > 0)
  if (records.length > maxEntries) observationLimit('file_count_exceeded')
  const result: GitTreeEntry[] = []
  for (const record of records) {
    const tab = record.indexOf(0x09)
    if (tab < 0) protocol('Invalid ls-tree record')
    const header = record.subarray(0, tab).toString('ascii')
    const match = /^([0-7]{6}) ([a-z]+) ([0-9a-f]+)$/u.exec(header)
    if (!match) protocol('Invalid ls-tree record')
    const value = { mode: match[1]!, type: match[2]!, oid: match[3]!, path: decodePath(record.subarray(tab + 1)) }
    retain?.(value)
    result.push(value)
  }
  return result
}

// [maint-note:26ea76203487]
// 维护导航：parseUntracked；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 maxEntries：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：string[]。
// 分支条件：records.length > maxEntries。
// 遍历边界：const record ← records。
// 返回路径：result。
// 直接调用：splitNul(output, maxParts(maxEntries, 1)).filter。
// 直接调用：splitNul。
// 直接调用：maxParts。
// 直接调用：observationLimit。
// 直接调用：decodePath。
// 直接调用：retain。
// 直接调用：result.push。
// [/maint-note:26ea76203487]
export function parseUntracked(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): string[] {
  const records = splitNul(output, maxParts(maxEntries, 1)).filter((record) => record.length > 0)
  if (records.length > maxEntries) observationLimit('file_count_exceeded')
  const result: string[] = []
  for (const record of records) {
    const value = decodePath(record)
    retain?.(value)
    result.push(value)
  }
  return result
}

// [maint-note:c4458decc5cb]
// 维护导航：parseCheckAttr；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 maxPaths：number。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：Array<{ path: string; attribute: string; value: string }>。
// 分支条件：values.at(-1)?.length === 0。
// 分支条件：values.length % 3 !== 0。
// 分支条件：result.length >= maxRecords。
// 分支条件：!paths.has(path) && paths.size >= maxPaths。
// 分支条件：!requestedAttributes.has(attribute)。
// 分支条件：seenAttributes.has(attribute)。
// 分支条件：paths.size !== maxPaths || result.length !== maxRecords || [...attributesByPath.values()].some((attributes) => attributes.size !== attributesPerPath)。
// 遍历边界：let index = 0; index < values.length; index += 3。
// 返回路径：result。
// 写入目标：index。
// 直接调用：maxParts。
// 直接调用：splitNul。
// 直接调用：values.at。
// 直接调用：values.pop。
// 直接调用：protocol。
// 直接调用：observationLimit。
// 直接调用：decodePath。
// 直接调用：paths.has。
// 直接调用：requestedAttributes.has。
// 直接调用：attributesByPath.get。
// 直接调用：seenAttributes.has。
// 直接调用：retain。
// 直接调用：paths.add。
// 直接调用：seenAttributes.add。
// 直接调用：attributesByPath.set。
// 直接调用：result.push。
// 直接调用：[...attributesByPath.values()].some。
// 直接调用：attributesByPath.values。
// [/maint-note:c4458decc5cb]
export function parseCheckAttr(
  output: Buffer,
  maxPaths: number,
  retain?: RetainParsedValue,
): Array<{ path: string; attribute: string; value: string }> {
  const attributesPerPath = 3
  const maxRecords = maxParts(maxPaths, attributesPerPath)
  const values = splitNul(output, maxParts(maxRecords, 3, 1))
  if (values.at(-1)?.length === 0) values.pop()
  if (values.length % 3 !== 0) protocol('Invalid check-attr record')
  const result: Array<{ path: string; attribute: string; value: string }> = []
  const paths = new Set<string>()
  const attributesByPath = new Map<string, Set<string>>()
  const requestedAttributes = new Set(['filter', 'diff', 'working-tree-encoding'])
  for (let index = 0; index < values.length; index += 3) {
    if (result.length >= maxRecords) observationLimit('wire_record_count_exceeded')
    const path = decodePath(values[index]!)
    if (!paths.has(path) && paths.size >= maxPaths) observationLimit('file_count_exceeded')
    const attribute = decodePath(values[index + 1]!)
    if (!requestedAttributes.has(attribute)) protocol('Unexpected check-attr attribute')
    const seenAttributes = attributesByPath.get(path) ?? new Set<string>()
    if (seenAttributes.has(attribute)) protocol('Duplicate check-attr attribute')
    const value = {
      path,
      attribute,
      value: decodePath(values[index + 2]!),
    }
    retain?.(value)
    paths.add(path)
    seenAttributes.add(attribute)
    attributesByPath.set(path, seenAttributes)
    result.push(value)
  }
  if (
    paths.size !== maxPaths || result.length !== maxRecords ||
    [...attributesByPath.values()].some((attributes) => attributes.size !== attributesPerPath)
  ) protocol('Incomplete check-attr records')
  return result
}

// [maint-note:ba6b99abed0d]
// 维护导航：parsePorcelainV2；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 mapPath：(path: string) => string | undefined。
// 输入 maxRecords：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：ParsedStatus。
// 分支条件：!record。
// 分支条件：header.startsWith('# branch.oid ')。
// 分支条件：newline < 0。
// 分支条件：record.startsWith('? ')。
// 分支条件：path !== undefined。
// 分支条件：records.length >= maxRecords。
// 分支条件：record.startsWith('! ')。
// 分支条件：record.startsWith('1 ')。
// 分支条件：!match。
// 分支条件：path === undefined。
// 分支条件：records.length + layerRecordCount(match[1]!) > maxRecords。
// 分支条件：record.startsWith('2 ')。
// 分支条件：!originalPart。
// 分支条件：newPath === undefined && oldPath === undefined。
// 分支条件：match[7]!.startsWith('C')。
// 分支条件：record.startsWith('u ')。
// 遍历边界：let index = 0; index < parts.length; index += 1。
// 遍历边界：record.startsWith('# ')。
// 返回路径：{ headOid, records, facts }。
// 显式抛出：new GitChangesError('git_changes_ambiguous_change', 'Copy-like Git changes are not supported', 'copy_change')。
// 写入目标：index。
// 写入目标：headOid。
// 写入目标：record。
// 直接调用：splitNul。
// 直接调用：maxParts。
// 直接调用：decodePath。
// 直接调用：record.startsWith。
// 直接调用：record.indexOf。
// 直接调用：record.slice。
// 直接调用：header.startsWith。
// 直接调用：header.slice。
// 直接调用：mapPath。
// 直接调用：observationLimit。
// 直接调用：retain。
// 直接调用：records.push。
// 直接调用：facts.push。
// 直接调用：/^1 (\S{2}) \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([\s\S]+)$/u.exec。
// 直接调用：protocol。
// 直接调用：layerRecordCount。
// 直接调用：appendLayerRecords。
// 直接调用：/^2 (\S{2}) \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([RC][0-9]+) ([\s\S]+)$/u.exec。
// 直接调用：match[7]!.startsWith。
// 直接调用：[newPath, oldPath].filter。
// 直接调用：/^u \S{2} \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) [0-9a-f]+ [0-9a-f]+ [0-9a-f]+ ([\s\S]+)$/u.exec。
// 直接调用：objectType。
// [/maint-note:ba6b99abed0d]
export function parsePorcelainV2(
  output: Buffer,
  mapPath: (path: string) => string | undefined,
  maxRecords = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): ParsedStatus {
  const records: GitFileRecord[] = []
  const facts: ParsedStatus['facts'] = []
  const parts = splitNul(output, maxParts(maxRecords, 2, 2))
  let headOid: ParsedStatus['headOid'] = 'unknown'
  for (let index = 0; index < parts.length; index += 1) {
    let record = decodePath(parts[index]!)
    if (!record) continue
    while (record.startsWith('# ')) {
      const newline = record.indexOf('\n')
      const header = newline < 0 ? record : record.slice(0, newline)
      if (header.startsWith('# branch.oid ')) {
        const value = header.slice('# branch.oid '.length)
        headOid = value === '(initial)' ? 'unborn' : value
      }
      if (newline < 0) {
        record = ''
        break
      }
      record = record.slice(newline + 1)
    }
    if (!record) continue

    if (record.startsWith('? ')) {
      const path = mapPath(record.slice(2))
      if (path !== undefined) {
        if (records.length >= maxRecords) observationLimit('file_count_exceeded')
        const value: GitFileRecord = {
          layer: 'unstaged', action: 'untracked', path,
          contentChanged: true, binary: 'unknown', modeChanged: false, objectType: 'file',
        }
        retain?.(value)
        records.push(value)
        const fact = { paths: [path], value: record }
        retain?.(fact)
        facts.push(fact)
      }
      continue
    }
    if (record.startsWith('! ')) continue

    if (record.startsWith('1 ')) {
      const match = /^1 (\S{2}) \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([\s\S]+)$/u.exec(record)
      if (!match) protocol('Invalid porcelain-v2 ordinary record')
      const path = mapPath(match[7]!)
      if (path === undefined) continue
      if (records.length + layerRecordCount(match[1]!) > maxRecords) observationLimit('file_count_exceeded')
      appendLayerRecords(records, match[1]!, path, undefined, match[2]!, match[3]!, match[4]!, match[5]!, match[6]!, false, retain)
      const fact = { paths: [path], value: record }
      retain?.(fact)
      facts.push(fact)
      continue
    }
    if (record.startsWith('2 ')) {
      const match = /^2 (\S{2}) \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([RC][0-9]+) ([\s\S]+)$/u.exec(record)
      if (!match) protocol('Invalid porcelain-v2 rename record')
      const originalPart = parts[++index]
      if (!originalPart) protocol('Missing porcelain-v2 rename origin')
      const newPath = mapPath(match[8]!)
      const oldRaw = decodePath(originalPart)
      const oldPath = mapPath(oldRaw)
      if (newPath === undefined && oldPath === undefined) continue
      if (match[7]!.startsWith('C')) {
        throw new GitChangesError('git_changes_ambiguous_change', 'Copy-like Git changes are not supported', 'copy_change')
      }
      if (records.length + layerRecordCount(match[1]!) > maxRecords) observationLimit('file_count_exceeded')
      appendLayerRecords(
        records,
        match[1]!,
        newPath ?? oldPath!,
        oldPath,
        match[2]!, match[3]!, match[4]!, match[5]!, match[6]!,
        newPath === undefined || oldPath === undefined,
        retain,
      )
      const fact = { paths: [newPath, oldPath].filter((path): path is string => path !== undefined), value: `${record}\0${oldRaw}` }
      retain?.(fact)
      facts.push(fact)
      continue
    }
    if (record.startsWith('u ')) {
      const match = /^u \S{2} \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) [0-9a-f]+ [0-9a-f]+ [0-9a-f]+ ([\s\S]+)$/u.exec(record)
      if (!match) protocol('Invalid porcelain-v2 unmerged record')
      const path = mapPath(match[5]!)
      if (path !== undefined) {
        if (records.length >= maxRecords) observationLimit('file_count_exceeded')
        const value: GitFileRecord = {
          layer: 'unmerged', action: 'unmerged', path,
          contentChanged: 'unknown', binary: 'unknown', modeChanged: true,
          oldMode: match[1], newMode: match[4], objectType: objectType(match[4]!),
        }
        retain?.(value)
        records.push(value)
        const fact = { paths: [path], value: record }
        retain?.(fact)
        facts.push(fact)
      }
      continue
    }
    protocol('Unsupported porcelain-v2 record')
  }
  return { headOid, records, facts }
}

// [maint-note:6507427e73de]
// 维护导航：appendLayerRecords；下列为本函数源码中的接口与路径索引。
// 输入 target：GitFileRecord[]。
// 输入 xy：string。
// 输入 path：string。
// 输入 oldPath：string | undefined。
// 输入 headMode：string。
// 输入 indexMode：string。
// 输入 workMode：string。
// 输入 headOid：string。
// 输入 indexOid：string。
// 输入 boundaryTruncated：类型由调用方与函数体约束；缺省表达式 false。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：void。
// 分支条件：indexCode && indexCode !== '.'。
// 分支条件：workCode && workCode !== '.'。
// 直接调用：makeRecord。
// 直接调用：retain。
// 直接调用：target.push。
// [/maint-note:6507427e73de]
function appendLayerRecords(
  target: GitFileRecord[],
  xy: string,
  path: string,
  oldPath: string | undefined,
  headMode: string,
  indexMode: string,
  workMode: string,
  headOid: string,
  indexOid: string,
  boundaryTruncated = false,
  retain?: RetainParsedValue,
): void {
  const [indexCode, workCode] = xy
  if (indexCode && indexCode !== '.') {
    const value = makeRecord('staged', indexCode, path, oldPath, headMode, indexMode, headOid !== indexOid, boundaryTruncated)
    retain?.(value)
    target.push(value)
  }
  if (workCode && workCode !== '.') {
    const value = makeRecord('unstaged', workCode, path, workCode === 'R' ? oldPath : undefined, indexMode, workMode, true, boundaryTruncated)
    retain?.(value)
    target.push(value)
  }
}

// [maint-note:dd18bfbef87e]
// 维护导航：makeRecord；下列为本函数源码中的接口与路径索引。
// 输入 layer：'staged' | 'unstaged'。
// 输入 code：string。
// 输入 path：string。
// 输入 oldPath：string | undefined。
// 输入 oldMode：string。
// 输入 newMode：string。
// 输入 contentChanged：boolean。
// 输入 boundaryTruncated：boolean。
// 返回契约：GitFileRecord。
// 返回路径：{ layer, action, path, ...(action === 'renamed' && oldPath ? { oldPath } : {}), ...(boundaryTruncated ? { boundaryTruncated: true } : {}), contentChanged, binary: 'unknown', modeChanged: oldMode !== newMode, oldMode, newMode, objectType: ob。
// 直接调用：objectType。
// [/maint-note:dd18bfbef87e]
function makeRecord(
  layer: 'staged' | 'unstaged',
  code: string,
  path: string,
  oldPath: string | undefined,
  oldMode: string,
  newMode: string,
  contentChanged: boolean,
  boundaryTruncated: boolean,
): GitFileRecord {
  const action = code === 'A' ? 'added'
    : code === 'D' ? 'deleted'
      : code === 'R' ? 'renamed'
        : code === 'T' ? 'type_changed'
          : 'modified'
  return {
    layer,
    action,
    path,
    ...(action === 'renamed' && oldPath ? { oldPath } : {}),
    ...(boundaryTruncated ? { boundaryTruncated: true } : {}),
    contentChanged,
    binary: 'unknown',
    modeChanged: oldMode !== newMode,
    oldMode,
    newMode,
    objectType: objectType(action === 'deleted' ? oldMode : newMode),
  }
}

// [maint-note:e9b811fb4150]
// 维护导航：parseNumstat；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 maxEntries：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：NumstatEntry[]。
// 分支条件：record.length === 0。
// 分支条件：firstTab < 0 || secondTab < 0。
// 分支条件：tail.length > 0。
// 分支条件：entries.length >= maxEntries。
// 分支条件：!oldPart || !newPart。
// 遍历边界：let index = 0; index < parts.length; index += 1。
// 返回路径：entries。
// 写入目标：index。
// 直接调用：splitNul。
// 直接调用：maxParts。
// 直接调用：record.indexOf。
// 直接调用：protocol。
// 直接调用：record.subarray(0, firstTab).toString。
// 直接调用：record.subarray。
// 直接调用：record.subarray(firstTab + 1, secondTab).toString。
// 直接调用：parseCount。
// 直接调用：observationLimit。
// 直接调用：decodePath。
// 直接调用：retain。
// 直接调用：entries.push。
// [/maint-note:e9b811fb4150]
export function parseNumstat(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): NumstatEntry[] {
  const parts = splitNul(output, maxParts(maxEntries, 3, 1))
  const entries: NumstatEntry[] = []
  for (let index = 0; index < parts.length; index += 1) {
    const record = parts[index]!
    if (record.length === 0) continue
    const firstTab = record.indexOf(0x09)
    const secondTab = firstTab < 0 ? -1 : record.indexOf(0x09, firstTab + 1)
    if (firstTab < 0 || secondTab < 0) protocol('Invalid numstat record')
    const addedText = record.subarray(0, firstTab).toString('ascii')
    const deletedText = record.subarray(firstTab + 1, secondTab).toString('ascii')
    const tail = record.subarray(secondTab + 1)
    const added = addedText === '-' ? 'binary' : parseCount(addedText)
    const deleted = deletedText === '-' ? 'binary' : parseCount(deletedText)
    if (tail.length > 0) {
      if (entries.length >= maxEntries) observationLimit('file_count_exceeded')
      const value: NumstatEntry = { path: decodePath(tail), added, deleted }
      retain?.(value)
      entries.push(value)
      continue
    }
    const oldPart = parts[++index]
    const newPart = parts[++index]
    if (!oldPart || !newPart) protocol('Invalid rename numstat record')
    if (entries.length >= maxEntries) observationLimit('file_count_exceeded')
    const value: NumstatEntry = { oldPath: decodePath(oldPart), path: decodePath(newPart), added, deleted }
    retain?.(value)
    entries.push(value)
  }
  return entries
}

// [maint-note:d2479bd47132]
// 维护导航：parsePatch；下列为本函数源码中的接口与路径索引。
// 输入 output：Buffer。
// 输入 layer：'staged' | 'unstaged'。
// 输入 file：GitFileRecord。
// 输入 toWorkspacePath：(repoPath: string) => string | undefined。
// 输入 maxHunks：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 输入 retain：RetainParsedValue（可省略）。
// 返回契约：GitPatchLayer。
// 分支条件：!text。
// 分支条件：file.contentChanged === false || file.binary === true。
// 分支条件：first.done || !first.value.startsWith('diff --git ')。
// 分支条件：toWorkspacePath(headerOld) !== expectedOld || toWorkspacePath(headerNew) !== file.path。
// 分支条件：line.startsWith('diff --git ')。
// 分支条件：line.startsWith('@@ ')。
// 分支条件：!match。
// 分支条件：current.oldStart + Math.max(0, current.oldLines - 1) > Number.MAX_SAFE_INTEGER || current.newStart + Math.max(0, current.newLines - 1) > Number.MAX_SAFE_INTEGER。
// 分支条件：hunks.length >= maxHunks。
// 分支条件：!current。
// 分支条件：line.startsWith('--- ')。
// 分支条件：sawOldHeader。
// 分支条件：line.startsWith('+++ ')。
// 分支条件：sawNewHeader。
// 分支条件：line.startsWith('rename from ')。
// 分支条件：sawRenameFrom。
// 分支条件：line.startsWith('rename to ')。
// 分支条件：sawRenameTo。
// 分支条件：line.startsWith('index ')。
// 分支条件：sawIndex || !match || match[1]!.length !== match[2]!.length。
// 分支条件：line.startsWith('old mode ')。
// 分支条件：sawOldMode || line.slice('old mode '.length) !== file.oldMode。
// 分支条件：line.startsWith('new mode ')。
// 分支条件：sawNewMode || line.slice('new mode '.length) !== file.newMode。
// 分支条件：line.startsWith('new file mode ')。
// 分支条件：sawNewFileMode || file.action !== 'added' || line.slice('new file mode '.length) !== file.newMode。
// 分支条件：line.startsWith('deleted file mode ')。
// 分支条件：sawDeletedFileMode || file.action !== 'deleted' || line.slice('deleted file mode '.length) !== file.oldMode。
// 分支条件：line.startsWith('similarity index ') || line.startsWith('dissimilarity index ')。
// 分支条件：!match || similarityPercent !== undefined。
// 分支条件：line.startsWith('Binary files ') || line.startsWith('GIT binary patch')。
// 分支条件：isAllowedPatchMetadata(line)。
// 分支条件：line === '\\ No newline at end of file'。
// 分支条件：!previousKind。
// 分支条件：prefix === ' '。
// 分支条件：prefix === '-'。
// 分支条件：prefix === '+'。
// 分支条件：sawOldHeader !== sawNewHeader。
// 分支条件：sawRenameFrom !== sawRenameTo。
// 分支条件：sawOldMode !== sawNewMode。
// 分支条件：hunks.length > 0 && (!sawOldHeader || !sawNewHeader)。
// 分支条件：hunks.length === 0。
// 分支条件：sawOldHeader || sawNewHeader。
// 分支条件：!supportedMetadataOnly。
// 遍历边界：const line ← lines。
// 返回路径：{ layer, file, hunks: [] }。
// 返回路径：{ layer, file: projectedFile, hunks }。
// 显式抛出：new GitChangesError( 'git_changes_invalid_content_encoding', 'Git returned a text patch that is not valid UTF-8', 'invalid_utf8_patch', )。
// 写入目标：text。
// 写入目标：current。
// 写入目标：oldLine。
// 写入目标：newLine。
// 写入目标：consumedOld。
// 写入目标：consumedNew。
// 写入目标：previousKind。
// 写入目标：sawOldHeader。
// 写入目标：sawNewHeader。
// 写入目标：sawRenameFrom。
// 写入目标：sawRenameTo。
// 写入目标：sawIndex。
// 写入目标：indexMetadata。
// 写入目标：sawOldMode。
// 写入目标：sawNewMode。
// 写入目标：sawNewFileMode。
// 写入目标：sawDeletedFileMode。
// 写入目标：similarityPercent。
// 写入目标：sawBinary。
// 写入目标：projectedFile.binary。
// 写入目标：projectedFile.contentChanged。
// 直接调用：decoder.decode。
// 直接调用：protocol。
// 直接调用：iteratePatchLines。
// 直接调用：lines.next。
// 直接调用：first.value.startsWith。
// 直接调用：assertPatchLineBound。
// 直接调用：parseDiffGitHeader。
// 直接调用：toWorkspacePath。
// 直接调用：retain。
// 直接调用：line.startsWith。
// 直接调用：finishHunk。
// 直接调用：/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: ?(.*))?$/u.exec。
// 直接调用：parsePatchNumber。
// 直接调用：Math.max。
// 直接调用：observationLimit。
// 直接调用：hunks.push。
// 直接调用：validatePatchEndpoint。
// 直接调用：line.slice。
// 直接调用：validateRenameEndpoint。
// 直接调用：/^index ([0-9a-f]+)\.\.([0-9a-f]+)(?: ([0-7]{6}))?$/u.exec。
// 直接调用：/^(dis)?similarity index (100|[0-9]{1,2})%$/u.exec。
// 直接调用：Number。
// 直接调用：isAllowedPatchMetadata。
// 直接调用：current.lines.push。
// 直接调用：hasFullObjectIds。
// 直接调用：isZeroOid。
// 直接调用：isEmptyBlobOid。
// [/maint-note:d2479bd47132]
export function parsePatch(
  output: Buffer,
  layer: 'staged' | 'unstaged',
  file: GitFileRecord,
  toWorkspacePath: (repoPath: string) => string | undefined,
  maxHunks = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): GitPatchLayer {
  let text: string
  try {
    text = decoder.decode(output)
  } catch {
    throw new GitChangesError(
      'git_changes_invalid_content_encoding',
      'Git returned a text patch that is not valid UTF-8',
      'invalid_utf8_patch',
    )
  }
  if (!text) {
    if (file.contentChanged === false || file.binary === true) return { layer, file, hunks: [] }
    protocol('Git returned no patch for the selected change')
  }
  const lines = iteratePatchLines(text)
  const first = lines.next()
  if (first.done || !first.value.startsWith('diff --git ')) protocol('Patch must contain exactly one section')
  assertPatchLineBound(first.value)
  const expectedOld = file.oldPath ?? file.path
  const [headerOld, headerNew] = parseDiffGitHeader(
    first.value,
    (path) => toWorkspacePath(path),
    expectedOld,
    file.path,
  )
  if (toWorkspacePath(headerOld) !== expectedOld || toWorkspacePath(headerNew) !== file.path) protocol('Patch endpoints do not match selected Git metadata')

  const projectedFile: GitFileRecord = {
    ...file,
  }
  retain?.(projectedFile)
  const hunks: GitPatchHunk[] = []
  let current: GitPatchHunk | undefined
  let oldLine = 0
  let newLine = 0
  let consumedOld = 0
  let consumedNew = 0
  let previousKind: 'context' | 'add' | 'delete' | undefined
  let sawOldHeader = false
  let sawNewHeader = false
  let sawOldMode = false
  let sawNewMode = false
  let sawNewFileMode = false
  let sawDeletedFileMode = false
  let sawRenameFrom = false
  let sawRenameTo = false
  let sawIndex = false
  let indexMetadata: { oldOid: string; newOid: string; mode?: string } | undefined
  let similarityPercent: number | undefined
  let sawBinary = false

  // [maint-note:d590187dde88]
  // 维护导航：finishHunk；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：!current。
  // 分支条件：consumedOld !== current.oldLines || consumedNew !== current.newLines。
  // 返回路径：无返回值。
  // 直接调用：protocol。
  // [/maint-note:d590187dde88]
  const finishHunk = (): void => {
    if (!current) return
    if (consumedOld !== current.oldLines || consumedNew !== current.newLines) {
      protocol('Patch hunk line counts do not match its header')
    }
  }

  for (const line of lines) {
    assertPatchLineBound(line)
    if (line.startsWith('diff --git ')) protocol('Patch must contain exactly one section')
    if (line.startsWith('@@ ')) {
      finishHunk()
      const match = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: ?(.*))?$/u.exec(line)
      if (!match) protocol('Invalid patch hunk header')
      const nextHunk: GitPatchHunk = {
        oldStart: parsePatchNumber(match[1]!), oldLines: parsePatchNumber(match[2] ?? '1'),
        newStart: parsePatchNumber(match[3]!), newLines: parsePatchNumber(match[4] ?? '1'),
        ...(match[5] ? { section: match[5] } : {}), lines: [],
      }
      retain?.({ ...nextHunk, lines: undefined })
      current = nextHunk
      if (
        current.oldStart + Math.max(0, current.oldLines - 1) > Number.MAX_SAFE_INTEGER ||
        current.newStart + Math.max(0, current.newLines - 1) > Number.MAX_SAFE_INTEGER
      ) protocol('Patch hunk line range is unsafe')
      oldLine = current.oldStart
      newLine = current.newStart
      consumedOld = 0
      consumedNew = 0
      if (hunks.length >= maxHunks) observationLimit('hunk_count_exceeded')
      hunks.push(current)
      previousKind = undefined
      continue
    }
    if (!current) {
      if (line.startsWith('--- ')) {
        if (sawOldHeader) protocol('Duplicate old patch endpoint')
        validatePatchEndpoint(line.slice(4), 'old', file, toWorkspacePath)
        sawOldHeader = true
        continue
      }
      if (line.startsWith('+++ ')) {
        if (sawNewHeader) protocol('Duplicate new patch endpoint')
        validatePatchEndpoint(line.slice(4), 'new', file, toWorkspacePath)
        sawNewHeader = true
        continue
      }
      if (line.startsWith('rename from ')) {
        if (sawRenameFrom) protocol('Duplicate rename origin')
        validateRenameEndpoint(line.slice('rename from '.length), file.oldPath ?? file.path, toWorkspacePath)
        sawRenameFrom = true
        continue
      }
      if (line.startsWith('rename to ')) {
        if (sawRenameTo) protocol('Duplicate rename destination')
        validateRenameEndpoint(line.slice('rename to '.length), file.path, toWorkspacePath)
        sawRenameTo = true
        continue
      }
      if (line.startsWith('index ')) {
        const match = /^index ([0-9a-f]+)\.\.([0-9a-f]+)(?: ([0-7]{6}))?$/u.exec(line)
        if (sawIndex || !match || match[1]!.length !== match[2]!.length) protocol('Invalid index metadata')
        sawIndex = true
        indexMetadata = {
          oldOid: match[1]!,
          newOid: match[2]!,
          ...(match[3] === undefined ? {} : { mode: match[3] }),
        }
      } else if (line.startsWith('old mode ')) {
        if (sawOldMode || line.slice('old mode '.length) !== file.oldMode) protocol('Invalid old mode metadata')
        sawOldMode = true
      } else if (line.startsWith('new mode ')) {
        if (sawNewMode || line.slice('new mode '.length) !== file.newMode) protocol('Invalid new mode metadata')
        sawNewMode = true
      } else if (line.startsWith('new file mode ')) {
        if (
          sawNewFileMode || file.action !== 'added' ||
          line.slice('new file mode '.length) !== file.newMode
        ) protocol('Invalid new-file mode metadata')
        sawNewFileMode = true
      } else if (line.startsWith('deleted file mode ')) {
        if (
          sawDeletedFileMode || file.action !== 'deleted' ||
          line.slice('deleted file mode '.length) !== file.oldMode
        ) protocol('Invalid deleted-file mode metadata')
        sawDeletedFileMode = true
      } else if (line.startsWith('similarity index ') || line.startsWith('dissimilarity index ')) {
        const match = /^(dis)?similarity index (100|[0-9]{1,2})%$/u.exec(line)
        if (!match || similarityPercent !== undefined) protocol('Invalid similarity metadata')
        const percentage = Number(match[2])
        similarityPercent = match[1] ? 100 - percentage : percentage
      }
      else if (line.startsWith('Binary files ') || line.startsWith('GIT binary patch')) {
        sawBinary = true
        projectedFile.binary = true
      }
      if (isAllowedPatchMetadata(line)) continue
      protocol('Unexpected patch metadata')
    }
    if (line === '\\ No newline at end of file') {
      if (!previousKind) protocol('Orphan no-newline marker')
      const value: GitPatchHunk['lines'][number] = {
        kind: 'no_newline',
        appliesTo: previousKind === 'delete' ? 'old' : 'new',
      }
      retain?.(value)
      current.lines.push(value)
      previousKind = undefined
      continue
    }
    const prefix = line[0]
    const body = line.slice(1)
    if (prefix === ' ') {
      const value: GitPatchHunk['lines'][number] = { kind: 'context', text: body, oldLine, newLine }
      retain?.(value)
      current.lines.push(value)
      oldLine += 1
      newLine += 1
      consumedOld += 1
      consumedNew += 1
      previousKind = 'context'
    } else if (prefix === '-') {
      const value: GitPatchHunk['lines'][number] = { kind: 'delete', text: body, oldLine, newLine: null }
      retain?.(value)
      current.lines.push(value)
      oldLine += 1
      consumedOld += 1
      previousKind = 'delete'
    } else if (prefix === '+') {
      const value: GitPatchHunk['lines'][number] = { kind: 'add', text: body, oldLine: null, newLine }
      retain?.(value)
      current.lines.push(value)
      newLine += 1
      consumedNew += 1
      previousKind = 'add'
    } else {
      protocol('Invalid patch hunk line')
    }
  }
  finishHunk()
  if (sawOldHeader !== sawNewHeader) protocol('Patch endpoint pair is incomplete')
  if (sawRenameFrom !== sawRenameTo) protocol('Rename endpoint pair is incomplete')
  if (sawOldMode !== sawNewMode) protocol('Mode metadata pair is incomplete')
  if (hunks.length > 0 && (!sawOldHeader || !sawNewHeader)) protocol('Text patch omitted old/new file headers')
  if (hunks.length === 0) {
    if (sawOldHeader || sawNewHeader) protocol('Patch endpoints were not followed by a text hunk')
    const unchangedIndex = indexMetadata === undefined || (
      hasFullObjectIds(indexMetadata) && !isZeroOid(indexMetadata.oldOid) &&
      indexMetadata.oldOid === indexMetadata.newOid &&
      (indexMetadata.mode === undefined || (
        !file.modeChanged && indexMetadata.mode === file.oldMode && indexMetadata.mode === file.newMode
      ))
    )
    const supportedMetadataOnly =
      sawBinary ||
      (
        file.action === 'modified' && file.modeChanged && sawOldMode && sawNewMode &&
        unchangedIndex
      ) ||
      (
        file.action === 'renamed' && sawRenameFrom && sawRenameTo &&
        similarityPercent === 100 && unchangedIndex
      ) ||
      (
        file.action === 'added' && sawNewFileMode && sawIndex &&
        indexMetadata !== undefined && isZeroOid(indexMetadata.oldOid) && isEmptyBlobOid(indexMetadata.newOid) &&
        (indexMetadata.mode === undefined || indexMetadata.mode === file.newMode)
      ) ||
      (
        file.action === 'deleted' && sawDeletedFileMode && sawIndex &&
        indexMetadata !== undefined && isEmptyBlobOid(indexMetadata.oldOid) && isZeroOid(indexMetadata.newOid) &&
        (indexMetadata.mode === undefined || indexMetadata.mode === file.oldMode)
      )
    if (!supportedMetadataOnly) protocol('Patch omitted content for a content-changing file')
  }
  projectedFile.contentChanged = sawBinary || hunks.length > 0
  return { layer, file: projectedFile, hunks }
}

// [maint-note:f9e39d432166]
// 维护导航：hasFullObjectIds；下列为本函数源码中的接口与路径索引。
// 输入 metadata：{ oldOid: string; newOid: string }。
// 返回契约：boolean。
// 返回路径：(metadata.oldOid.length === 40 || metadata.oldOid.length === 64) && metadata.newOid.length === metadata.oldOid.length。
// [/maint-note:f9e39d432166]
function hasFullObjectIds(metadata: { oldOid: string; newOid: string }): boolean {
  return (metadata.oldOid.length === 40 || metadata.oldOid.length === 64) &&
    metadata.newOid.length === metadata.oldOid.length
}

// [maint-note:d70f26b6c37b]
// 维护导航：isZeroOid；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：boolean。
// 返回路径：(value.length === 40 || value.length === 64) && /^0+$/u.test(value)。
// 直接调用：/^0+$/u.test。
// [/maint-note:d70f26b6c37b]
function isZeroOid(value: string): boolean {
  return (value.length === 40 || value.length === 64) && /^0+$/u.test(value)
}

// [maint-note:65a43c7097e9]
// 维护导航：isEmptyBlobOid；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：boolean。
// 分支条件：value.length !== 40 && value.length !== 64。
// 返回路径：false。
// 返回路径：createHash(algorithm).update('blob 0\0').digest('hex') === value。
// 直接调用：createHash(algorithm).update('blob 0\0').digest。
// 直接调用：createHash(algorithm).update。
// 直接调用：createHash。
// [/maint-note:65a43c7097e9]
function isEmptyBlobOid(value: string): boolean {
  if (value.length !== 40 && value.length !== 64) return false
  const algorithm = value.length === 40 ? 'sha1' : 'sha256'
  return createHash(algorithm).update('blob 0\0').digest('hex') === value
}

// [maint-note:a812ce63dbbb]
// 维护导航：iteratePatchLines；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 返回契约：Generator<string>。
// 分支条件：newline < 0。
// 遍历边界：offset < text.length。
// 返回路径：无返回值。
// 写入目标：offset。
// 直接调用：text.indexOf。
// 直接调用：text.slice。
// [/maint-note:a812ce63dbbb]
function* iteratePatchLines(text: string): Generator<string> {
  let offset = 0
  while (offset < text.length) {
    const newline = text.indexOf('\n', offset)
    if (newline < 0) {
      yield text.slice(offset)
      return
    }
    yield text.slice(offset, newline)
    offset = newline + 1
  }
}

// [maint-note:26bb8d8d18f6]
// 维护导航：assertPatchLineBound；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 返回契约：void。
// 分支条件：Buffer.byteLength(line, 'utf8') > MAX_GIT_DIFF_LINE_BYTES。
// 显式抛出：new GitChangesError('git_changes_observation_limit', 'Git patch line exceeds limit', 'diff_line_too_large')。
// 直接调用：Buffer.byteLength。
// [/maint-note:26bb8d8d18f6]
function assertPatchLineBound(line: string): void {
  if (Buffer.byteLength(line, 'utf8') > MAX_GIT_DIFF_LINE_BYTES) {
    throw new GitChangesError('git_changes_observation_limit', 'Git patch line exceeds limit', 'diff_line_too_large')
  }
}

// [maint-note:460ab4116053]
// 维护导航：parseDiffGitHeader；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 输入 toWorkspacePath：(path: string) => string | undefined。
// 输入 expectedOld：string。
// 输入 expectedNew：string。
// 返回契约：[string, string]。
// 分支条件：!left?.startsWith('a/') || !right?.startsWith('b/')。
// 分支条件：toWorkspacePath(oldPath) === expectedOld && toWorkspacePath(newPath) === expectedNew。
// 分支条件：!candidates.some(([old, next]) => old === oldPath && next === newPath)。
// 分支条件：candidates.length !== 1。
// 遍历边界：const index ← unquotedSeparatorOffsets(body)。
// 返回路径：candidates[0]!。
// 直接调用：line.slice。
// 直接调用：unquotedSeparatorOffsets。
// 直接调用：decodeHeaderPath。
// 直接调用：body.slice。
// 直接调用：left?.startsWith。
// 直接调用：right?.startsWith。
// 直接调用：left.slice。
// 直接调用：right.slice。
// 直接调用：toWorkspacePath。
// 直接调用：candidates.some。
// 直接调用：candidates.push。
// 直接调用：protocol。
// [/maint-note:460ab4116053]
function parseDiffGitHeader(
  line: string,
  toWorkspacePath: (path: string) => string | undefined,
  expectedOld: string,
  expectedNew: string,
): [string, string] {
  const body = line.slice('diff --git '.length)
  const candidates: Array<[string, string]> = []
  for (const index of unquotedSeparatorOffsets(body)) {
    const left = decodeHeaderPath(body.slice(0, index))
    const right = decodeHeaderPath(body.slice(index + 1))
    if (!left?.startsWith('a/') || !right?.startsWith('b/')) continue
    const oldPath = left.slice(2)
    const newPath = right.slice(2)
    if (toWorkspacePath(oldPath) === expectedOld && toWorkspacePath(newPath) === expectedNew) {
      if (!candidates.some(([old, next]) => old === oldPath && next === newPath)) {
        candidates.push([oldPath, newPath])
      }
    }
  }
  if (candidates.length !== 1) protocol('Invalid or ambiguous diff --git header')
  return candidates[0]!
}

// [maint-note:cfabcc13d04f]
// 维护导航：unquotedSeparatorOffsets；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number[]。
// 分支条件：quoted && escaped。
// 分支条件：quoted && character === '\\'。
// 分支条件：character === '"'。
// 分支条件：!quoted && character === ' ' && index > 0 && index < value.length - 1。
// 分支条件：quoted || escaped。
// 遍历边界：let index = 0; index < value.length; index += 1。
// 返回路径：result。
// 写入目标：index。
// 写入目标：escaped。
// 写入目标：quoted。
// 直接调用：result.push。
// 直接调用：protocol。
// [/maint-note:cfabcc13d04f]
function unquotedSeparatorOffsets(value: string): number[] {
  const result: number[] = []
  let quoted = false
  let escaped = false
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]!
    if (quoted && escaped) {
      escaped = false
      continue
    }
    if (quoted && character === '\\') {
      escaped = true
      continue
    }
    if (character === '"') {
      quoted = !quoted
      continue
    }
    if (!quoted && character === ' ' && index > 0 && index < value.length - 1) result.push(index)
  }
  if (quoted || escaped) protocol('Unterminated quoted Git path')
  return result
}

// [maint-note:7e750a9ee90d]
// 维护导航：validatePatchEndpoint；下列为本函数源码中的接口与路径索引。
// 输入 encoded：string。
// 输入 side：'old' | 'new'。
// 输入 file：GitFileRecord。
// 输入 toWorkspacePath：(repoPath: string) => string | undefined。
// 返回契约：void。
// 分支条件：token === '/dev/null'。
// 分支条件：(side === 'old' && file.action === 'added') || (side === 'new' && file.action === 'deleted')。
// 分支条件：!decoded?.startsWith(prefix)。
// 分支条件：toWorkspacePath(decoded.slice(2)) !== expected。
// 返回路径：无返回值。
// 直接调用：encoded.endsWith。
// 直接调用：encoded.slice。
// 直接调用：protocol。
// 直接调用：decodeHeaderPath。
// 直接调用：decoded?.startsWith。
// 直接调用：toWorkspacePath。
// 直接调用：decoded.slice。
// [/maint-note:7e750a9ee90d]
function validatePatchEndpoint(
  encoded: string,
  side: 'old' | 'new',
  file: GitFileRecord,
  toWorkspacePath: (repoPath: string) => string | undefined,
): void {
  // Git terminates an unquoted ---/+++ pathname with a tab. Pathnames that
  // contain a literal tab are C-quoted, so stripping this delimiter is
  // unambiguous.
  const token = encoded.endsWith('\t') ? encoded.slice(0, -1) : encoded
  if (token === '/dev/null') {
    if ((side === 'old' && file.action === 'added') || (side === 'new' && file.action === 'deleted')) return
    protocol('Unexpected /dev/null patch endpoint')
  }
  const decoded = decodeHeaderPath(token)
  const prefix = side === 'old' ? 'a/' : 'b/'
  if (!decoded?.startsWith(prefix)) protocol('Invalid old/new patch endpoint')
  const expected = side === 'old' ? (file.oldPath ?? file.path) : file.path
  if (toWorkspacePath(decoded.slice(2)) !== expected) protocol('Old/new patch endpoint does not match selected Git metadata')
}

// [maint-note:daa2e9831e63]
// 维护导航：validateRenameEndpoint；下列为本函数源码中的接口与路径索引。
// 输入 encoded：string。
// 输入 expected：string。
// 输入 toWorkspacePath：(repoPath: string) => string | undefined。
// 返回契约：void。
// 分支条件：!decoded || toWorkspacePath(decoded) !== expected。
// 直接调用：decodeHeaderPath。
// 直接调用：toWorkspacePath。
// 直接调用：protocol。
// [/maint-note:daa2e9831e63]
function validateRenameEndpoint(
  encoded: string,
  expected: string,
  toWorkspacePath: (repoPath: string) => string | undefined,
): void {
  const decoded = decodeHeaderPath(encoded)
  if (!decoded || toWorkspacePath(decoded) !== expected) protocol('Rename endpoint does not match selected Git metadata')
}

// [maint-note:597ae3414f84]
// 维护导航：decodeHeaderPath；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string | undefined。
// 分支条件：!value。
// 分支条件：!value.startsWith('"')。
// 返回路径：undefined。
// 返回路径：value.includes('"') ? undefined : value。
// 返回路径：parsed.length === 1 ? parsed[0] : undefined。
// 直接调用：value.startsWith。
// 直接调用：value.includes。
// 直接调用：parseQuotedTokens。
// [/maint-note:597ae3414f84]
function decodeHeaderPath(value: string): string | undefined {
  if (!value) return undefined
  if (!value.startsWith('"')) return value.includes('"') ? undefined : value
  const parsed = parseQuotedTokens(value)
  return parsed.length === 1 ? parsed[0] : undefined
}

// [maint-note:c74201511ef4]
// 维护导航：parseQuotedTokens；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string[]。
// 分支条件：offset >= value.length。
// 分支条件：value[offset] !== '"'。
// 分支条件：character === '"'。
// 分支条件：character !== '\\'。
// 分支条件：escaped === undefined。
// 分支条件：simple[escaped] !== undefined。
// 分支条件：/[0-7]/u.test(escaped)。
// 分支条件：!/^[0-7]{3}$/u.test(digits)。
// 分支条件：byte > 0xff。
// 分支条件：!closed。
// 遍历边界：offset < value.length。
// 遍历边界：value[offset] === ' '。
// 返回路径：result。
// 写入目标：offset。
// 写入目标：closed。
// 直接调用：value.indexOf。
// 直接调用：result.push。
// 直接调用：value.slice。
// 直接调用：bytes.push。
// 直接调用：Buffer.from。
// 直接调用：protocol。
// 直接调用：/[0-7]/u.test。
// 直接调用：/^[0-7]{3}$/u.test。
// 直接调用：Number.parseInt。
// 直接调用：decodePath。
// [/maint-note:c74201511ef4]
function parseQuotedTokens(value: string): string[] {
  const result: string[] = []
  let offset = 0
  while (offset < value.length) {
    while (value[offset] === ' ') offset += 1
    if (offset >= value.length) break
    if (value[offset] !== '"') {
      const end = value.indexOf(' ', offset)
      result.push(value.slice(offset, end < 0 ? value.length : end))
      offset = end < 0 ? value.length : end + 1
      continue
    }
    const bytes: number[] = []
    offset += 1
    let closed = false
    while (offset < value.length) {
      const character = value[offset++]!
      if (character === '"') { closed = true; break }
      if (character !== '\\') {
        bytes.push(...Buffer.from(character, 'utf8'))
        continue
      }
      const escaped = value[offset++]
      if (escaped === undefined) protocol('Invalid quoted Git path')
      const simple: Record<string, number> = { a: 7, b: 8, t: 9, n: 10, v: 11, f: 12, r: 13, '\\': 92, '"': 34 }
      if (simple[escaped] !== undefined) {
        bytes.push(simple[escaped]!)
        continue
      }
      if (/[0-7]/u.test(escaped)) {
        const digits = escaped + value.slice(offset, offset + 2)
        if (!/^[0-7]{3}$/u.test(digits)) protocol('Invalid octal Git path escape')
        const byte = Number.parseInt(digits, 8)
        if (byte > 0xff) protocol('Git path octal escape exceeds one byte')
        bytes.push(byte)
        offset += 2
        continue
      }
      protocol('Invalid Git path escape')
    }
    if (!closed) protocol('Unterminated quoted Git path')
    result.push(decodePath(Buffer.from(bytes)))
  }
  return result
}

// [maint-note:a032a75d729c]
// 维护导航：isAllowedPatchMetadata；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 返回契约：boolean。
// 返回路径：line.startsWith('index ') || line.startsWith('old mode ') || line.startsWith('new mode ') || line.startsWith('new file mode ') || line.startsWith('deleted file mode ') || line.startsWith('similarity index ') || line.startsWith('dissimilarit。
// 直接调用：line.startsWith。
// [/maint-note:a032a75d729c]
function isAllowedPatchMetadata(line: string): boolean {
  return line.startsWith('index ') || line.startsWith('old mode ') || line.startsWith('new mode ') ||
    line.startsWith('new file mode ') || line.startsWith('deleted file mode ') ||
    line.startsWith('similarity index ') || line.startsWith('dissimilarity index ') ||
    line.startsWith('rename from ') || line.startsWith('rename to ') ||
    line.startsWith('--- ') || line.startsWith('+++ ') ||
    line.startsWith('Binary files ') || line.startsWith('GIT binary patch') || line === ''
}

// [maint-note:8d5f7b2d08c6]
// 维护导航：splitNul；下列为本函数源码中的接口与路径索引。
// 输入 value：Buffer。
// 输入 maxRecords：类型由调用方与函数体约束；缺省表达式 Number.MAX_SAFE_INTEGER。
// 返回契约：Buffer[]。
// 分支条件：value.length > 0 && value[value.length - 1] !== 0。
// 分支条件：end < 0。
// 分支条件：result.length > maxRecords。
// 分支条件：offset === value.length。
// 遍历边界：; ; 。
// 返回路径：result。
// 写入目标：offset。
// 直接调用：protocol。
// 直接调用：value.indexOf。
// 直接调用：result.push。
// 直接调用：value.subarray。
// 直接调用：observationLimit。
// 直接调用：Buffer.alloc。
// [/maint-note:8d5f7b2d08c6]
function splitNul(value: Buffer, maxRecords = Number.MAX_SAFE_INTEGER): Buffer[] {
  if (value.length > 0 && value[value.length - 1] !== 0) protocol('NUL-delimited Git output is incomplete')
  const result: Buffer[] = []
  let offset = 0
  for (;;) {
    const end = value.indexOf(0, offset)
    if (end < 0) {
      result.push(value.subarray(offset))
      if (result.length > maxRecords) observationLimit('wire_record_count_exceeded')
      return result
    }
    result.push(value.subarray(offset, end))
    if (result.length > maxRecords) observationLimit('wire_record_count_exceeded')
    offset = end + 1
    if (offset === value.length) {
      result.push(Buffer.alloc(0))
      return result
    }
  }
}

// [maint-note:6cc4e91b5772]
// 维护导航：parseCount；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 分支条件：!/^(?:0|[1-9][0-9]*)$/u.test(value)。
// 分支条件：!Number.isSafeInteger(parsed)。
// 返回路径：parsed。
// 直接调用：/^(?:0|[1-9][0-9]*)$/u.test。
// 直接调用：protocol。
// 直接调用：Number。
// 直接调用：Number.isSafeInteger。
// [/maint-note:6cc4e91b5772]
function parseCount(value: string): number {
  if (!/^(?:0|[1-9][0-9]*)$/u.test(value)) protocol('Invalid numstat count')
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed)) protocol('Unsafe numstat count')
  return parsed
}

// [maint-note:cf07adcbee82]
// 维护导航：parsePatchNumber；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 分支条件：!/^(?:0|[1-9][0-9]*)$/u.test(value)。
// 分支条件：!Number.isSafeInteger(parsed)。
// 返回路径：parsed。
// 直接调用：/^(?:0|[1-9][0-9]*)$/u.test。
// 直接调用：protocol。
// 直接调用：Number。
// 直接调用：Number.isSafeInteger。
// [/maint-note:cf07adcbee82]
function parsePatchNumber(value: string): number {
  if (!/^(?:0|[1-9][0-9]*)$/u.test(value)) protocol('Invalid patch line number')
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed)) protocol('Unsafe patch line number')
  return parsed
}

// [maint-note:fb5610fcffea]
// 维护导航：objectType；下列为本函数源码中的接口与路径索引。
// 输入 mode：string。
// 返回契约：GitFileRecord['objectType']。
// 分支条件：mode === '120000'。
// 分支条件：mode === '160000'。
// 分支条件：mode.startsWith('100')。
// 返回路径：'symlink'。
// 返回路径：'gitlink'。
// 返回路径：'file'。
// 返回路径：'other'。
// 直接调用：mode.startsWith。
// [/maint-note:fb5610fcffea]
function objectType(mode: string): GitFileRecord['objectType'] {
  if (mode === '120000') return 'symlink'
  if (mode === '160000') return 'gitlink'
  if (mode.startsWith('100')) return 'file'
  return 'other'
}

// [maint-note:666e7324e471]
// 维护导航：protocol；下列为本函数源码中的接口与路径索引。
// 输入 message：string。
// 返回契约：never。
// 显式抛出：new GitChangesError('git_changes_patch_protocol_error', message, 'invalid_git_wire')。
// [/maint-note:666e7324e471]
function protocol(message: string): never {
  throw new GitChangesError('git_changes_patch_protocol_error', message, 'invalid_git_wire')
}

// [maint-note:c55bf6e643f7]
// 维护导航：observationLimit；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：never。
// 显式抛出：new GitChangesError('git_changes_observation_limit', 'Git observation exceeded a bounded limit', reason)。
// [/maint-note:c55bf6e643f7]
function observationLimit(reason: string): never {
  throw new GitChangesError('git_changes_observation_limit', 'Git observation exceeded a bounded limit', reason)
}

// [maint-note:66279d128670]
// 维护导航：maxParts；下列为本函数源码中的接口与路径索引。
// 输入 entries：number。
// 输入 multiplier：number。
// 输入 extra：类型由调用方与函数体约束；缺省表达式 0。
// 返回契约：number。
// 返回路径：Number.isSafeInteger(value) ? value : Number.MAX_SAFE_INTEGER。
// 直接调用：Number.isSafeInteger。
// [/maint-note:66279d128670]
function maxParts(entries: number, multiplier: number, extra = 0): number {
  const value = entries * multiplier + extra
  return Number.isSafeInteger(value) ? value : Number.MAX_SAFE_INTEGER
}

// [maint-note:797142449182]
// 维护导航：layerRecordCount；下列为本函数源码中的接口与路径索引。
// 输入 xy：string。
// 返回契约：number。
// 返回路径：[...xy].filter((code) => code !== '.').length。
// 直接调用：[...xy].filter。
// [/maint-note:797142449182]
function layerRecordCount(xy: string): number {
  return [...xy].filter((code) => code !== '.').length
}
