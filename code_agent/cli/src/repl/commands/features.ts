import { spawnSync } from 'node:child_process'
import { codeAgentDir, pythonPrelude } from '../../pythonPath.js'

interface FeatureInfo {
  name: string
  label: string
  description: string
  execution_mode: string
  config_schema?: ConfigField[]
  error?: string
}

interface ConfigOption {
  value: string
  label?: string
}

interface ConfigField {
  name: string
  label: string
  type: string
  required: boolean
  default?: unknown
  placeholder?: string
  help_text?: string
  options?: ConfigOption[] | null
  accept?: string | null
  multiple?: boolean
}

function runFeatureQuery(expression: string, payload: Record<string, unknown> = {}): unknown {
  const script = [
    pythonPrelude,
    'from feature_runner import list_features, describe_feature',
    'payload = json.loads(sys.stdin.read() or "{}")',
    `print(json.dumps(${expression}, ensure_ascii=False))`,
  ].join('\n')

  const result = spawnSync('python3', ['-c', script], {
    input: JSON.stringify(payload),
    encoding: 'utf-8',
    cwd: codeAgentDir,
  })

  if (result.error) {
    throw new Error(result.error.message)
  }
  if (result.status !== 0) {
    const stderr = result.stderr.trimEnd()
    throw new Error(stderr || `Python process exited with code ${result.status ?? 1}`)
  }

  const jsonLine = result.stdout
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .at(-1)
  if (!jsonLine) {
    throw new Error('Python feature query returned no output')
  }

  try {
    return JSON.parse(jsonLine)
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    throw new Error(`invalid feature query output: ${message}`)
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function boolValue(value: unknown): boolean {
  return value === true
}

function toFeatureInfo(value: unknown): FeatureInfo {
  const record = asRecord(value)
  return {
    name: stringValue(record.name),
    label: stringValue(record.label),
    description: stringValue(record.description),
    execution_mode: stringValue(record.execution_mode),
    config_schema: Array.isArray(record.config_schema)
      ? record.config_schema.map(toConfigField)
      : undefined,
    error: stringValue(record.error) || undefined,
  }
}

function toConfigField(value: unknown): ConfigField {
  const record = asRecord(value)
  return {
    name: stringValue(record.name),
    label: stringValue(record.label),
    type: stringValue(record.type),
    required: boolValue(record.required),
    default: record.default,
    placeholder: stringValue(record.placeholder) || undefined,
    help_text: stringValue(record.help_text) || undefined,
    options: Array.isArray(record.options)
      ? record.options.map((item) => {
          const option = asRecord(item)
          return {
            value: stringValue(option.value),
            label: stringValue(option.label) || undefined,
          }
        })
      : null,
    accept: stringValue(record.accept) || null,
    multiple: boolValue(record.multiple),
  }
}

function formatDefault(value: unknown): string {
  if (value === undefined || value === null) return '-'
  if (typeof value === 'string') return value || '""'
  return JSON.stringify(value)
}

function formatOptions(options: ConfigOption[] | null | undefined): string {
  if (!options || options.length === 0) return ''
  return options
    .map((option) => {
      if (!option.label || option.label === option.value) return option.value
      return `${option.value} (${option.label})`
    })
    .join(', ')
}

function buildDefaultConfig(fields: ConfigField[]): Record<string, unknown> {
  const config: Record<string, unknown> = {}
  for (const field of fields) {
    if (field.default !== undefined && field.default !== null) {
      config[field.name] = field.default
    }
  }
  return config
}

function featureQueryError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err)
  return `Feature query failed: ${message}`
}

export function listFeaturesText(): string {
  try {
    const result = runFeatureQuery('list_features()')
    if (!Array.isArray(result)) {
      return 'Feature query failed: expected feature list'
    }

    const features = result.map(toFeatureInfo).filter((feature) => feature.name)
    if (features.length === 0) {
      return 'No features registered.'
    }

    const nameWidth = Math.max('Feature'.length, ...features.map((feature) => feature.name.length))
    const modeWidth = Math.max('Mode'.length, ...features.map((feature) => feature.execution_mode.length))
    const lines = [
      'Available features:',
      '',
      `${'Feature'.padEnd(nameWidth)}  ${'Mode'.padEnd(modeWidth)}  Label`,
      `${'-'.repeat(nameWidth)}  ${'-'.repeat(modeWidth)}  -----`,
    ]

    for (const feature of features) {
      lines.push(`${feature.name.padEnd(nameWidth)}  ${feature.execution_mode.padEnd(modeWidth)}  ${feature.label}`)
    }
    lines.push('', 'Use /feature-schema <feature> to inspect config fields.')
    return lines.join('\n')
  } catch (err) {
    return featureQueryError(err)
  }
}

export function featureSchemaText(rawFeature: string): string {
  const feature = rawFeature.trim().split(/\s+/)[0] || ''
  if (!feature) {
    return 'Usage: /feature-schema <feature>'
  }

  try {
    const info = toFeatureInfo(runFeatureQuery('describe_feature(payload.get("feature"))', { feature }))
    if (info.error) {
      return info.error
    }

    const fields = info.config_schema ?? []
    const lines = [
      `Feature: ${info.name}`,
      `Label: ${info.label}`,
      `Mode: ${info.execution_mode}`,
      `Description: ${info.description}`,
      '',
      'Config:',
    ]

    if (fields.length === 0) {
      lines.push('- no config fields')
    } else {
      for (const field of fields) {
        const requirement = field.required ? 'required' : 'optional'
        lines.push(`- ${field.name} (${field.type}, ${requirement}, default: ${formatDefault(field.default)})`)
        if (field.options?.length) {
          lines.push(`  options: ${formatOptions(field.options)}`)
        }
        if (field.accept) {
          lines.push(`  accepts: ${field.accept}${field.multiple ? ', multiple' : ''}`)
        }
        if (field.help_text) {
          lines.push(`  help: ${field.help_text}`)
        }
      }
    }

    const defaultConfig = buildDefaultConfig(fields)
    if (Object.keys(defaultConfig).length > 0) {
      lines.push('', `Default --feature-config: ${JSON.stringify(defaultConfig)}`)
    }

    return lines.join('\n')
  } catch (err) {
    return featureQueryError(err)
  }
}
