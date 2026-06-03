import type {Role} from './message.js'

export interface Ctx{
    // 消息操作
    addMsg(role : Role,context : string): void
    clearMessages(): void
    error(msg : string): void

    // 设置 — 读取
    files: string[]
    model: string
    apiKey: string | null
    projectDir: string
    symbol: string | null
    completionType: string | null                                                                                                                                                                                       
    prefix: string                                                                                                                                                                                                      
    preview: boolean

    // 设置 — 写入
    setFiles(files: string[]): void 
    setModel(model: string): void
    setApiKey(key: string | null): void
    setProjectDir(dir: string): void
    setSymbol(symbol: string | null): void
    setCompletionType(type: string | null): void
    setPrefix(prefix: string): void
    togglePreview(): void
    toggleSettings(): void
    resetSettings(): void

    // 执行
    execute(input: string): void
    rerun(): void

    // 进程
    exit(): void
}