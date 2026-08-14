import type { MessageType, Role } from '../types/message.js'

export interface Ctx{
    // 消息操作
    addMsg(role: Role, context: string, status?: MessageType): void
    clearMessages(): void | Promise<void>

    // 执行
    execute(input: string): boolean

    // 进程
    exit(): void
}
