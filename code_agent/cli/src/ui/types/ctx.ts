import type { MessageType, Role } from '../types/message.js'

// [maint-note:ee067f498da3]
// 数据契约导航：Ctx；字段含义以消费端校验为准。
// 成员 addMsg：void。
// 成员 clearMessages：void | Promise<void>。
// 成员 execute：boolean。
// 成员 exit：void。
// [/maint-note:ee067f498da3]
export interface Ctx{
    // 消息操作
    addMsg(role: Role, context: string, status?: MessageType): void
    clearMessages(): void | Promise<void>

    // 执行
    execute(input: string): boolean

    // 进程
    exit(): void
}
