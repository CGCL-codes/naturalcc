export class ModelProtocolError extends Error {
  // [maint-note:284e70a8df18]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 message：string。
  // 写入目标：this.name。
  // 直接调用：super。
  // [/maint-note:284e70a8df18]
  constructor(message: string) {
    super(message)
    this.name = 'ModelProtocolError'
  }
}

export class ModelResponseLimitError extends Error {
  // [maint-note:e02630ae2cdb]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 message：string。
  // 写入目标：this.name。
  // 直接调用：super。
  // [/maint-note:e02630ae2cdb]
  constructor(message: string) {
    super(message)
    this.name = 'ModelResponseLimitError'
  }
}
