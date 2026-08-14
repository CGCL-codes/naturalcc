export class ModelProtocolError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ModelProtocolError'
  }
}

export class ModelResponseLimitError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ModelResponseLimitError'
  }
}
