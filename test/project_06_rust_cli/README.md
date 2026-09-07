# 测试主题：Rust 命令行工具开发

这是一个用于测试 coding agent Rust 编程能力的项目。

## 测试目标
- 测试 agent 是否能正确处理 Cargo 项目结构和依赖
- 测试 agent 是否能正确实现 CLI 参数解析（clap）
- 测试 agent 是否能正确处理文件 I/O 和错误传播
- 测试 agent 是否能正确实现单元测试

## 项目结构
```
project_06_rust_cli/
├── README.md
├── Cargo.toml          # 项目配置
├── src/
│   ├── main.rs         # 入口
│   ├── lib.rs          # 库入口
│   ├── cli.rs          # CLI 参数定义
│   └── commands.rs     # 命令实现
└── tests/
    └── integration.rs  # 集成测试
```
