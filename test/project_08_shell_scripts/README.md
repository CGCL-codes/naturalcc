# 测试主题：Shell 自动化脚本与 DevOps

这是一个用于测试 coding agent Shell 脚本编写能力的项目。

## 测试目标
- 测试 agent 是否能正确处理 Bash 语法（条件、循环、函数）
- 测试 agent 是否能正确处理命令行参数解析
- 测试 agent 是否能正确实现错误处理（set -e、trap）
- 测试 agent 是否能正确处理文件 I/O 和文本处理（grep/awk/sed）

## 项目结构
```
project_08_shell_scripts/
├── README.md
├── deploy.sh           # 应用部署脚本
├── log_analyzer.sh     # 日志分析工具
├── backup.sh           # 备份脚本
└── lib/
    └── common.sh       # 公共函数库
```
