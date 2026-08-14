export function help_info(): string {
  return [
    'commands:',
    '/help  显示帮助信息',
    '/clear 清除 UI 与 agent 对话历史',
    '/exit  退出 REPL',
    '',
    'CLI options:',
    '--model <name>       使用指定模型',
    '--api-key <key>      使用指定 API Key（默认读取 OPENAI_API_KEY）',
    '--base-url <url>     使用指定 OpenAI-compatible endpoint',
    '--project-dir <dir> 使用指定项目目录',
    '--max-turns <n>      设置单回合最大循环次数',
    '--no-tool-results    隐藏正常工具输出',
    '--approval <mode>   审批模式：prompt、deny 或 allow',
    '--yes               允许需要审批的工具（等同 allow）',
    'prompt 审批显示真实 command/patch 与模型 Reason，默认选 No；用 ↑/↓ 和 Enter 选择，Esc 中断；one-shot 使用 [y/N]。',
    '结构化文件工具限制在项目目录；list/search 依赖 PATH 中的 rg，search_text 仅支持字面搜索；bash 不是沙箱。',
    '',
    'Esc  中断当前回合；Ctrl-C 连按两次退出。',
    '文件路径请直接作为提示词或结构化工具参数提供。',
  ].join('\n')
}
