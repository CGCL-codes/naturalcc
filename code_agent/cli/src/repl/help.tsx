export function help_info() : string{
    return  "command:\n"+
            "/help 显示帮助信息\n"+
            "/exit 退出 REPL\n"+
            "/clear 清除对话历史\n"+
            "/settings 显示/隐藏当前设置面板\n"+
            "\n"+
            "instruction:\n"+
            "-f --file [files...], 设置目标文件列表，如 src/main.c src/utils.c\n"+
            "-m, --model <model>, 设置模型, 默认 openrouter/deepseek/deepseek-chat\n"+
            "-k, --apiKey <apiKey>, 设置 API Key(默认读环境变量)\n"+
            "-d, --projectDir <dir>, 设置项目根目录，默认使用当前运行程序的目录\n"+
            "-s, --symbol <symbol>, 设置目标符号(可选)\n"+
            "-t, --completionType <type>, 设置补全类型(可选)\n"+
            "\t member, variable, function, function_body, type\n"+
            "--prefix <prefix>, 设置补全前缀\n"+
            "--preview 切换预览模式开关,预览模式仅预览最终 Prompt ,不执行 Aider, 默认不开启\n"+
            "--run ,以当前设置重新执行上一次的instruction"
}