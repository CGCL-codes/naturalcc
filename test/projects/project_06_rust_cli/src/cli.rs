// 项目: project_06_rust_cli
// 测试主题: Rust 命令行工具开发
// 功能: CLI 参数定义

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "test_cli")]
#[command(about = "用于测试 coding agent 的 CLI 工具")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// 向某人打招呼
    Greet {
        /// 要打招呼的名字
        name: String,
        /// 重复次数
        #[arg(short, long, default_value_t = 1)]
        times: u32,
    },
    /// 统计文件行数
    Count {
        /// 目标文件路径
        file: String,
    },
    /// 格式化 JSON
    Json {
        /// 输入 JSON 字符串（也可用 - 从 stdin 读取）
        #[arg(short, long)]
        input: String,
        /// 美化输出
        #[arg(long)]
        pretty: bool,
    },
}
