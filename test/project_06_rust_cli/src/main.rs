// 项目: project_06_rust_cli
// 测试主题: Rust 命令行工具开发
// 功能: CLI 工具入口

use anyhow::Result;
use clap::Parser;

mod cli;
mod commands;

use cli::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Greet { name, times } => commands::greet(&name, times),
        Commands::Count { file } => commands::count_lines(&file),
        Commands::Json { input, pretty } => commands::format_json(&input, pretty),
    }
}
