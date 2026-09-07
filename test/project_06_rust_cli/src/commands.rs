// 项目: project_06_rust_cli
// 测试主题: Rust 命令行工具开发
// 功能: 命令实现

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

pub fn greet(name: &str, times: u32) -> Result<()> {
    for i in 1..=times {
        println!("[{}/{}] 你好, {}!", i, times, name);
    }
    Ok(())
}

pub fn count_lines(file: &str) -> Result<()> {
    let path = Path::new(file);
    if !path.exists() {
        anyhow::bail!("文件不存在: {}", file);
    }
    let content = fs::read_to_string(path)
        .with_context(|| format!("无法读取文件: {}", file))?;
    let line_count = content.lines().count();
    let char_count = content.chars().count();
    println!("文件: {}", file);
    println!("行数: {}", line_count);
    println!("字符数: {}", char_count);
    Ok(())
}

pub fn format_json(input: &str, pretty: bool) -> Result<()> {
    let data: Value = if input == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        serde_json::from_str(&buf)?
    } else {
        serde_json::from_str(input)?
    };
    let output = if pretty {
        serde_json::to_string_pretty(&data)?
    } else {
        serde_json::to_string(&data)?
    };
    println!("{}", output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet_runs() {
        // 简单的烟雾测试
        assert!(greet("test", 1).is_ok());
    }

    #[test]
    fn test_count_lines_missing_file() {
        assert!(count_lines("/nonexistent/path/file.txt").is_err());
    }

    #[test]
    fn test_format_json_valid() {
        assert!(format_json(r#"{"key": "value"}"#, true).is_ok());
    }

    #[test]
    fn test_format_json_invalid() {
        assert!(format_json("not valid json", false).is_err());
    }
}
