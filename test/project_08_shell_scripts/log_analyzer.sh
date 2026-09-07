#!/bin/bash
###############################################################################
# 项目: project_08_shell_scripts
# 测试主题: Shell 自动化脚本 - 日志分析工具
# 功能: 分析 Nginx/应用日志，统计访问量、错误率、慢请求等指标
###############################################################################

set -euo pipefail

LOG_FILE="${1:-/var/log/nginx/access.log}"
TOP_N="${TOP_N:-10}"

if [ ! -f "${LOG_FILE}" ]; then
    echo "错误: 日志文件不存在: ${LOG_FILE}" >&2
    exit 1
fi

echo "=========================================="
echo "日志分析报告"
echo "文件: ${LOG_FILE}"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 总行数
total=$(wc -l < "${LOG_FILE}")
echo -e "\n📊 总请求数: ${total}"

# 状态码分布
echo -e "\n📈 HTTP 状态码分布:"
awk '{print $9}' "${LOG_FILE}" 2>/dev/null | sort | uniq -c | sort -rn | head -10

# 错误率
errors=$(awk '$9 >= 400 {count++} END {print count+0}' "${LOG_FILE}")
if [ "${total}" -gt 0 ]; then
    error_rate=$(awk "BEGIN {printf \"%.2f\", ${errors}/${total}*100}")
    echo -e "\n❌ 错误率: ${error_rate}% (${errors}/${total})"
fi

# TOP IP
echo -e "\n🌐 TOP ${TOP_N} IP 地址:"
awk '{print $1}' "${LOG_FILE}" | sort | uniq -c | sort -rn | head -"${TOP_N}"

# TOP URL
echo -e "\n🔗 TOP ${TOP_N} 访问 URL:"
awk '{print $7}' "${LOG_FILE}" | sort | uniq -c | sort -rn | head -"${TOP_N}"

# TOP User-Agent
echo -e "\n🌍 TOP ${TOP_N} User-Agent:"
awk -F'"' '{print $6}' "${LOG_FILE}" | sort | uniq -c | sort -rn | head -"${TOP_N}"

# 时间分布（按小时）
echo -e "\n⏰ 请求按小时分布:"
awk '{print substr($4, 14, 2)}' "${LOG_FILE}" | sort | uniq -c

echo -e "\n=========================================="
echo "分析完成"
echo "=========================================="
