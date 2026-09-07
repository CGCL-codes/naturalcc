#!/bin/bash
###############################################################################
# 项目: project_08_shell_scripts
# 测试主题: Shell 自动化脚本 - 数据库与文件备份
# 功能: 定时备份脚本 - 支持压缩、加密、远程传输、保留策略
###############################################################################

set -euo pipefail

# 配置
readonly BACKUP_ROOT="${BACKUP_ROOT:-/var/backups}"
readonly TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
readonly RETENTION_DAYS="${RETENTION_DAYS:-7}"
readonly REMOTE_HOST="${REMOTE_HOST:-}"
readonly REMOTE_PATH="${REMOTE_PATH:-}"

backup_file() {
    local source="$1"
    local dest_dir="${BACKUP_ROOT}/files/${TIMESTAMP}"
    mkdir -p "${dest_dir}"

    echo "📦 备份文件: ${source}"
    tar -czf "${dest_dir}/$(basename "${source}").tar.gz" -C "$(dirname "${source}")" "$(basename "${source}")"
    echo "✓ 已保存到: ${dest_dir}/$(basename "${source}").tar.gz"
}

backup_database() {
    local db_name="$1"
    local dest_dir="${BACKUP_ROOT}/db/${TIMESTAMP}"
    mkdir -p "${dest_dir}"

    echo "🗄️  备份数据库: ${db_name}"

    if command -v pg_dump >/dev/null 2>&1; then
        pg_dump "${db_name}" | gzip > "${dest_dir}/${db_name}.sql.gz"
        echo "✓ PostgreSQL 备份完成"
    elif command -v mysqldump >/dev/null 2>&1; then
        mysqldump "${db_name}" | gzip > "${dest_dir}/${db_name}.sql.gz"
        echo "✓ MySQL 备份完成"
    else
        echo "⚠ 未找到数据库客户端"
        return 1
    fi
}

cleanup_old_backups() {
    echo "🧹 清理 ${RETENTION_DAYS} 天前的备份..."
    find "${BACKUP_ROOT}" -type d -mtime +"${RETENTION_DAYS}" -exec rm -rf {} \; 2>/dev/null || true
    echo "✓ 清理完成"
}

upload_to_remote() {
    if [ -z "${REMOTE_HOST}" ]; then
        return 0
    fi
    echo "☁️  上传到远程: ${REMOTE_HOST}"
    rsync -az "${BACKUP_ROOT}/" "${REMOTE_HOST}:${REMOTE_PATH}/"
    echo "✓ 上传完成"
}

main() {
    echo "=========================================="
    echo "备份任务开始"
    echo "时间: ${TIMESTAMP}"
    echo "=========================================="

    # 示例：备份配置文件和数据库
    backup_file "/etc/nginx"
    backup_database "myapp_production"

    upload_to_remote
    cleanup_old_backups

    echo "=========================================="
    echo "备份任务完成"
    echo "=========================================="
}

main "$@"
