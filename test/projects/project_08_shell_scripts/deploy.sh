#!/bin/bash
###############################################################################
# 项目: project_08_shell_scripts
# 测试主题: Shell 自动化脚本 - 部署与运维
# 功能: 应用部署脚本 - 支持环境配置、构建、健康检查、回滚
###############################################################################

set -euo pipefail

# 颜色输出
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 配置
readonly APP_NAME="${APP_NAME:-test-app}"
readonly ENV="${ENV:-dev}"
readonly VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'local')}"
readonly DEPLOY_DIR="/opt/${APP_NAME}"
readonly BACKUP_DIR="/var/backups/${APP_NAME}"

usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  -e, --env ENV       部署环境 (dev/staging/prod), 默认: dev
  -v, --version VER   指定版本号, 默认: 当前 git commit
  -r, --rollback      回滚到上一个版本
  -h, --help          显示帮助

环境变量:
  APP_NAME            应用名称
  REMOTE_HOST         远程部署主机

示例:
  $0 -e prod -v v1.2.3
  $0 --rollback
EOF
}

rollback() {
    log_warn "开始回滚..."
    if [ ! -d "${BACKUP_DIR}" ]; then
        log_error "备份目录不存在: ${BACKUP_DIR}"
        exit 1
    fi
    local latest_backup
    latest_backup=$(ls -t "${BACKUP_DIR}" | head -n1)
    if [ -z "${latest_backup}" ]; then
        log_error "未找到备份"
        exit 1
    fi
    log_info "恢复到: ${latest_backup}"
    rsync -a --delete "${BACKUP_DIR}/${latest_backup}/" "${DEPLOY_DIR}/"
    log_info "回滚完成"
}

deploy() {
    log_info "开始部署 ${APP_NAME}@${VERSION} 到 ${ENV} 环境"

    # 1. 健康检查当前版本
    if [ -x "${DEPLOY_DIR}/healthcheck.sh" ]; then
        "${DEPLOY_DIR}/healthcheck.sh" || log_warn "当前版本健康检查失败"
    fi

    # 2. 备份当前版本
    if [ -d "${DEPLOY_DIR}" ]; then
        log_info "备份当前版本..."
        mkdir -p "${BACKUP_DIR}"
        local backup_name="backup-$(date +%Y%m%d-%H%M%S)"
        cp -a "${DEPLOY_DIR}" "${BACKUP_DIR}/${backup_name}"
        log_info "已备份到 ${BACKUP_DIR}/${backup_name}"
    fi

    # 3. 部署新版本
    log_info "复制新版本文件..."
    mkdir -p "${DEPLOY_DIR}"
    cp -a ./build/. "${DEPLOY_DIR}/"

    # 4. 重启服务
    log_info "重启服务..."
    if command -v systemctl >/dev/null 2>&1; then
        sudo systemctl restart "${APP_NAME}.service" || log_warn "systemctl 重启失败"
    fi

    # 5. 健康检查
    log_info "健康检查新版本..."
    sleep 3
    if [ -x "${DEPLOY_DIR}/healthcheck.sh" ]; then
        if "${DEPLOY_DIR}/healthcheck.sh"; then
            log_info "部署成功 ✓"
        else
            log_error "健康检查失败，触发回滚"
            rollback
            exit 1
        fi
    fi
}

main() {
    local do_rollback=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--env) ENV="$2"; shift 2 ;;
            -v|--version) VERSION="$2"; shift 2 ;;
            -r|--rollback) do_rollback=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) log_error "未知参数: $1"; usage; exit 1 ;;
        esac
    done

    if [ "${do_rollback}" = true ]; then
        rollback
    else
        deploy
    fi
}

main "$@"
