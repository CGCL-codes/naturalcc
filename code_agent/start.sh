#!/bin/bash

# 一键启动 NaturalCC Agent Web UI
# 终端 1: 启动 Python API 服务
# 终端 2: 启动前端开发服务器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate naturalcc

# 检测可用的终端模拟器
find_terminal() {
    for term in gnome-terminal konsole xfce4-terminal mate-terminal terminator alacritty wezterm kitty; do
        if command -v "$term" &> /dev/null; then
            echo "$term"
            return
        fi
    done
    echo ""
}

TERMINAL=$(find_terminal)

if [ -z "$TERMINAL" ]; then
    echo "未找到图形终端模拟器，回退到 tmux 分屏模式..."
    if ! command -v tmux &> /dev/null; then
        echo "错误: tmux 也未安装"
        echo "请安装以下任意一种: gnome-terminal, konsole, xfce4-terminal, mate-terminal, terminator, alacritty, wezterm, kitty, tmux"
        exit 1
    fi

    SESSION_NAME="ncc-agent"
    # 如果 session 已存在则先杀掉
    tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"

    # 创建新 session，默认运行 API
    tmux new-session -d -s "$SESSION_NAME" -n "agent"
    tmux send-keys -t "$SESSION_NAME" "cd '$SCRIPT_DIR' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && echo '启动 API 服务...' && python agent_web_api.py --host 127.0.0.1 --port 7860" C-m

    # 水平分割，下方运行前端
    tmux split-window -v -t "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "cd '$SCRIPT_DIR/webui' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && echo '启动前端开发服务器...' && npm run dev" C-m

    # 调整上下比例（上面大一点）
    tmux resize-pane -t "$SESSION_NAME" -U 5

    # 附加到 session
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

echo "使用终端: $TERMINAL"

# 根据终端类型执行不同的命令
case "$TERMINAL" in
    gnome-terminal|mate-terminal)
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && echo '启动 API 服务...' && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash"
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR/webui' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && echo '启动前端开发服务器...' && npm run dev; exec bash"
        ;;
    konsole)
        konsole --new-tab -e bash -c "cd '$SCRIPT_DIR' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash" &
        konsole --new-tab -e bash -c "cd '$SCRIPT_DIR/webui' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && npm run dev; exec bash" &
        ;;
    xfce4-terminal)
        xfce4-terminal --command="bash -c 'cd \"$SCRIPT_DIR\" && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate naturalcc && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash'"
        xfce4-terminal --command="bash -c 'cd \"$SCRIPT_DIR/webui\" && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate naturalcc && npm run dev; exec bash'"
        ;;
    terminator)
        terminator -e "bash -c 'cd \"$SCRIPT_DIR\" && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate naturalcc && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash'" &
        terminator -e "bash -c 'cd \"$SCRIPT_DIR/webui\" && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate naturalcc && npm run dev; exec bash'" &
        ;;
    alacritty)
        alacritty -e bash -c "cd '$SCRIPT_DIR' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash" &
        alacritty -e bash -c "cd '$SCRIPT_DIR/webui' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && npm run dev; exec bash" &
        ;;
    wezterm)
        wezterm start -- bash -c "cd '$SCRIPT_DIR' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash"
        wezterm start -- bash -c "cd '$SCRIPT_DIR/webui' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && npm run dev; exec bash"
        ;;
    kitty)
        kitty bash -c "cd '$SCRIPT_DIR' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && python agent_web_api.py --host 127.0.0.1 --port 7860; exec bash" &
        kitty bash -c "cd '$SCRIPT_DIR/webui' && source '$(conda info --base)/etc/profile.d/conda.sh' && conda activate naturalcc && npm run dev; exec bash" &
        ;;
esac

echo "服务已启动!"
echo "  API 服务:   http://127.0.0.1:7860"
echo "  前端服务:   请查看终端 2 的输出中的实际地址 (通常是 http://localhost:5173)"
