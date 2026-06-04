#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
conda activate trans_env
python src/hunyuan_api.py

export CUDA_VISIBLE_DEVICES=1
conda activate main_env
python src/qwen25_api.py

export CUDA_VISIBLE_DEVICES=0
conda activate main_env
python src/api_main.py

# tmux
# 创建一个名为 video_services 的会话，第一个窗口命名为 hunyuan
tmux new -s video_services -n hunyuan
# 进入 tmux 窗口，执行：
export CUDA_VISIBLE_DEVICES=1
conda activate trans_env
python src/hunyuan_api.py

# 按快捷键创建新窗口：
# 先按 Ctrl+b，松开后再按 c
Ctrl+b c
# 新窗口会自动打开，执行：
export CUDA_VISIBLE_DEVICES=1
conda activate main_env
python src/qwen25_api.py

# 再次按快捷键创建新窗口：
Ctrl+b c
# 新窗口执行：
export CUDA_VISIBLE_DEVICES=0
conda activate main_env
python src/api_main.py

# 查看日志
# 切换到窗口 0 (hunyuan)：
Ctrl+b 0
# 切换到窗口 1 (qwen25)：
Ctrl+b 1
# 切换到窗口 2 (api_main)：
Ctrl+b 2

# 切换到下一个窗口：
Ctrl+b n   # next window
# 切换到上一个窗口：
Ctrl+b p   # previous window

# 退出但后台运行
# 按快捷键退出（detach）：
Ctrl+b d
# 会回到普通终端，但所有服务仍在后台运行

# 重新连接到 tmux
# 列出所有 tmux 会话
tmux ls
# 重新连接到 video_services 会话
tmux attach -t video_services
# 或者简写：
tmux a -t video_services

# 停止服务
# 普通终端执行：
tmux kill-session -t video_services

tmux new -s download
python3 -m http.server 6077

tmux new -s video_task