#!/bin/bash
set -e  # 出现错误时退出脚本

# --------------------------------------------------
# 第一步：启动 vllm 服务器（分配 100 CPU 模拟操作）
# --------------------------------------------------
echo "启动 vllm 服务器 (分配100 CPU 模拟操作) ..."
python -m vllm.entrypoints.openai.api_server \
     --model /ceph/home/yaodong01/s1-m/Models/Skywork-R1V2-38B/ \
     --port 8000 \
     --tensor-parallel-size 8 \
     --gpu-memory-utilization 0.8 \
     --served-model-name Skywork-R1V2-38B \
     --trust-remote-code \
     --enable-prefix-caching \
     --max_num_batched_tokens 131072 \
     --max_num_seqs 2048 \
     --scheduler-delay-factor 12 \
     --limit-mm-per-prompt "image=5" \
     --chat-template /ceph/home/yaodong01/s1-m/pcwen/s1m/tempalte_skyt1_stts.jinja \
     > vllm_server.log 2>&1 &
# 将后台进程PID保存到文件
echo $! > vllm_server.pid

# --------------------------------------------------
# 第二步：等待 vllm 服务器启动
# --------------------------------------------------
echo "等待 vllm 服务器启动（等待 7.5 分钟）..."
sleep 450

# --------------------------------------------------
# 第三步：执行 run_Oly_R1V2.py（分配50 CPU模拟操作）
# --------------------------------------------------
echo "切换到脚本目录并执行 run_Oly_R1V2.py (分配50 CPU模拟操作)..."
# 切换到对应的工作目录
cd /ceph/home/yaodong01/s1-m/pcwen/s1m/ || { echo "目录不存在，退出。"; exit 1; }
# 设置环境变量并执行脚本，记录日志
HF_ENDPOINT=https://hf-mirror.com python run_Oly_R1V2.py > run_Oly_R1V2.log 2>&1
RUN_STATUS=$?

if [ ${RUN_STATUS} -eq 0 ]; then
    echo "run_Oly_R1V2.py 执行成功"
else
    echo "run_Oly_R1V2.py 执行失败，退出代码: ${RUN_STATUS}"
fi

# --------------------------------------------------
# 第四步：停止 vllm 服务器
# --------------------------------------------------
echo "停止 vllm 服务器..."
if [ -f vllm_server.pid ]; then
    kill $(cat vllm_server.pid)
    rm -f vllm_server.pid
    echo "vllm 服务器已停止."
else
    echo "未检测到 vllm 服务器 PID 文件."
fi

# 等待所有后台任务结束（如有）
wait

echo "全部任务完成."