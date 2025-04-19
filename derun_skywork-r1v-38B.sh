#!/bin/bash
#SBATCH --job-name=skywork-R1v         # 任务名称
#SBATCH --output=/ceph/home/yaodong01/s1-m/pcwen/budget-forcing/derun_%j.log    # 输出日志文件
#SBATCH --error=/ceph/home/yaodong01/s1-m/pcwen/budget-forcing/derun_%j.log    # 输出日志文件
#SBATCH --partition=IAI_SLURM_HGX                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:8                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=220
#SBATCH --mem=500GB
#SBATCH --qos=12gpu-hgx 
#SBATCH --time=3:00:00

# 启动vllm服务器在后台运行
echo "启动vllm服务器..."
python -m vllm.entrypoints.openai.api_server --model /ceph/home/yaodong01/s1-m/Models/Skywork-R1V-38B/ \
                                                  --port 8000 \
                                                  --tensor-parallel-size 8 \
                                                  --gpu-memory-utilization 0.8 \
                                                  --served-model-name Skywork-R1V-38B \
                                                  --trust-remote-code \
                                                  --enable-prefix-caching \
                                                  --max_num_batched_tokens 131072 \
                                                  --max_num_seqs 2048 \
                                                  --scheduler-delay-factor 12 \
                                                  --limit-mm-per-prompt "image=1" \
                                                  --chat-template /ceph/home/yaodong01/s1-m/pcwen/budget-forcing/tempalte_skyt1_stts.jinja > vllm_server.log 2>&1 &

# 保存服务器进程ID
VLLM_PID=$!
echo "vllm服务器进程ID: $VLLM_PID"

# 等待20分钟让服务器完全启动
echo "等待5分钟..."
sleep 300  # 20分钟 = 1200秒

# 执行Python脚本
echo "开始执行run_mathv_R1V.py..."
cd /ceph/home/yaodong01/s1-m/pcwen/budget-forcing/  # 假设脚本在这个目录，根据实际情况修改
python run_mathv_R1V.py

# 检查Python脚本的退出状态
if [ $? -eq 0 ]; then
    echo "run_mathv_R1V.py执行成功"
else
    echo "run_mathv_R1V.py执行失败，退出代码: $?"
fi

# 停止vllm服务器
echo "停止vllm服务器..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null  # 等待进程结束，忽略可能的错误输出

echo "任务完成"