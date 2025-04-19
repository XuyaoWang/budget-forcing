#!/bin/bash
#SBATCH --job-name=skywork-R1v         # 任务名称
#SBATCH --output=/ceph/home/yaodong01/s1-m/pcwen/budget-forcing/derun_%j.log    # 输出日志文件
#SBATCH --error=/ceph/home/yaodong01/s1-m/pcwen/budget-forcing/derun_%j.log    # 输出日志文件
#SBATCH --partition=IAI_SLURM_HGX                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:4                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=220
#SBATCH --mem=500GB
#SBATCH --qos=12gpu-hgx 
#SBATCH --time=3:00:00

# 启动vllm服务器在后台运行
echo "启动vllm服务器..."
python -m vllm.entrypoints.openai.api_server --model /ceph/home/yaodong01/s1-m/Models/R1-Onevision-7B-RL \
                                                    --port 9000 \
                                                    --tensor-parallel-size 4 \
                                                    --gpu-memory-utilization 0.6 \
                                                    --served-model-name r1-ov-7b \
                                                    --trust-remote-code \
                                                    --enable-prefix-caching \
                                                    --max_num_batched_tokens  128000 \
                                                    --max_num_seqs  2048 \
                                                    --scheduler-delay-factor 12 \
                                                    --limit-mm-per-prompt "image=5" \
                                                    --chat-template /ceph/home/yaodong01/s1-m/pcwen/budget-forcing/qvq_test-scaling.jinja > R1-Onevision-7B-RL——vllm_server.log 2>&1 &

# 保存服务器进程ID
VLLM_PID=$!
echo "vllm服务器进程ID: $VLLM_PID"

# 等待20分钟让服务器完全启动
echo "等待5分钟..."
sleep 300  # 20分钟 = 1200秒

# 执行Python脚本
echo "开始执行run_mathv_R1V.py..."
cd /ceph/home/yaodong01/s1-m/pcwen/budget-forcing/  # 假设脚本在这个目录，根据实际情况修改
HF_ENDPOINT=https://hf-mirror.com  python run_Oly_R1ov7b.py

# 检查Python脚本的退出状态
if [ $? -eq 0 ]; then
    echo "run_Oly_R1ov7b.py执行成功"
else
    echo "run_Oly_R1ov7b.py执行失败，退出代码: $?"
fi

# 停止vllm服务器
echo "停止vllm服务器..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null  # 等待进程结束，忽略可能的错误输出

echo "任务完成"