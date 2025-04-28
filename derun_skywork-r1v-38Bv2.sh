#!/bin/bash
#SBATCH --job-name=skyr1v2         # 任务名称
#SBATCH --output=/ceph/home/yaodong01/s1-m/pcwen/s1m/derun_%j.log    # 输出日志文件
#SBATCH --error=/ceph/home/yaodong01/s1-m/pcwen/s1m/derun_%j.log    # 输出日志文件
#SBATCH --partition=IAI_SLURM_HGX                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:8                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=50
#SBATCH --ntasks-per-node=2
#SBATCH --mem=500GB
#SBATCH --qos=12gpu-hgx 
#SBATCH --time=3:00:00

# 显示分配到的节点列表
echo "分配到的节点列表："
scontrol show hostnames $SLURM_NODELIST
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
echo "Head 节点：${HEAD_NODE}"

# 第一步：在 head 节点上启动 vllm 服务器，并将后台进程 ID 写入文件
echo "在 Head 节点上启动 vllm 服务器..."
srun --nodelist=${HEAD_NODE} --ntasks=1 bash -c '
    # 启动 vllm 服务器（注意：这里后台执行并将 PID 写入文件）
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
    echo $! > vllm_server.pid
    # 使用 wait 命令保持该 shell 进程，直到外部 kill 结束
    wait
' &

# 给 vllm 服务器足够时间启动（原脚本注释中写等待20分钟，但 sleep 300 实际为 5 分钟，可根据情况调整）
echo "等待 vllm 服务器启动（等待 5 分钟）..."
sleep 450

# 第二步：在 head 节点上执行 run_mathv_R1V.py
echo "在 Head 节点上执行 run_mathv_R1V2.py..."
srun --nodelist=${HEAD_NODE} --ntasks=1 --overlap bash -c '
    cd /ceph/home/yaodong01/s1-m/pcwen/s1m/ &&
    HF_ENDPOINT=https://hf-mirror.com python run_Oly_R1V2.py > run_Oly_R1V2.log 2>&1
'
RUN_STATUS=$?
if [ ${RUN_STATUS} -eq 0 ]; then
    echo "run_mathv_R1V2.py 执行成功"
else
    echo "run_mathv_R1V2.py 执行失败，退出代码: ${RUN_STATUS}"
fi

# 第三步：在 head 节点上停止 vllm 服务器
echo "停止 vllm 服务器..."
srun --nodelist=${HEAD_NODE} --ntasks=1 --overlap bash -c '
    if [ -f vllm_server.pid ]; then
        kill $(cat vllm_server.pid)
        rm -f vllm_server.pid
        echo "vllm 服务器已停止."
    else
        echo "未检测到 vllm 服务器 PID 文件."
    fi
'

# 等待所有后台任务结束
wait

echo "全部任务完成."