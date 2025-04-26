#!/bin/bash
#SBATCH --job-name=skywork-R1v         # 任务名称
#SBATCH --output=/ceph/home/yaodong01/s1-m/pcwen/budget-forcing/deploy_%j.log    # 输出日志文件
#SBATCH --error=/ceph/home/yaodong01/s1-m/pcwen/budget-forcing/deploy_%j.log    # 输出日志文件
#SBATCH --partition=IAI_SLURM_HGX                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:4                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=220
#SBATCH --mem=500GB
#SBATCH --qos=12gpu-hgx 
#SBATCH --time 1:00:00



python -m vllm.entrypoints.openai.api_server --model /aifs4su/hansirui_2nd/models/Qwen2.5-VL-72B-Instruct \
                                                    --port 8000 \
                                                    --tensor-parallel-size 8 \
                                                    --gpu-memory-utilization 0.8 \
                                                    --served-model-name Qwen2.5-VL-72B-Instruct \
                                                    --trust-remote-code \
                                                    --enable-prefix-caching \
                                                    --scheduler-delay-factor 12 \
                                                    --limit-mm-per-prompt "image=5" \
                                                    --chat-template /home/hansirui_2nd/pcwen_workspace/s1m_assemble/qvq_test-scaling.jinja



python -m vllm.entrypoints.openai.api_server --model /aifs4su/hansirui_2nd/models/Qwen2.5-VL-32B-Instruct \
--port 8000 \
--tensor-parallel-size 8 \
--gpu-memory-utilization 0.8 \
--served-model-name Qwen2.5-VL-32B-Instruct \
--trust-remote-code \
--enable-prefix-caching \
--scheduler-delay-factor 12 \
--limit-mm-per-prompt "image=5" \
--chat-template /home/hansirui_2nd/pcwen_workspace/s1m_assemble/qvq_test-scaling.jinja