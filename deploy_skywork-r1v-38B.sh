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



python -m vllm.entrypoints.openai.api_server --model /ceph/home/yaodong01/s1-m/Models/Skywork-R1V-38B/ \
                                                    --port 8000 \
                                                    --tensor-parallel-size 4 \
                                                    --gpu-memory-utilization 0.8 \
                                                    --served-model-name Skywork-R1V-38B \
                                                    --trust-remote-code \
                                                    --enable-prefix-caching \
                                                    --max_num_batched_tokens  32768 \
                                                    --max_num_seqs  2048 \
                                                    --scheduler-delay-factor 12 \
                                                    --limit-mm-per-prompt "image=1" \
                                                    --chat-template /ceph/home/yaodong01/s1-m/pcwen/budget-forcing/tempalte_skyt1_stts.jinja