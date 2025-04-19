#!/bin/bash
#SBATCH --job-name=qwen         # 任务名称
#SBATCH --output=/aifs4su/hansirui/pcwen_output/slurm_logs/deploy_%j.log    # 输出日志文件
#SBATCH --error=/aifs4su/hansirui/pcwen_output/slurm_logs/deploy_%j.log      # 错误日志文件
#SBATCH --partition=gov                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:8                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=220
#SBATCH --mem=500GB
# 5120 不行
# 3120 不行
# scheduler-delay-factor 这个参数很重要, 有上涨的空间
# max_num_batched_tokens 
# max_num_batched_tokens  32768*2 \ 会爆
# 使用mathviosn 时 limit-mm-per-prompt 将这个调整为1  这个值尽可能小 这样并发能调大
# 使用OLY 时 limit-mm-per-prompt 将这个调整为5  这个值尽可能小 这样并发能调大
conda activate vllm-server
srun python -m vllm.entrypoints.openai.api_server --model Skywork/Skywork-R1V-38B \
                                                    --port 8000 \
                                                    --tensor-parallel-size 8 \
                                                    --gpu-memory-utilization 0.8 \
                                                    --served-model-name Skywork-R1V-38B \
                                                    --trust-remote-code \
                                                    --enable-prefix-caching \
                                                    --max_num_batched_tokens  131072 \
                                                    --max_num_seqs  2048 \
                                                    --scheduler-delay-factor 12 \
                                                    --limit-mm-per-prompt "image=5" \
                                                    --chat-template /home/hansirui_2nd/pcwen_workspace/s1m_assemble/tempalte_skyt1_stts.jinja

# source ~/s1-m/xuyao/venv/xuyao/bin/activate
# conda activate base
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download Skywork/Skywork-R1V-38B --local-dir ~/s1-m/Models/Skywork-R1V-38B
# huggingface-cli download Qwen/QVQ-72B-Preview --local-dir ~/s1-m/Models/QVQ-72B-Preview
# huggingface-cli download Fancy-MLLM/R1-Onevision-7B-RL --local-dir ~/s1-m/Models/R1-Onevision-7B-RL
# huggingface-cli download omkarthawakar/LlamaV-o1 --local-dir ~/s1-m/Models/LlamaV-o1


Fancy-MLLM/R1-Onevision-7B-RL

python -m vllm.entrypoints.openai.api_server --model Skywork/Skywork-R1V-38B \
                                                    --port 8000 \
                                                    --tensor-parallel-size 8 \
                                                    --gpu-memory-utilization 0.8 \
                                                    --served-model-name Skywork-R1V-38B \
                                                    --trust-remote-code \
                                                    --enable-prefix-caching \
                                                    --max_num_batched_tokens  32768 \
                                                    --max_num_seqs  2048 \
                                                    --scheduler-delay-factor 12 \
                                                    --limit-mm-per-prompt "image=1" \
                                                    --chat-template /home/hansirui_2nd/pcwen_workspace/s1m_assemble/tempalte_skyt1_stts.jinja