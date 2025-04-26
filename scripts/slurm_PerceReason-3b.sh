#!/bin/bash
#SBATCH --job-name=3blmm         # 任务名称
#SBATCH --output=/ceph/home/yaodong01/s1-m/pcwen/derun_%j.log    # 输出日志文件
#SBATCH --error=/ceph/home/yaodong01/s1-m/pcwen/derun_%j.log    # 输出日志文件
#SBATCH --partition=IAI_SLURM_HGX                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:4                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=220
#SBATCH --mem=500GB
#SBATCH --qos=12gpu-hgx 
#SBATCH --time=3:00:00


srun  run_PerceReason-3b.sh