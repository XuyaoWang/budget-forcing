#!/bin/bash
#SBATCH --job-name=skyr1v2         # 任务名称
#SBATCH --output=/ceph/home/yaodong01/s1-m/pcwen/s1m/derun_%j.log    # 输出日志文件
#SBATCH --error=/ceph/home/yaodong01/s1-m/pcwen/s1m/derun_%j.log    # 输出日志文件
#SBATCH --partition=IAI_SLURM_HGX                # 分区
#SBATCH --nodes=1                      # 节点数
#SBATCH --gres=gpu:8                   # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=100
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500GB
#SBATCH --qos=12gpu-hgx 
#SBATCH --time=36:00:00

srun bigrun.sh