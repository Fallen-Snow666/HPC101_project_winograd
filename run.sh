#!/bin/bash
#SBATCH --job-name=winograd
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# 设置CUDA环境
module load cuda

# 运行GPU程序
./winograd inputs/config.txt