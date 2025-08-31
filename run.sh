#!/bin/bash
#SBATCH --job-name=winograd
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# 设置环境
source /pxe/opt/spack/share/spack/setup-env.sh
spack load nvhpc

NCCL_LIB_PATH=/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/nvhpc-25.1-gfpvhsdurdxu5qqwgkxsn6m76eohxn25/Linux_x86_64/25.1/comm_libs/12.6/nccl/lib
export LD_LIBRARY_PATH=$NCCL_LIB_PATH:$LD_LIBRARY_PATH

# 运行GPU程序
./winograd inputs/config.txt