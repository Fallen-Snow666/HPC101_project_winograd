# Winograd 卷积

本项目是 Winograd 卷积算法的一个高性能、双卡并行实现。代码经过深度优化，旨在最大化利用双 NVIDIA V100 GPU 节点的计算能力。

### 核心优化技术

* **并行模型**: 采用基于 **数据并行** 的双 GPU 方案，将每个卷积层的批次 (Batch) 工作量平分给两张 GPU 卡处理。
* **通信库**: 使用 NVIDIA 官方的 **NCCL** 库 (`ncclAllGather`) 实现 GPU 间高效、低延迟的结果合并。
* **计算核心**: 使用 NVIDIA 开源的 **CUTLASS** C++ 模板库执行核心的批量矩阵乘法（Batched GEMM），以实现编译时深度优化。
* **Winograd 方案**: 主力方案采用计算密度更高的 **F(4x4, 3x3)**，对特定小通道层则回退到轻量级的 **F(2x2, 3x3)** 方案以保证鲁棒性。
* **性能调优**: 针对已知网络层维度，采用**启发式参数分发 (Heuristic-Based Dispatcher)** 策略，为每个核函数选择预先手动调优好的最优启动参数，以压榨硬件性能。
* **内存管理**: 在主机代码中为 GPU 预先分配大型“工作区 (Workspace)”内存，并在循环中复用，避免了反复内存分配/释放带来的巨大开销。

### 环境与依赖

* **硬件**: 2 x NVIDIA V100 GPU (或任何支持 `sm_70` 及以上架构的双卡节点)
* **软件环境**:
    * **Spack 包管理器**: 用于加载编译和运行环境。
    * **nvhpc 包**: NVIDIA HPC SDK，提供了 CUDA 编译器 (`nvcc`) 和 NCCL 库。
    * **CUTLASS 库**: 需要从 GitHub 手动克隆。
* **作业调度**: Slurm

### 如何编译

1.  **获取 CUTLASS 库**

    编译前，必须先将 CUTLASS 库克隆到项目根目录。`Makefile` 文件配置为在 `./cutlass/include` 路径下寻找其头文件。
    ```bash
    # 确保你位于项目根目录 (包含 Makefile 的目录)
    git clone [https://github.com/NVIDIA/cutlass.git](https://github.com/NVIDIA/cutlass.git)
    ```

2.  **加载编译环境**

    使用 Spack 加载 `nvhpc` 套件，以获取 `nvcc` 编译器和 NCCL 库。
    ```bash
    spack load nvhpc
    ```

3.  **执行编译**

    直接运行 `make` 命令即可。
    ```bash
    make
    ```
    成功后，会生成一个名为 `winograd` 的可执行文件。

### 如何运行

1.  **准备运行脚本 (`run.sh`)**

    代码的运行通过 Slurm 作业脚本提交。请创建或确保 `run.sh` 文件内容如下。该脚本负责申请资源、设置运行时环境并执行程序。
    ```bash
    #!/bin/bash
    #SBATCH --job-name=winograd
    #SBATCH --partition=V100
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --gpus=2         # 必须申请2个GPU
    #SBATCH --cpus-per-task=8
    #SBATCH --time=01:00:00
    #SBATCH --output=%x_%j.log

    # 设置 Spack 环境
    source /pxe/opt/spack/share/spack/setup-env.sh
    spack load nvhpc

    # 手动添加 NCCL 库的运行时搜索路径
    # (这是一个在 Spack 环境可能出问题时的保障措施)
    NCCL_LIB_PATH=/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/nvhpc-25.1-gfpvhsdurdxu5qqwgkxsn6m76eohxn25/Linux_x86_64/25.1/comm_libs/12.6/nccl/lib
    export LD_LIBRARY_PATH=$NCCL_LIB_PATH:$LD_LIBRARY_PATH

    # 运行 GPU 程序
    ./winograd inputs/config.txt
    ```

2.  **提交作业**

    使用 `sbatch` 命令提交作业。
    ```bash
    sbatch run.sh
    ```

3.  **查看结果**

    作业运行结束后，结果会保存在一个名为 `winograd_[作业ID].log` 的文件中。你可以使用 `cat` 或 `less` 命令查看。

### 预期输出示例

一个成功的运行输出应该类似下面这样，显示所有层计算正确，并给出最终的性能数据和加速比。

```
NCCL communicators initialized for 2 GPUs.
============================================================

=== Running Naive Convolution (Baseline on a single GPU) ===
... (各层结果) ...
Baseline Total: 1087.195 ms (1953.27 GFLOPS)

=== Running Winograd Convolution (2 GPUs with NCCL) ===
... (各层结果, 包含 INFO: Using heuristic config ... 等信息) ...
Winograd Total (2 GPUs): 76.382 ms (27802.30 GFLOPS)

=== Correctness Check ===
... (各层 CORRECTNESS 检查) ...

=== Final Results ===
All layers passed correctness check!
Baseline Total (1 GPU): 1087.066 ms (1953.51 GFLOPS)
Winograd Total (2 GPUs): 76.344 ms (27816.04 GFLOPS)
Overall Speedup: 14.24x
```