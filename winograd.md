## 优化思路

Winograd 算法的核心优势在于将卷积中大量的乘法运算，通过数学变换，转化到了一个计算量更小的**“Winograd 域”**中 。在变换到 Winograd 域之后，需要执行以下计算：

```
M(k, p, α, β) = Σ [c=0 to C-1] U(k, c, α, β) * V(c, p, α, β)
```

其中：

- `α, β` 是 Winograd 域中 4x4 矩阵的坐标（共16个）。
- `k` 是输出通道，`c` 是输入通道，`p` 是空间图块。
- `U` 是变换后的滤波器，`V` 是变换后的输入数据。

即**变换后的滤波器矩阵 U** 和 **变换后的输入矩阵 V** 之间的**元素级乘积累加**。

这个公式看起来是元素级别的乘积累加。但如果我们固定 `α` 和 `β`，那么对于任意一个固定的 `(α, β)` 坐标，我们可以定义两个新的二维矩阵：

- `U_αβ`: 一个 `K x C` 的矩阵，其第 `k` 行 `c` 列的元素是 `U(k, c, α, β)`。
- `V_αβ`: 一个 `C x P` 的矩阵，其第 `c` 行 `p` 列的元素是 `V(c, p, α, β)`。

这样一来，对于这个固定的 `(α, β)`，原先的求和公式就变成了一个标准的**矩阵乘法 (GEMM)**： `M_αβ = U_αβ * V_αβ` 这里 `M_αβ` 是一个 `K x P` 的结果矩阵。由于 `(α, β)` 共有 `4x4 = 16` 种组合，所以 Winograd 的核心计算就等价于 **16 个独立的、规模完全相同的 GEMM 操作**。这个过程，就是**批量矩阵乘法 (Batched GEMM)**。

那么我们就可以使用 **cuBLAS** 进行优化。**cuBLAS** 是 NVIDIA 官方提供的、为自家 GPU 深度优化的基础线性代数子程序库（BLAS），其中包含的 `sgemm` (单精度通用矩阵乘法) 函数是 GPU 上性能最高的函数之一。

## 优化方法

基于上述思路，将原来的核函数拆分成四个阶段：

1. **阶段一：滤波器变换 (Filter Transformation)**

   我们创建了一个专门的核函数 `filter_transform_kernel`。它将在卷积计算开始前，一次性将所有 `K x C` 个 `3x3` 的卷积核权重变换为 `4x4` 的 Winograd 域矩阵 `U`，从而避免在主循环中重复计算，消除冗余。

   ```pseudocode
   // 在设备端(GPU)并行执行
   KERNEL filter_transform_kernel(filter, U_gpu, K, C):
       
       // 每个线程获取其负责的滤波器索引 (k, c)
       k, c = get_global_thread_id()
       
       // 从全局内存加载一个 3x3 滤波器 g
       g_3x3 = load_from(filter, index=(k, c))
       
       // 执行Winograd变换: U_kc = G * g * G^T
       temp_4x3 = matrix_multiply(G_4x3, g_3x3)
       U_kc_4x4 = matrix_multiply(temp_4x3, G_transpose_3x4)
       
       // 将变换后的 4x4 结果 U_kc 存回全局内存
       // 存入16个矩阵中的某一个
       store_to(U_gpu, index=(k, c), data=U_kc_4x4)
       
   END KERNEL
   ```

2. **阶段二：输入变换 (Image Transformation)**

   接下来创建了第二个核函数 `image_transform_kernel`。它将所有输入的 `4x4` 图块也变换到 `4x4` 的 Winograd 域矩阵 `V`，从而准备好进行矩阵乘法的数据。

   ```pseudocode
   // 在设备端(GPU)并行执行
   KERNEL image_transform_kernel(image, V_gpu, N, C, H, W, P):
       
       // 每个线程获取其负责的通道和图块索引 (c, p)
       c, p = get_global_thread_id()
       
       // 根据 p 计算出输入图像中的具体位置 (n, tile_y, tile_x)
       n, tile_y, tile_x = get_tile_coordinates_from_p(p, N, P_H, P_W)
       
       // 从全局内存加载一个 4x4 的输入图块 d
       d_4x4 = load_from(image, index=(n, c, tile_y, tile_x))
       
       // 执行Winograd变换: V_cp = B^T * d * B
       temp_4x4 = matrix_multiply(B_transpose_4x4, d_4x4)
       V_cp_4x4 = matrix_multiply(temp_4x4, B_4x4)
       
       // 将变换后的 4x4 结果 V_cp 存回全局内存
       store_to(V_gpu, index=(c, p), data=V_cp_4x4)
       
   END KERNEL
   ```

3. **阶段三：批量矩阵乘法 (Batched GEMM via cuBLAS)**

   调用 `cublasSgemmBatched` 函数，执行16次 `K x P = (K x C) * (C x P)` 的矩阵乘法。

   - **创建句柄**：首先需要创建一个 `cublasHandle_t`，这是所有 cuBLAS 操作的上下文。

   - **准备指针数组**：`cublasSgemmBatched` 函数不直接接收 `U` 和 `V` 的大块内存指针，而是接收一个**指针的数组**。这个数组的每个元素，分别指向16个独立矩阵的起始地址。因此，我们需要：

     - 在主机端创建一个 `thrust::host_vector<float*>`。

     - 用一个循环，计算出 `U` 和 `V` 中每个子矩阵的起始地址，并存入这个主机向量。

     - 将这个存满指针的主机向量，拷贝到 `thrust::device_vector<float*>`，即GPU设备端。

   - **处理行列主序**：

     - cuBLAS 沿用 Fortran 的习惯，默认输入矩阵是**列主序**存储的。而 CUDA 中的二维数组默认是**行主序**存储。

     - 一个 `m x k` 的行主序矩阵，在 cuBLAS 看来是一个 `k x m` 的列主序矩阵。

     - 我们想计算 `M = U * V`（行主序）。利用矩阵转置性质 `(U * V)ᵀ = Vᵀ * Uᵀ`，可以通过安排参数，让 cuBLAS 去计算 `M_T = V * U`（cuBLAS 会把我们的行主序V和U当作列主序的Vᵀ和Uᵀ）。cuBLAS 计算出的列主序结果 `M_T`，在内存中的排列，就是想要的行主序 `M` 的排列。

     - 因此，在调用 `cublasSgemmBatched` 时，我们传入的矩阵顺序是 `V` 在前，`U` 在后，并相应调整维度参数。

   - **执行调用**：配置好所有参数（矩阵维度 `m,n,k`、批次数 `batchCount=16`、指针数组等）后，执行 `cublasSgemmBatched`。

4. **阶段四：输出变换 (Output Transformation)**

   创建第三个核函数 `output_transform_kernel` 将 cuBLAS 计算得到的结果矩阵 `M`，通过逆变换，从 `4x4` 的 Winograd 域转换回 `2x2` 的空间域，得到最终的输出像素。

   ```pseudocode
   // 在设备端(GPU)并行执行
   KERNEL output_transform_kernel(M_gpu, output, N, K, outH, outW, P):
       
       // 每个线程获取其负责的输出通道和图块索引 (k, p)
       k, p = get_global_thread_id()
       
       // 从全局内存加载一个由cuBLAS计算出的 4x4 结果矩阵 M_kp
       M_kp_4x4 = load_from(M_gpu, index=(k, p))
       
       // 执行逆变换: Y = A^T * M * A
       temp_2x4 = matrix_multiply(A_transpose_2x4, M_kp_4x4)
       Y_2x2 = matrix_multiply(temp_2x4, A_4x2)
       
       // 根据 p 计算出输出图像中的具体位置 (n, tile_y, tile_x)
       n, tile_y, tile_x = get_tile_coordinates_from_p(p, N, P_H, P_W)
       
       [cite_start]// 将最终的 2x2 结果 Y 存回输出张量的正确位置 [cite: 3]
       store_to(output, index=(n, k, tile_y, tile_x), data=Y_2x2)
       
   END KERNEL
   ```

   加速比 7.5x

## 其它优化内容

在对 winograd 进行优化的过程中，我们还尝试了指令和启动参数调优。

1. **循环展开 (`#pragma unroll`)**

   在我们的变换核函数中，存在一些固定迭代次数的短循环（例如矩阵变换中的 3x3 或 4x4 循环）。可以在这些循环前加上 `#pragma unroll` 指令，告诉编译器完全展开循环。这可以消除循环本身的开销，并为编译器提供更大的指令调度空间，从而提升效率。

   **实际提升并不是很明显。**

2. **调整启动参数**

   我们当前为每个核函数固定了 `threads_per_block = 256`。通过尝试不同的线程块大小（如 128、512、1024），并测试哪种配置能获得最佳性能。

   | threads_per_block | Winograd 总时间 (ms) |  GFLOPS  | 加速比 |
   | ----------------: | :------------------: | :------: | ------ |
   |                64 |        126.11        | 16839.18 | 7.37x  |
   |               128 |        126.08        | 16843.24 | 7.35x  |
   |               256 |        123.05        | 17257.32 | 7.50x  |
   |               512 |        126.00        | 16854.37 | 7.35x  |
   |              1024 |        125.62        | 16904.34 | 7.39x  |

   **最后选择 256。**

## 进一步优化

### 数据并行

对于每一个计算量可能很大的卷积层，我们都将其输入数据（Batch `N`）一分为二，让两张V100 GPU卡集中算力，同时处理这一层的计算任务。这样可以最大限度地缩短单个瓶颈层的耗时，从而降低总体时间。通过`main.cu`中的多线程调度，将`h_images`数据分片后异步拷贝到两张GPU上，并启动`winograd_conv`函数。

### 核心计算优化：采用更高阶的 Winograd + CUTLASS

1. **F(4x4, 3x3) Winograd**

   相比我们早期使用的F(2,3)方案，F(4,3)方案具有更高的**计算密度**。它通过一次`6x6`的矩阵变换，可以计算出`4x4=16`个输出点，算术复杂度和内存访问量的比值更高，能更好地发挥V100这类计算密集型GPU的性能。`winograd_conv.cu`中的变换核函数 `transform_filter_GgGT`, `transform_input_BTdB` 等都是基于F(4,3)的`6x6`矩阵变换实现的。

2. **CUTLASS GEMM**

   将Winograd变换后最核心的批量矩阵乘法（Batched GEMM）任务，交给比cuBLAS更灵活、更可控的 **CUTLASS** 库来完成。在`winograd_conv`函数中，我们定义了`cutlass::gemm::device::GemmBatched` 操作，并构造其参数以执行36次并行的矩阵乘法。

### 启发式参数调优 (Heuristic-Based Tuning)

CUDA核函数的性能对启动参数（线程块大小、网格大小等）极其敏感，因此，我们为 `config.txt` 中每一个已知的网络层，都匹配一套**预先通过大量实验找到的最优启动参数**。`winograd_conv.cu`中有大量的 `if/else if` 分支，通过检查当前层的维度 `C, H, W, K, N`，来选择对应的、硬编码的“最优解”启动参数。日志中会打印的 `INFO: Using heuristic config for layer...` 表示选择不同的参数。

### NCCL 多卡通信

数据并行后，需要将两张卡各自计算的结果高效地合并成一个完整的输出。手动通过CPU中转或P2P拷贝效率低下。我们使用NVIDIA官方的**NCCL**，它是专门为多GPU间通信设计的、性能极高的库。在`main.cu`中，我们初始化了NCCL通信域，并在每层计算结束后调用 `ncclAllGather` 函数，它能以极高的带宽（通常是NVLink带宽）完成结果的合并。

### 精细的内存管理

为避免在循环中反复进行昂贵的GPU内存分配和释放（`cudaMalloc`/`cudaFree`），在`main.cu`的主循环开始前，我们计算出所有层中可能需要的最大内存，为每个GPU**一次性地预分配**好足够大的“工作区（Workspace）”。在循环中，我们只向这个工作区异步拷贝数据，大大降低了驱动开销。