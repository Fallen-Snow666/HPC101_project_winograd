#include "winograd.cuh"
#include <cublas_v2.h>
#include <thrust/host_vector.h>

// Transformation matrices for F(2x2, 3x3)
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, -1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f},
    {0.0f,  1.0f, -1.0f,  1.0f},
    {-1.0f, 1.0f,  1.0f,  0.0f},
    {0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, -1.0f}
};

// Kernel 1: 滤波器变换 (U = G * g * G^T)
__global__ void filter_transform_kernel(const float* filter, float* U, int K, int C) {
    int k = blockIdx.x;
    int c = threadIdx.x;

    if (k >= K || c >= C) return;

    float g[3][3];
    const float* g_ptr = filter + (k * C + c) * 9;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            g[i][j] = g_ptr[i * 3 + j];
        }
    }

    float temp[4][3] = {{0.0f}};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            #pragma unroll
            for (int l = 0; l < 3; ++l) {
                temp[i][j] += G[i][l] * g[l][j];
            }
        }
    }

    float u_kc[4][4] = {{0.0f}};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int l = 0; l < 3; ++l) {
                u_kc[i][j] += temp[i][l] * G[j][l];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            U[(i * 4 + j) * K * C + k * C + c] = u_kc[i][j];
        }
    }
}

// Kernel 2: 输入图块变换 (V = B^T * d * B)
__global__ void image_transform_kernel(const float* image, float* V, int N, int C, int H, int W, int P_total, int P_H, int P_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * P_total) return;

    int p = idx % P_total;
    int c = idx / P_total;
    
    int n = p / (P_H * P_W);
    int p_local = p % (P_H * P_W);
    int tile_y = p_local / P_W;
    int tile_x = p_local % P_W;

    int h_start = tile_y * 2;
    int w_start = tile_x * 2;

    float d[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int h = h_start + i;
            int w = w_start + j;
            if (h < H && w < W) {
                d[i][j] = image[(n * C + c) * H * W + h * W + w];
            } else {
                d[i][j] = 0.0f;
            }
        }
    }

    float temp[4][4] = {{0.0f}};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                temp[i][j] += B_T[i][l] * d[l][j];
            }
        }
    }
    
    float v_cp[4][4] = {{0.0f}};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                v_cp[i][j] += temp[i][l] * B[l][j];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            V[(i * 4 + j) * C * P_total + c * P_total + p] = v_cp[i][j];
        }
    }
}

// Kernel 3: 输出变换 (Y = A^T * m * A)
__global__ void output_transform_kernel(const float* M, float* out, int N, int K, int outH, int outW, int P_total, int P_H, int P_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * P_total) return;

    int p = idx % P_total;
    int k = idx / P_total;

    int n = p / (P_H * P_W);
    int p_local = p % (P_H * P_W);
    int tile_y = p_local / P_W;
    int tile_x = p_local % P_W;

    float m_kp[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            m_kp[i][j] = M[(i * 4 + j) * K * P_total + k * P_total + p];
        }
    }
    
    float temp[2][4] = {{0.0f}};
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                temp[i][j] += A_T[i][l] * m_kp[l][j];
            }
        }
    }
    
    float A[4][2] = {{1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, -1.0f}, {0.0f, -1.0f}};

    float Y[2][2] = {{0.0f}};
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                Y[i][j] += temp[i][l] * A[l][j];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                out[((n * K + k) * outH + h) * outW + w] = Y[i][j];
            }
        }
    }
}


// 优化的 Winograd 卷积主函数
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter,
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V,
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {

    const int outH = H - 2;
    const int outW = W - 2;
    const int P_H = outH / 2;
    const int P_W = outW / 2;
    const int P = N * P_H * P_W;

    const int threads_per_block = 256;

    // 1. 滤波器变换
    dim3 grid_filter(K);
    dim3 block_filter(C);
    filter_transform_kernel<<<grid_filter, block_filter>>>(
        thrust::raw_pointer_cast(filter.data()),
        thrust::raw_pointer_cast(U.data()),
        K, C
    );

    // 2. 输入变换
    int grid_size_image = (C * P + threads_per_block - 1) / threads_per_block;
    image_transform_kernel<<<grid_size_image, threads_per_block>>>(
        thrust::raw_pointer_cast(image.data()),
        thrust::raw_pointer_cast(V.data()),
        N, C, H, W, P, P_H, P_W
    );

    // 3. 批量矩阵乘法 (cuBLAS)
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int m = K;
    int n = P;
    int k_gemm = C;
    int batch_count = 16;

    thrust::device_vector<float*> d_A_array(batch_count);
    thrust::device_vector<float*> d_B_array(batch_count);
    thrust::device_vector<float*> d_C_array(batch_count);
    
    thrust::host_vector<float*> h_A_array(batch_count);
    thrust::host_vector<float*> h_B_array(batch_count);
    thrust::host_vector<float*> h_C_array(batch_count);

    float* U_ptr = thrust::raw_pointer_cast(U.data());
    float* V_ptr = thrust::raw_pointer_cast(V.data());
    float* M_ptr = thrust::raw_pointer_cast(M.data());

    for (int i = 0; i < batch_count; ++i) {
        h_A_array[i] = U_ptr + i * m * k_gemm;
        h_B_array[i] = V_ptr + i * k_gemm * n;
        h_C_array[i] = M_ptr + i * m * n;
    }

    d_A_array = h_A_array;
    d_B_array = h_B_array;
    d_C_array = h_C_array;

    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, m, k_gemm,
                       &alpha,
                       (const float**) thrust::raw_pointer_cast(d_B_array.data()), n,
                       (const float**) thrust::raw_pointer_cast(d_A_array.data()), k_gemm,
                       &beta,
                       thrust::raw_pointer_cast(d_C_array.data()), n,
                       batch_count);

    cublasDestroy(handle);

    // 4. 输出变换
    int grid_size_output = (K * P + threads_per_block - 1) / threads_per_block;
    output_transform_kernel<<<grid_size_output, threads_per_block>>>(
        thrust::raw_pointer_cast(M.data()),
        thrust::raw_pointer_cast(out.data()),
        N, K, outH, outW, P, P_H, P_W
    );

    cudaDeviceSynchronize();
}