#include "winograd.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thrust/device_vector.h>

// Use pragma to suppress specific warnings from CUTLASS headers
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015

// Include CUTLASS GEMM library
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>

// Restore default warning settings
#pragma nv_diag_default 20013
#pragma nv_diag_default 20015


// =======================================================================
// ====================== Error Checking and Basic Definitions =========================
// =======================================================================

// CUDA API call error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// =======================================================================
// ====================== Scheme A: F(2x2, 3x3) Fused Kernel (Fallback) ============
// =======================================================================
// This part of the code is correct and remains unchanged.
__constant__ float G_2x2[4][3] = {
    {1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}
};
__constant__ float B_T_2x2[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, -1.0f}
};
__constant__ float B_2x2[4][4] = {
    {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, -1.0f}
};
__constant__ float A_T_2x2[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, -1.0f, -1.0f}
};
__constant__ float A_2x2[4][2] = {
    {1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, -1.0f}, {0.0f, -1.0f}
};


__global__
void winograd_conv_fused_kernel_2x2(const float* __restrict__ image,
                                  const float* __restrict__ filter,
                                  float* __restrict__ output,
                                  int N, int C, int H, int W, int K, int outH, int outW) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N * K * ((outH + 1) / 2) * ((outW + 1) / 2);
         idx += gridDim.x * blockDim.x) {

        const int tiles_w = (outW + 1) / 2;
        const int tiles_h = (outH + 1) / 2;
        const int tiles_per_image = tiles_h * tiles_w;

        int p_local = idx % tiles_per_image;
        int k = (idx / tiles_per_image) % K;
        int n = idx / (K * tiles_per_image);
        int tile_y = p_local / tiles_w;
        int tile_x = p_local % tiles_w;

        float m[4][4] = {{0.0f}};

        for (int c = 0; c < C; ++c) {
            const float* g = filter + (k * C + c) * 9;
            float u_kc[4][4];
            float temp_g[4][3];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 3; ++j) {
                    temp_g[i][j] = G_2x2[i][0] * g[0 * 3 + j] + G_2x2[i][1] * g[1 * 3 + j] + G_2x2[i][2] * g[2 * 3 + j];
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                u_kc[i][0] = temp_g[i][0];
                u_kc[i][1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
                u_kc[i][2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
                u_kc[i][3] = temp_g[i][2];
            }

            int h_start = tile_y * 2;
            int w_start = tile_x * 2;
            float d[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int h_in = h_start + i;
                    int w_in = w_start + j;
                    if (h_in < H && w_in < W) {
                        d[i][j] = image[(n * C + c) * H * W + h_in * W + w_in];
                    } else {
                        d[i][j] = 0.0f;
                    }
                }
            }
            float v_ncp[4][4];
            float temp_d[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    temp_d[i][j] = B_T_2x2[i][0] * d[0][j] + B_T_2x2[i][1] * d[1][j] + B_T_2x2[i][2] * d[2][j] + B_T_2x2[i][3] * d[3][j];
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v_ncp[i][j] = temp_d[i][0] * B_2x2[0][j] + temp_d[i][1] * B_2x2[1][j] + temp_d[i][2] * B_2x2[2][j] + temp_d[i][3] * B_2x2[3][j];
                }
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    m[i][j] += u_kc[i][j] * v_ncp[i][j];
                }
            }
        }

        float temp_m[2][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                temp_m[i][j] = A_T_2x2[i][0] * m[0][j] + A_T_2x2[i][1] * m[1][j] + A_T_2x2[i][2] * m[2][j] + A_T_2x2[i][3] * m[3][j];
            }
        }
        float Y[2][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            Y[i][0] = temp_m[i][0] + temp_m[i][1] + temp_m[i][2];
            Y[i][1] = temp_m[i][1] - temp_m[i][2] - temp_m[i][3];
        }

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int h = tile_y * 2 + i;
                int w = tile_x * 2 + j;
                if (h < outH && w < outW) {
                    output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
                }
            }
        }
    }
}


// =======================================================================
// ======== Scheme B: F(4x4, 3x3) Multi-kernel+Tiling+CUTLASS (Main Scheme) =========
// =======================================================================
namespace winograd_4x4_kernels {

// --- Kernel 1: Filter Transform (U = G * g * G^T) ---
template <class T>
__device__ void multiply_G(T col_in[3], T col_out[6]) {
    auto temp1 = static_cast<T>(-1.0f / 6.0f) * col_in[0];
    auto temp2 = static_cast<T>(-1.0f / 6.0f) * col_in[1];
    auto temp3 = static_cast<T>(-1.0f / 6.0f) * col_in[2];
    col_out[0] = static_cast<T>(1.0f / 4.0f) * col_in[0];
    col_out[1] = temp1 + temp2 + temp3;
    col_out[2] = temp1 - temp2 + temp3;

    temp1 = static_cast<T>(1.0f / 24.0f) * col_in[0];
    temp2 = static_cast<T>(1.0f / 12.0f) * col_in[1];
    auto temp4 = static_cast<T>(1.0f / 6.0f) * col_in[2];
    col_out[3] = temp1 + temp2 + temp4;
    col_out[4] = temp1 - temp2 + temp4;
    col_out[5] = col_in[2];
}

// --- Tunable parameters for Filter Transform (FALLBACK/DEFAULT) ---
constexpr int TUNABLE_FILTER_NUM_KERNELS_PER_BLOCK = 40;
constexpr int TUNABLE_FILTER_BLOCK_SIZE = 128;

template <class ElementInput, class ElementCompute, class ElementOutput, int NUM_KERNELS_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void transform_filter_GgGT(
    ElementOutput* __restrict__ output, const ElementInput* __restrict__ input,
    int C, int K) {
    // RETAINED OPTIMIZATION: Added padding to shared memory to avoid bank conflicts
    constexpr int PADDING = 1;

    const auto num_kernels = K * C;
    const auto kernel_idx_base = static_cast<int>(blockIdx.x) * NUM_KERNELS_PER_BLOCK;

    __shared__ ElementInput shared_kernel[NUM_KERNELS_PER_BLOCK][3][3];
    for (int idx = threadIdx.x; idx < NUM_KERNELS_PER_BLOCK * 9; idx += BLOCK_SIZE) {
        auto el = idx % 9;
        auto kc = idx / 9;
        if (kernel_idx_base + kc < num_kernels)
            shared_kernel[kc][el / 3][el % 3] = input[(kernel_idx_base + kc) * 9 + el];
    }
    __syncthreads();

    __shared__ ElementCompute shared_6x3[NUM_KERNELS_PER_BLOCK][6][3 + PADDING];
    for (int idx = threadIdx.x; idx < NUM_KERNELS_PER_BLOCK * 3; idx += BLOCK_SIZE) {
        auto col_idx = idx % 3;
        auto kc = idx / 3;
        if (kernel_idx_base + kc < num_kernels) {
            ElementCompute col_in[3];
            for (int i = 0; i < 3; i++) col_in[i] = shared_kernel[kc][i][col_idx];
            ElementCompute col_out[6];
            multiply_G(col_in, col_out);
            for (int i = 0; i < 6; i++) shared_6x3[kc][i][col_idx] = col_out[i];
        }
    }
    __syncthreads();

    __shared__ ElementCompute shared_6x6[NUM_KERNELS_PER_BLOCK][6][6 + PADDING];
    for (int idx = threadIdx.x; idx < NUM_KERNELS_PER_BLOCK * 6; idx += BLOCK_SIZE) {
        auto row_idx = idx % 6;
        auto kc = idx / 6;
        if (kernel_idx_base + kc < num_kernels) {
            ElementCompute row_in[3];
            for (int i = 0; i < 3; i++) row_in[i] = shared_6x3[kc][row_idx][i];
            ElementCompute row_out[6];
            multiply_G(row_in, row_out);
            for (int i = 0; i < 6; i++) shared_6x6[kc][row_idx][i] = row_out[i];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 36 * NUM_KERNELS_PER_BLOCK; i += BLOCK_SIZE) {
        auto kc_local = i % NUM_KERNELS_PER_BLOCK;
        auto tile_offset = i / NUM_KERNELS_PER_BLOCK; // This is 'alpha'
        auto global_kc = kernel_idx_base + kc_local;
        if (global_kc < num_kernels) {
            output[tile_offset * num_kernels + global_kc] = shared_6x6[kc_local][tile_offset / 6][tile_offset % 6];
        }
    }
}

// --- Kernel 2: Input Transform (V = B^T * d * B) ---
template <class T>
__device__ void multiply_BT(T in[6], T out[6]) {
    auto temp1 = -4.0f * in[1] + in[3];
    auto temp2 = -4.0f * in[2] + in[4];
    out[0] = 4.0f * in[0] - 5.0f * in[2] + in[4];
    out[1] = temp1 + temp2;
    out[2] = -temp1 + temp2;
    out[3] = -2.0f * in[1] - in[2] + 2.0f * in[3] + in[4];
    out[4] = 2.0f * in[1] - in[2] - 2.0f * in[3] + in[4];
    out[5] = 4.0f * in[1] - 5.0f * in[3] + in[5];
}

// --- Tunable parameters for Input Transform (FALLBACK/DEFAULT) ---
constexpr int TUNABLE_INPUT_TILES_Y_PER_BLOCK = 2;
constexpr int TUNABLE_INPUT_TILES_X_PER_BLOCK = 8;
constexpr int TUNABLE_INPUT_BLOCK_SIZE = 128;


template <class ElementInput, class ElementCompute, class ElementOutput, int TILES_Y_PER_BLOCK, int TILES_X_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void transform_input_BTdB(
    ElementOutput* __restrict__ output, const ElementInput* __restrict__ input,
    int C, int H, int W, int P_h, int P_w, int N, int tiles_h, int tiles_w) {
    // RETAINED OPTIMIZATION: Added padding to shared memory to avoid bank conflicts
    constexpr int PADDING = 1;

    const auto tile_x_start = static_cast<int>(blockIdx.x) * TILES_X_PER_BLOCK;
    const auto tile_y_start = static_cast<int>(blockIdx.y) * TILES_Y_PER_BLOCK;
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    auto load_element = [&] (int y, int x)->ElementInput {
        if (x < 0 || y < 0 || x >= W || y >= H) return 0;
        return input[(n * C + c) * H * W + y * W + x];
    };

    constexpr int INPUT_FRAME_X = TILES_X_PER_BLOCK * 4 + 2;
    constexpr int INPUT_FRAME_Y = TILES_Y_PER_BLOCK * 4 + 2;
    __shared__ ElementInput frame[INPUT_FRAME_Y][INPUT_FRAME_X + PADDING];

    for (int i = threadIdx.x; i < INPUT_FRAME_Y * INPUT_FRAME_X; i += BLOCK_SIZE) {
        const auto local_x = i % INPUT_FRAME_X;
        const auto local_y = i / INPUT_FRAME_X;
        auto x = tile_x_start * 4 - P_w + local_x;
        auto y = tile_y_start * 4 - P_h + local_y;
        frame[local_y][local_x] = load_element(y, x);
    }
    __syncthreads();

    __shared__ ElementCompute shared_6x6[TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK][6][6 + PADDING];
    for (int i = threadIdx.x; i < TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK * 6; i += BLOCK_SIZE) {
        const auto col_idx = i % 6;
        const auto tile_idx = i / 6;
        const auto tile_x = tile_idx % TILES_X_PER_BLOCK;
        const auto tile_y = tile_idx / TILES_X_PER_BLOCK;
        if (tile_x_start + tile_x < tiles_w && tile_y_start + tile_y < tiles_h) {
            ElementCompute col_in[6];
            for (int j = 0; j < 6; j++) col_in[j] = frame[tile_y * 4 + j][tile_x * 4 + col_idx];
            ElementCompute col_out[6];
            multiply_BT(col_in, col_out);
            for (int j = 0; j < 6; j++) shared_6x6[tile_idx][j][col_idx] = col_out[j];
        }
    }
    __syncthreads();

    __shared__ ElementOutput BTdB[TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK][36];
    for (int i = threadIdx.x; i < TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK * 6; i += BLOCK_SIZE) {
        const auto row_idx = i % 6;
        const auto tile_idx = i / 6;
        const auto tile_x = tile_idx % TILES_X_PER_BLOCK;
        const auto tile_y = tile_idx / TILES_X_PER_BLOCK;
        if (tile_x_start + tile_x < tiles_w && tile_y_start + tile_y < tiles_h) {
            ElementCompute row_in[6];
            for (int j = 0; j < 6; j++) row_in[j] = shared_6x6[tile_idx][row_idx][j];
            ElementCompute row_out[6];
            multiply_BT(row_in, row_out);
            for (int j = 0; j < 6; j++) BTdB[tile_idx][row_idx * 6 + j] = row_out[j];
        }
    }
    __syncthreads();

    const int P = N * tiles_h * tiles_w;
    for (int i = threadIdx.x; i < 36 * TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK; i += BLOCK_SIZE) {
        auto tile_local_idx = i % (TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK);
        auto alpha = i / (TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK);
        auto tile_x_local = tile_local_idx % TILES_X_PER_BLOCK;
        auto tile_y_local = tile_local_idx / TILES_X_PER_BLOCK;

        auto tile_x_global = tile_x_start + tile_x_local;
        auto tile_y_global = tile_y_start + tile_y_local;

        if (tile_x_global < tiles_w && tile_y_global < tiles_h) {
            int p_idx = (n * tiles_h + tile_y_global) * tiles_w + tile_x_global;
            int out_idx = alpha * (C * P) + c * P + p_idx;
            output[out_idx] = BTdB[tile_local_idx][alpha];
        }
    }
}

// --- Kernel 3: Output Transform (Y = A^T * M * A) ---
template <class T>
__device__ void multiply_AT(T col_in[6], T col_out[4]) {
    T temp1 = col_in[1] + col_in[2];
    T temp2 = col_in[1] - col_in[2];
    T temp3 = col_in[3] + col_in[4];
    T temp4 = col_in[3] - col_in[4];

    col_out[0] = col_in[0] + temp1 + temp3;
    col_out[1] = temp2 + 2.0f * temp4;
    col_out[2] = temp1 + 4.0f * temp3;
    col_out[3] = temp2 + 8.0f * temp4 + col_in[5];
}

template <class T>
__device__ void multiply_A(T row_in[6], T row_out[4]) {
    row_out[0] = row_in[0] + row_in[1] + row_in[2] + row_in[3] + row_in[4];
    row_out[1] = row_in[1] - row_in[2] + 2 * row_in[3] - 2 * row_in[4];
    row_out[2] = row_in[1] + row_in[2] + 4 * row_in[3] + 4 * row_in[4];
    row_out[3] = row_in[1] - row_in[2] + 8 * row_in[3] - 8 * row_in[4] + row_in[5];
}

// --- Tunable parameters for Output Transform (FALLBACK/DEFAULT) ---
constexpr int TUNABLE_OUTPUT_NUM_TILES_PER_BLOCK = 32;
constexpr int TUNABLE_OUTPUT_BLOCK_SIZE = 128;

template <class ElementInput, class ElementCompute, class ElementOutput, int NUM_TILES_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void transform_output_ATtA(
    ElementOutput* __restrict__ output, const ElementInput* __restrict__ input,
    int N, int K, int outH, int outW, int tiles_h, int tiles_w) {
    // RETAINED OPTIMIZATION: Added padding to shared memory
    constexpr int PADDING = 1;

    const int P = N * tiles_h * tiles_w;
    const int num_tiles = K * P;
    const auto tile_idx_start = static_cast<int>(blockIdx.x) * NUM_TILES_PER_BLOCK;

    __shared__ ElementInput shared_6x6[6][6 + PADDING];
    __shared__ ElementCompute shared_4x6[4][6 + PADDING];

    for (int tile_idx_local = 0; tile_idx_local < NUM_TILES_PER_BLOCK; tile_idx_local++) {

        const int tile_global_idx = tile_idx_start + tile_idx_local;
        if (tile_global_idx >= num_tiles) {
            break;
        }

        const int k = tile_global_idx / P;
        const int p_idx = tile_global_idx % P;
        for (int i = threadIdx.x; i < 36; i += BLOCK_SIZE) {
            const int alpha = i;
            const int in_idx = alpha * (K * P) + k * P + p_idx;
            shared_6x6[alpha / 6][alpha % 6] = input[in_idx];
        }
        __syncthreads();

        for (int col_idx = threadIdx.x; col_idx < 6; col_idx += BLOCK_SIZE) {
            ElementCompute col_in[6];
            for (int j = 0; j < 6; j++) {
                col_in[j] = shared_6x6[j][col_idx];
            }
            ElementCompute col_out[4];
            multiply_AT(col_in, col_out);
            for (int j = 0; j < 4; j++) {
                shared_4x6[j][col_idx] = col_out[j];
            }
        }
        __syncthreads();

        for (int row_idx = threadIdx.x; row_idx < 4; row_idx += BLOCK_SIZE) {
            ElementCompute row_in[6];
            for (int j = 0; j < 6; j++) {
                row_in[j] = shared_4x6[row_idx][j];
            }
            ElementCompute row_out[4];
            multiply_A(row_in, row_out);

            const int n = p_idx / (tiles_h * tiles_w);
            const int tile_flat = p_idx % (tiles_h * tiles_w);
            const int tile_y = tile_flat / tiles_w;
            const int tile_x = tile_flat % tiles_w;

            const int map_y = tile_y * 4 + row_idx;

            if (map_y < outH) {
                for (int j = 0; j < 4; j++) {
                    const int map_x = tile_x * 4 + j;
                    if (map_x < outW) {
                        const int out_idx = n * K * outH * outW + k * outH * outW + map_y * outW + map_x;
                        output[out_idx] = row_out[j];
                    }
                }
            }
        }
        __syncthreads();
    }
}
} // namespace winograd_4x4_kernels


// --- Host Function: Winograd Convolution Scheduler ---
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter,
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V,
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N,
                   cudaStream_t stream) {
    const int outH = H - 2;
    const int outW = W - 2;

    if (outH <= 0 || outW <= 0) {
        CUDA_CHECK(cudaMemsetAsync(out.data().get(), 0, out.size() * sizeof(float), stream));
        return;
    }

    // Fallback for small channel sizes
    if (C < 16) {
        const int threads_per_block = 256;
        const int tiles_w = (outW + 1) / 2;
        const int tiles_h = (outH + 1) / 2;
        const int num_tiles = N * K * tiles_h * tiles_w;
        int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

        winograd_conv_fused_kernel_2x2<<<grid_size, threads_per_block, 0, stream>>>(
            image.data().get(), filter.data().get(), out.data().get(),
            N, C, H, W, K, outH, outW
        );
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto divUp = [](int x, int y) { return (x + y - 1) / y; };
    const int tiles_h = divUp(outH, 4);
    const int tiles_w = divUp(outW, 4);
    const int P = N * tiles_h * tiles_w;

    if (P == 0) {
        CUDA_CHECK(cudaMemsetAsync(out.data().get(), 0, out.size() * sizeof(float), stream));
        return;
    }

    size_t required_U_size = (size_t)36 * K * C;
    size_t required_V_size = (size_t)36 * C * P;
    size_t required_M_size = (size_t)36 * K * P;

    if (U.size() < required_U_size) U.resize(required_U_size);
    if (V.size() < required_V_size) V.resize(required_V_size);
    if (M.size() < required_M_size) M.resize(required_M_size);

    using namespace winograd_4x4_kernels;

    // Common GEMM operator
    using GemmBatched = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float
    >;
    GemmBatched gemm_op;
    cutlass::gemm::GemmCoord problem_size(K, P, C);
    typename GemmBatched::Arguments arguments(
        problem_size,
        {U.data().get(), C}, (long long)K * C,
        {V.data().get(), P}, (long long)C * P,
        {M.data().get(), P}, (long long)K * P,
        {M.data().get(), P}, (long long)K * P,
        {1.0f, 0.0f}, 36
    );

    // =================================================================================
    // ================= HEURISTIC-BASED PARAMETER DISPATCHER ==========================
    // Check for specific layer shapes and launch a complete set of tuned kernels.
    // N value here is N_per_gpu (Total_N / 2)
    // =================================================================================

    // Layer 0: C=3, H=112, W=112, K=64, N_total=64
    if (C == 3 && H == 112 && W == 112 && K == 64 && N == 32) {
        printf("INFO: Using heuristic config for layer_0 (C=3, H=112, K=64, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 64, 128><<<divUp(K * C, 64), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 8), divUp(tiles_h, 2), N * C);
        transform_input_BTdB<float, float, float, 2, 8, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 64, 128><<<divUp(K * P, 64), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 1: C=32, H=112, W=112, K=64, N_total=64
    else if (C == 32 && H == 112 && W == 112 && K == 64 && N == 32) {
        printf("INFO: Using heuristic config for layer_1 (C=32, H=112, K=64, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 40, 128><<<divUp(K * C, 40), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 8), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 8, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 128><<<divUp(K * P, 16), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 2: C=64, H=112, W=112, K=64, N_total=64
    else if (C == 64 && H == 112 && W == 112 && K == 64 && N == 32) {
        printf("INFO: Using heuristic config for layer_2 (C=64, H=112, K=64, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 20, 256><<<divUp(K * C, 20), 256, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 10), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 10, 224><<<grid_size_input, 224, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 20, 96><<<divUp(K * P, 20), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 3: C=64, H=112, W=112, K=128, N_total=64
    else if (C == 64 && H == 112 && W == 112 && K == 128 && N == 32) {
        printf("INFO: Using heuristic config for layer_3 (C=64, H=112, K=128, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 56, 160><<<divUp(K * C, 56), 160, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 8), divUp(tiles_h, 6), N * C);
        transform_input_BTdB<float, float, float, 6, 8, 160><<<grid_size_input, 160, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 24, 96><<<divUp(K * P, 24), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 4: C=128, H=112, W=112, K=128, N_total=64
    else if (C == 128 && H == 112 && W == 112 && K == 128 && N == 32) {
        printf("INFO: Using heuristic config for layer_4 (C=128, H=112, K=128, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 72, 224><<<divUp(K * C, 72), 224, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 8), divUp(tiles_h, 6), N * C);
        transform_input_BTdB<float, float, float, 6, 8, 160><<<grid_size_input, 160, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 64, 96><<<divUp(K * P, 64), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 5: C=128, H=50, W=50, K=256, N_total=64
    else if (C == 128 && H == 50 && W == 50 && K == 256 && N == 32) {
        printf("INFO: Using heuristic config for layer_5 (C=128, H=50, K=256, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 64, 224><<<divUp(K * C, 64), 224, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 6), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 6, 96><<<grid_size_input, 96, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 96><<<divUp(K * P, 16), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 6: C=256, H=50, W=50, K=256, N_total=64
    else if (C == 256 && H == 50 && W == 50 && K == 256 && N == 32) {
        printf("INFO: Using heuristic config for layer_6 (C=256, H=50, K=256, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 72, 128><<<divUp(K * C, 72), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 6), divUp(tiles_h, 2), N * C);
        transform_input_BTdB<float, float, float, 2, 6, 128><<<grid_size_input, 128, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 96><<<divUp(K * P, 16), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 7: C=256, H=50, W=50, K=512, N_total=64
    else if (C == 256 && H == 50 && W == 50 && K == 512 && N == 32) {
        printf("INFO: Using heuristic config for layer_7 (C=256, H=50, K=512, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 16, 256><<<divUp(K * C, 16), 256, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 2), N * C);
        transform_input_BTdB<float, float, float, 2, 4, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 128><<<divUp(K * P, 16), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 8: C=512, H=50, W=50, K=512, N_total=64
    else if (C == 512 && H == 50 && W == 50 && K == 512 && N == 32) {
        printf("INFO: Using heuristic config for layer_8 (C=512, H=50, K=512, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 40, 96><<<divUp(K * C, 40), 96, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 6), N * C);
        transform_input_BTdB<float, float, float, 6, 4, 160><<<grid_size_input, 160, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 24, 96><<<divUp(K * P, 24), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 9: C=512, H=16, W=16, K=2048, N_total=64
    else if (C == 512 && H == 16 && W == 16 && K == 2048 && N == 32) {
        printf("INFO: Using heuristic config for layer_9 (C=512, H=16, K=2048, N_total=64)\n");
        transform_filter_GgGT<float, float, float, 32, 128><<<divUp(K * C, 32), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 4, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 32, 128><<<divUp(K * P, 32), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 10: C=3, H=100, W=100, K=32, N_total=128
    else if (C == 3 && H == 100 && W == 100 && K == 32 && N == 64) {
        printf("INFO: Using heuristic config for layer_10 (C=3, H=100, K=32, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 16, 128><<<divUp(K * C, 16), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 8), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 8, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 32, 128><<<divUp(K * P, 32), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 11: C=32, H=100, W=100, K=64, N_total=128
    else if (C == 32 && H == 100 && W == 100 && K == 64 && N == 64) {
        printf("INFO: Using heuristic config for layer_11 (C=32, H=100, K=64, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 12, 96><<<divUp(K * C, 12), 96, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 6), N * C);
        transform_input_BTdB<float, float, float, 6, 4, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 224><<<divUp(K * P, 16), 224, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 12: C=64, H=50, W=50, K=64, N_total=128
    else if (C == 64 && H == 50 && W == 50 && K == 64 && N == 64) {
        printf("INFO: Using heuristic config for layer_12 (C=64, H=50, K=64, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 48, 256><<<divUp(K * C, 48), 256, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 2), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 2, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 128><<<divUp(K * P, 16), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 13: C=64, H=50, W=50, K=128, N_total=128
    else if (C == 64 && H == 50 && W == 50 && K == 128 && N == 64) {
        printf("INFO: Using heuristic config for layer_13 (C=64, H=50, K=128, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 64, 128><<<divUp(K * C, 64), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 2), N * C);
        transform_input_BTdB<float, float, float, 2, 4, 256><<<grid_size_input, 256, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 64, 128><<<divUp(K * P, 64), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 14: C=128, H=20, W=26, K=96, N_total=128
    else if (C == 128 && H == 20 && W == 26 && K == 96 && N == 64) {
        printf("INFO: Using heuristic config for layer_14 (C=128, H=20, K=96, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 40, 160><<<divUp(K * C, 40), 160, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 2), divUp(tiles_h, 2), N * C);
        transform_input_BTdB<float, float, float, 2, 2, 96><<<grid_size_input, 96, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 40, 96><<<divUp(K * P, 40), 96, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 15: C=96, H=12, W=12, K=192, N_total=128
    else if (C == 96 && H == 12 && W == 12 && K == 192 && N == 64) {
        printf("INFO: Using heuristic config for layer_15 (C=96, H=12, K=192, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 32, 128><<<divUp(K * C, 32), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 4, 128><<<grid_size_input, 128, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 128, 256><<<divUp(K * P, 128), 256, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 16: C=192, H=12, W=12, K=256, N_total=128
    else if (C == 192 && H == 12 && W == 12 && K == 256 && N == 64) {
        printf("INFO: Using heuristic config for layer_16 (C=192, H=12, K=256, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 16, 128><<<divUp(K * C, 16), 128, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 2), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 2, 128><<<grid_size_input, 128, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 64, 256><<<divUp(K * P, 64), 256, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Layer 17: C=256, H=8, W=8, K=512, N_total=128
    else if (C == 256 && H == 8 && W == 8 && K == 512 && N == 64) {
        printf("INFO: Using heuristic config for layer_17 (C=256, H=8, K=512, N_total=128)\n");
        transform_filter_GgGT<float, float, float, 16, 256><<<divUp(K * C, 16), 256, 0, stream>>>(U.data().get(), filter.data().get(), C, K);
        dim3 grid_size_input(divUp(tiles_w, 4), divUp(tiles_h, 4), N * C);
        transform_input_BTdB<float, float, float, 4, 4, 128><<<grid_size_input, 128, 0, stream>>>(V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);
        gemm_op(arguments, nullptr, stream);
        transform_output_ATtA<float, float, float, 16, 128><<<divUp(K * P, 16), 128, 0, stream>>>(out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // =================================================================================
    // ================= FALLBACK/DEFAULT KERNEL LAUNCH ================================
    // This part will only be executed if no heuristic matched the layer shape.
    // =================================================================================
    printf("INFO: Using default config for shape (C=%d, H=%d, W=%d, K=%d, N=%d)\n", C, H, W, K, N);

    // 1. Default Filter Transform
    const auto grid_size_filter = divUp(K * C, TUNABLE_FILTER_NUM_KERNELS_PER_BLOCK);
    transform_filter_GgGT<float, float, float, TUNABLE_FILTER_NUM_KERNELS_PER_BLOCK, TUNABLE_FILTER_BLOCK_SIZE><<<grid_size_filter, TUNABLE_FILTER_BLOCK_SIZE, 0, stream>>>(
        U.data().get(), filter.data().get(), C, K);

    // 2. Default Input Transform
    dim3 grid_size_input;
    grid_size_input.x = divUp(tiles_w, TUNABLE_INPUT_TILES_X_PER_BLOCK);
    grid_size_input.y = divUp(tiles_h, TUNABLE_INPUT_TILES_Y_PER_BLOCK);
    grid_size_input.z = N * C;
    transform_input_BTdB<float, float, float, TUNABLE_INPUT_TILES_Y_PER_BLOCK, TUNABLE_INPUT_TILES_X_PER_BLOCK, TUNABLE_INPUT_BLOCK_SIZE><<<grid_size_input, TUNABLE_INPUT_BLOCK_SIZE, 0, stream>>>(
        V.data().get(), image.data().get(), C, H, W, 0, 0, N, tiles_h, tiles_w);

    // 3. Default Batched GEMM
    gemm_op(arguments, nullptr, stream);

    // 4. Default Output Transform
    const auto grid_size_output = divUp(K * P, TUNABLE_OUTPUT_NUM_TILES_PER_BLOCK);
    transform_output_ATtA<float, float, float, TUNABLE_OUTPUT_NUM_TILES_PER_BLOCK, TUNABLE_OUTPUT_BLOCK_SIZE><<<grid_size_output, TUNABLE_OUTPUT_BLOCK_SIZE, 0, stream>>>(
        out.data().get(), M.data().get(), N, K, outH, outW, tiles_h, tiles_w);

    CUDA_CHECK(cudaGetLastError());
}