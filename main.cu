#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>
#include <thread> // 用于启动多线程
#include <chrono> // 用于主机端计时
#include <cassert> // 用于断言
#include <algorithm> // for std::max
#include <nccl.h>  // 引入 NCCL 头文件

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "winograd.cuh"

#define LOOP_NUM 3
#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

// 定义一个CUDA API调用错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 定义一个NCCL API调用错误检查宏
#define NCCL_CHECK(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL Error in %s at line %d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


class Config {
public:
    std::vector<int> C, H, W, K, Batch;
    int layer_num;

    Config(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filename);
        }

        file >> layer_num;
        C.resize(layer_num);
        H.resize(layer_num);
        W.resize(layer_num);
        K.resize(layer_num);
        Batch.resize(layer_num);

        for (int i = 0; i < layer_num; i++) {
            file >> C[i] >> H[i] >> W[i] >> K[i] >> Batch[i];
            if (Batch[i] % 2 != 0) {
                 printf("Warning: Batch size for layer %d (%d) is not even. Adjusting to %d for 2 GPUs.\n", i, Batch[i], Batch[i] - 1);
                 Batch[i] -= 1;
                 if(Batch[i] == 0) {
                     fprintf(stderr, "Error: Batch size for layer %d is 0 after adjustment. Please use even batch sizes.\n", i);
                     exit(EXIT_FAILURE);
                 }
            }
        }
    }
};

class LayerResult {
public:
    double baseline_time = 0.0;
    double winograd_time = 0.0;
    double baseline_gflops = 0.0;
    double winograd_gflops = 0.0;
    double speedup = 0.0;
    bool passed = false;
};


// 为每个GPU执行单次卷积的工作函数（用于预热）
void winograd_conv_worker(int gpu_id, int N_per_gpu, int H, int W, int C, int K,
                          thrust::device_vector<float>& d_image_slice,
                          thrust::device_vector<float>& d_filter,
                          thrust::device_vector<float>& d_out_slice,
                          thrust::device_vector<float>& d_U,
                          thrust::device_vector<float>& d_V,
                          thrust::device_vector<float>& d_M,
                          cudaStream_t stream) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    winograd_conv(d_image_slice, d_filter, d_out_slice, d_U, d_V, d_M, H, W, C, K, N_per_gpu, stream);
}

// 为每个GPU执行循环卷积的工作函数（用于计时）
void winograd_conv_looped_worker(int gpu_id, int N_per_gpu, int H, int W, int C, int K,
                                 thrust::device_vector<float>& d_image_slice,
                                 thrust::device_vector<float>& d_filter,
                                 thrust::device_vector<float>& d_out_slice,
                                 thrust::device_vector<float>& d_U,
                                 thrust::device_vector<float>& d_V,
                                 thrust::device_vector<float>& d_M,
                                 cudaStream_t stream) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    for (int i = 0; i < LOOP_NUM; ++i) {
        winograd_conv(d_image_slice, d_filter, d_out_slice, d_U, d_V, d_M, H, W, C, K, N_per_gpu, stream);
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    int num_gpus = 2;
    ncclComm_t comms[num_gpus];
    cudaStream_t streams[num_gpus];
    int devs[num_gpus];

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < num_gpus) {
        fprintf(stderr, "Error: System has %d GPUs, but %d are required.\n", device_count, num_gpus);
        return 1;
    }

    for(int i = 0; i < num_gpus; ++i) {
        devs[i] = i;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));
    CUDA_CHECK(cudaSetDevice(0));
    printf("NCCL communicators initialized for %d GPUs.\n", num_gpus);
    printf("============================================================\n\n");

    auto cfg = std::make_unique<Config>(argv[1]);
    std::vector<LayerResult> results(cfg->layer_num);

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    float milliseconds = 0;

    thrust::host_vector<thrust::host_vector<float>> h_images(cfg->layer_num);
    thrust::host_vector<thrust::host_vector<float>> h_filters(cfg->layer_num);

    std::random_device rd;
    thrust::default_random_engine rng(rd());
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9;
        h_images[l].resize(N * C * sizeI);
        h_filters[l].resize(K * C * sizeF);
        thrust::generate(h_images[l].begin(), h_images[l].end(), [&] { return dist(rng); });
        thrust::generate(h_filters[l].begin(), h_filters[l].end(), [&] { return dist(rng); });
    }

    thrust::host_vector<thrust::host_vector<float>> h_baseline_outputs(cfg->layer_num);
    thrust::host_vector<thrust::host_vector<float>> h_custom_outputs(cfg->layer_num);

    // =======================================================================
    // ======== Baseline Test (Single GPU) with Pre-allocation =============
    // =======================================================================
    std::cout << "=== Running Naive Convolution (Baseline on a single GPU) ===" << std::endl;
    double baseline_total_time = 0;
    long total_flops = 0;

    { // Scoping to manage buffer lifetimes
        CUDA_CHECK(cudaSetDevice(0));
        size_t max_img_size = 0, max_flt_size = 0, max_out_size = 0;
        for (int l = 0; l < cfg->layer_num; l++) {
            max_img_size = std::max(max_img_size, (size_t)cfg->Batch[l] * cfg->C[l] * cfg->H[l] * cfg->W[l]);
            max_flt_size = std::max(max_flt_size, (size_t)cfg->K[l] * cfg->C[l] * 9);
            max_out_size = std::max(max_out_size, (size_t)cfg->Batch[l] * cfg->K[l] * (cfg->H[l] - 2) * (cfg->W[l] - 2));
        }

        thrust::device_vector<float> d_image(max_img_size);
        thrust::device_vector<float> d_filter(max_flt_size);
        thrust::device_vector<float> d_out(max_out_size);
        thrust::device_vector<float> d_U, d_V, d_M; // Not used by naive, but needed for interface

        for (int l = 0; l < cfg->layer_num; l++) {
            int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
            long img_size = (long)N * C * H * W;
            long flt_size = (long)K * C * 9;
            long out_size = (long)N * K * (H - 2) * (W - 2);

            d_image.resize(img_size);
            d_filter.resize(flt_size);
            d_out.resize(out_size);

            CUDA_CHECK(cudaMemcpy(d_image.data().get(), h_images[l].data(), img_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_filter.data().get(), h_filters[l].data(), flt_size * sizeof(float), cudaMemcpyHostToDevice));

            naive_conv(d_image, d_filter, d_out, d_U, d_V, d_M, H, W, C, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start_event, 0));
            for (int i = 0; i < LOOP_NUM; i++) {
                naive_conv(d_image, d_filter, d_out, d_U, d_V, d_M, H, W, C, K, N);
            }
            CUDA_CHECK(cudaEventRecord(stop_event, 0));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
            results[l].baseline_time = static_cast<double>(milliseconds / 1000.0f) / LOOP_NUM;

            h_baseline_outputs[l].resize(out_size);
            CUDA_CHECK(cudaMemcpy(h_baseline_outputs[l].data(), d_out.data().get(), out_size * sizeof(float), cudaMemcpyDeviceToHost));

            long flops = static_cast<long>(N) * K * C * (H-2) * (W-2) * 18;
            results[l].baseline_gflops = flops * 1e-9 / results[l].baseline_time;

            std::cout << "Layer " << std::setw(2) << l << ": " << std::fixed << std::setprecision(3)
                      << results[l].baseline_time * 1000 << " ms (" << std::setprecision(2) << results[l].baseline_gflops << " GFLOPS)" << std::endl;
            baseline_total_time += results[l].baseline_time;
            total_flops += flops;
        }
    }
    std::cout << "Baseline Total: " << std::fixed << std::setprecision(3) << baseline_total_time * 1000
              << " ms (" << std::setprecision(2) << total_flops * 1e-9 / baseline_total_time << " GFLOPS)" << std::endl;

    // =======================================================================
    // ======== Winograd Test (2 GPUs) with Pre-allocation ===================
    // =======================================================================
    std::cout << "\n=== Running Winograd Convolution (2 GPUs with NCCL) ===" << std::endl;
    double winograd_total_time = 0;

    { // Scoping to manage buffer lifetimes
        size_t max_U_size = 0, max_V_size = 0, max_M_size = 0;
        size_t max_img_slice = 0, max_flt_size = 0, max_out_slice = 0, max_out_full = 0;
        auto divUp = [](int x, int y) { return (x + y - 1) / y; };

        for (int l = 0; l < cfg->layer_num; l++) {
            int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
            int N_per_gpu = N / num_gpus;
            int outH = H > 2 ? H - 2 : 0;
            int outW = W > 2 ? W - 2 : 0;

            max_img_slice = std::max(max_img_slice, (size_t)N_per_gpu * C * H * W);
            max_flt_size = std::max(max_flt_size, (size_t)K * C * 9);
            max_out_slice = std::max(max_out_slice, (size_t)N_per_gpu * K * outH * outW);
            max_out_full = std::max(max_out_full, (size_t)N * K * outH * outW);

            if (C >= 16 && outH > 0 && outW > 0) {
                int tiles_h = divUp(outH, 4);
                int tiles_w = divUp(outW, 4);
                int P = N_per_gpu * tiles_h * tiles_w;
                max_U_size = std::max(max_U_size, (size_t)36 * K * C);
                max_V_size = std::max(max_V_size, (size_t)36 * C * P);
                max_M_size = std::max(max_M_size, (size_t)36 * K * P);
            }
        }

        thrust::device_vector<float> d_image_slices[num_gpus];
        thrust::device_vector<float> d_filters[num_gpus];
        thrust::device_vector<float> d_out_slices[num_gpus];
        thrust::device_vector<float> d_out_full[num_gpus];
        thrust::device_vector<float> d_Us[num_gpus], d_Vs[num_gpus], d_Ms[num_gpus];

        for(int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            d_image_slices[i].resize(max_img_slice);
            d_filters[i].resize(max_flt_size);
            d_out_slices[i].resize(max_out_slice);
            d_out_full[i].resize(max_out_full);
            d_Us[i].resize(max_U_size);
            d_Vs[i].resize(max_V_size);
            d_Ms[i].resize(max_M_size);
        }

        for (int l = 0; l < cfg->layer_num; l++) {
            int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
            int N_per_gpu = N / num_gpus;
            long outH = H > 2 ? H - 2 : 0;
            long outW = W > 2 ? W - 2 : 0;
            long image_slice_size = (long)N_per_gpu * C * H * W;
            long filter_size = (long)K * C * 9;
            long output_slice_size = (long)N_per_gpu * K * outH * outW;
            long full_output_size = (long)N * K * outH * outW;

            for(int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                d_image_slices[i].resize(image_slice_size);
                d_filters[i].resize(filter_size);
                d_out_slices[i].resize(output_slice_size);

                CUDA_CHECK(cudaMemcpyAsync(d_image_slices[i].data().get(), h_images[l].data() + i * image_slice_size, image_slice_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
                CUDA_CHECK(cudaMemcpyAsync(d_filters[i].data().get(), h_filters[l].data(), filter_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
            }

            std::vector<std::thread> warmup_threads;
            for (int i = 0; i < num_gpus; ++i) {
                warmup_threads.emplace_back(winograd_conv_worker, i, N_per_gpu, H, W, C, K,
                                            std::ref(d_image_slices[i]), std::ref(d_filters[i]), std::ref(d_out_slices[i]),
                                            std::ref(d_Us[i]), std::ref(d_Vs[i]), std::ref(d_Ms[i]), streams[i]);
            }
            for (auto& t : warmup_threads) t.join();
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> compute_threads;
            for (int i = 0; i < num_gpus; ++i) {
                compute_threads.emplace_back(winograd_conv_looped_worker, i, N_per_gpu, H, W, C, K,
                                            std::ref(d_image_slices[i]), std::ref(d_filters[i]), std::ref(d_out_slices[i]),
                                            std::ref(d_Us[i]), std::ref(d_Vs[i]), std::ref(d_Ms[i]), streams[i]);
            }
            for (auto& t : compute_threads) t.join();
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            results[l].winograd_time = elapsed.count() / LOOP_NUM;

            NCCL_CHECK(ncclGroupStart());
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                d_out_full[i].resize(full_output_size);
                NCCL_CHECK(ncclAllGather(d_out_slices[i].data().get(), d_out_full[i].data().get(),
                                        output_slice_size, ncclFloat, comms[i], streams[i]));
            }
            NCCL_CHECK(ncclGroupEnd());
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            }

            h_custom_outputs[l].resize(full_output_size);
            if (full_output_size > 0) {
                 CUDA_CHECK(cudaSetDevice(0));
                 CUDA_CHECK(cudaMemcpy(h_custom_outputs[l].data(), d_out_full[0].data().get(), full_output_size * sizeof(float), cudaMemcpyDeviceToHost));
            }

            long flops = static_cast<long>(N) * K * C * outH * outW * 18;
            results[l].winograd_gflops = flops * 1e-9 / results[l].winograd_time;
            results[l].speedup = results[l].baseline_time / results[l].winograd_time;

            std::cout << "Layer " << std::setw(2) << l << ": " << std::fixed << std::setprecision(3)
                      << results[l].winograd_time * 1000 << " ms (" << std::setprecision(2) << results[l].winograd_gflops << " GFLOPS)" << std::endl;
            winograd_total_time += results[l].winograd_time;
        }
    }

    std::cout << "Winograd Total (2 GPUs): " << std::fixed << std::setprecision(3) << winograd_total_time * 1000
              << " ms (" << std::setprecision(2) << total_flops * 1e-9 / winograd_total_time << " GFLOPS)" << std::endl;

    std::cout << "\n=== Correctness Check ===" << std::endl;
    bool all_correct = true;
    for (int l = 0; l < cfg->layer_num; l++) {
        long out_size = h_baseline_outputs[l].size();
        results[l].passed = true;
        if (out_size == 0) { // Handle cases with no output
            std::cout << "Layer " << std::setw(2) << l << ": SKIPPED (no output)" << std::endl;
            continue;
        }
        for (long i = 0; i < out_size; i++) {
            if (std::abs((h_custom_outputs[l][i] - h_baseline_outputs[l][i]) / (h_baseline_outputs[l][i] + 1e-7f)) > 1e-3f) {
                results[l].passed = false;
                all_correct = false;
                fprintf(stderr, "Layer %d Mismatch at index %ld: Baseline=%.6f, 2-GPU=%.6f\n",
                        l, i, h_baseline_outputs[l][i], h_custom_outputs[l][i]);
                break;
            }
        }

        std::cout << "Layer " << std::setw(2) << l << ": "
                  << (results[l].passed ? GREEN "CORRECT" RESET : RED "INCORRECT" RESET)
                  << " (Speedup: " << std::fixed << std::setprecision(2) << results[l].speedup << "x)" << std::endl;
    }

    double overall_speedup = baseline_total_time / winograd_total_time;
    std::cout << "\n=== Final Results ===" << std::endl;
    if (all_correct) {
        std::cout << GREEN "All layers passed correctness check!" RESET << std::endl;
    } else {
        std::cout << RED "One or more layers failed correctness check!" RESET << std::endl;
    }

    std::cout << "Baseline Total (1 GPU): " << std::fixed << std::setprecision(3) << baseline_total_time * 1000
              << " ms (" << std::setprecision(2) << total_flops * 1e-9 / baseline_total_time << " GFLOPS)" << std::endl;
    std::cout << "Winograd Total (2 GPUs): " << std::fixed << std::setprecision(3) << winograd_total_time * 1000
              << " ms (" << std::setprecision(2) << total_flops * 1e-9 / winograd_total_time << " GFLOPS)" << std::endl;
    std::cout << "Overall Speedup: " << std::fixed << std::setprecision(2) << overall_speedup << "x" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    for(int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        // No need to synchronize streams here, already done in loops
        ncclCommDestroy(comms[i]);
        cudaStreamDestroy(streams[i]);
    }

    return all_correct ? 0 : -1;
}