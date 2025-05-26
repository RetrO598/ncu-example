#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Kernel A: 每个线程读取 1 个 float
__global__ void kernel_A(const float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx];
  }
}

// Kernel B: 每个线程读取 2 个 float（可能未对齐）
__global__ void kernel_B(const float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx * 2;
  if (i + 1 < N) {
    output[i] = input[i];
    output[i + 1] = input[i + 1];
  }
}

// Kernel B_aligned: 每个线程读取一个 float2（对齐读取）
__global__ void kernel_B_aligned(const float2 *input, float2 *output,
                                 int N_vec2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N_vec2) {
    output[idx] = input[idx];
  }
}

// 通用计时函数
float run_kernel_A(int threads_per_block, int N) {
  int blocks = (N + threads_per_block - 1) / threads_per_block;
  float *d_input, *d_output;
  CHECK(cudaMalloc(&d_input, sizeof(float) * N));
  CHECK(cudaMalloc(&d_output, sizeof(float) * N));

  std::vector<float> h_input(N);
  for (int i = 0; i < N; ++i)
    h_input[i] = static_cast<float>(i);
  CHECK(cudaMemcpy(d_input, h_input.data(), sizeof(float) * N,
                   cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));

  kernel_A<<<blocks, threads_per_block>>>(d_input, d_output, N);
  CHECK(cudaGetLastError());

  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms;
}

float run_kernel_B(int threads_per_block, int N) {
  int threads = N / 2;
  int blocks = (threads + threads_per_block - 1) / threads_per_block;
  float *d_input, *d_output;
  CHECK(cudaMalloc(&d_input, sizeof(float) * N));
  CHECK(cudaMalloc(&d_output, sizeof(float) * N));

  std::vector<float> h_input(N);
  for (int i = 0; i < N; ++i)
    h_input[i] = static_cast<float>(i);
  CHECK(cudaMemcpy(d_input, h_input.data(), sizeof(float) * N,
                   cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));

  kernel_B<<<blocks, threads_per_block>>>(d_input, d_output, N);
  CHECK(cudaGetLastError());

  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms;
}

float run_kernel_B_aligned(int threads_per_block, int N_float) {
  int N_vec2 = N_float / 2;
  int blocks = (N_vec2 + threads_per_block - 1) / threads_per_block;

  float2 *d_input, *d_output;
  CHECK(cudaMalloc(&d_input, sizeof(float2) * N_vec2));
  CHECK(cudaMalloc(&d_output, sizeof(float2) * N_vec2));

  std::vector<float> h_input(N_float);
  for (int i = 0; i < N_float; ++i)
    h_input[i] = static_cast<float>(i);

  std::vector<float2> h_input_vec(N_vec2);
  for (int i = 0; i < N_vec2; ++i) {
    h_input_vec[i].x = h_input[i * 2];
    h_input_vec[i].y = h_input[i * 2 + 1];
  }

  CHECK(cudaMemcpy(d_input, h_input_vec.data(), sizeof(float2) * N_vec2,
                   cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));

  kernel_B_aligned<<<blocks, threads_per_block>>>(d_input, d_output, N_vec2);
  CHECK(cudaGetLastError());

  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms;
}

int main() {
  const int total_MB = 100;
  const int total_elements = (total_MB * 1024 * 1024) / sizeof(float);
  const int threads_per_block = 256;

  std::cout << "Total data: " << total_MB << " MB (" << total_elements
            << " floats)\n";

  float time_A = run_kernel_A(threads_per_block, total_elements);
  float time_B = run_kernel_B(threads_per_block, total_elements);
  float time_B_aligned =
      run_kernel_B_aligned(threads_per_block, total_elements);

  size_t total_bytes = total_elements * sizeof(float);
  float bw_A = total_bytes / (time_A / 1000.0f) / (1024 * 1024 * 1024);
  float bw_B = total_bytes / (time_B / 1000.0f) / (1024 * 1024 * 1024);
  float bw_B_aligned =
      total_bytes / (time_B_aligned / 1000.0f) / (1024 * 1024 * 1024);

  std::cout << "\n--- Results ---\n";
  std::cout << "[Kernel A]         Threads: " << total_elements
            << ", Time: " << time_A << " ms, Bandwidth: " << bw_A << " GB/s\n";
  std::cout << "[Kernel B]         Threads: " << total_elements / 2
            << ", Time: " << time_B << " ms, Bandwidth: " << bw_B << " GB/s\n";
  std::cout << "[Kernel B_aligned] Threads: " << total_elements / 2
            << ", Time: " << time_B_aligned
            << " ms, Bandwidth: " << bw_B_aligned << " GB/s\n";

  return 0;
}
