#include "kernels.cuh"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <helper_functions.h>
#include <vector_types.h>

constexpr int N = 512;
constexpr int NTIMES = 10;

void stencilCPU(float *in, float *out, int N) {
  for (int k = 1; k < N - 1; ++k) {
    for (int j = 1; j < N - 1; ++j) {
      for (int i = 1; i < N - 1; ++i) {
        out[i + j * N + k * N * N] = c0 * in[i + j * N + k * N * N] +
                                     c1 * in[(i - 1) + j * N + k * N * N] +
                                     c2 * in[(i + 1) + j * N + k * N * N] +
                                     c3 * in[i + (j - 1) * N + k * N * N] +
                                     c4 * in[i + (j + 1) * N + k * N * N] +
                                     c5 * in[i + j * N + (k - 1) * N * N] +
                                     c6 * in[i + j * N + (k + 1) * N * N];
      }
    }
  }
}

bool compareResults(float *cpu, float *gpu, int N) {
  for (int i = 0; i < N * N * N; ++i) {
    if (fabs(cpu[i] - gpu[i]) > 1e-6) {
      printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu[i], gpu[i]);
      return false;
    }
  }
  return true;
}

void printfMat(float *mat, int N) {
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        printf("%f ", mat[i + j * N + k * N * N]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

int main() {
  size_t memsize = N * N * N * sizeof(float);
  float *in_h;
  float *out_h;
  float *in_d;
  float *out_d;

  in_h = (float *)malloc(memsize);
  out_h = (float *)malloc(memsize);
  float *out_h_gpu = (float *)malloc(memsize);

  for (int i = 0; i < N * N * N; ++i) {
    in_h[i] = static_cast<float>(rand()) / RAND_MAX;
    out_h[i] = 0.0f;
  }
  checkCudaErrors(cudaMalloc((void **)&in_d, memsize));
  checkCudaErrors(cudaMalloc((void **)&out_d, memsize));

  checkCudaErrors(cudaMemcpy(in_d, in_h, memsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(out_d, out_h, memsize, cudaMemcpyHostToDevice));

  stencilCPU(in_h, out_h, N);

  dim3 block;
  dim3 grid;
  auto kernel = &stencilNaive;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  for (int i = 0; i < 4; ++i) {
    if (i == 0) {
      printf("Testing Kernel stencilNaive......\n");
      block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
      grid = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    } else if (i == 1) {
      kernel = &stencilShared;
      block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
      grid = dim3((N + OUT_TILE - 1) / OUT_TILE, (N + OUT_TILE - 1) / OUT_TILE,
                  (N + OUT_TILE - 1) / OUT_TILE);
      printf("Testing Kernel stencilShared......\n");
    } else if (i == 2) {
      constexpr int blocksize = 16;
      constexpr int output_tile = blocksize - 2;
      kernel = &stencilThreadCoarsen<blocksize, output_tile>;
      block = dim3(blocksize, blocksize, 1);
      grid = dim3((N + output_tile - 1) / output_tile,
                  (N + output_tile - 1) / output_tile,
                  (N + output_tile - 1) / output_tile);
      printf("Testing Kernel stencilThreadCoarsen......\n");
    } else {
      constexpr int blocksize = 16;
      constexpr int output_tile = blocksize - 2;
      kernel = &stencilThreadCoarsenRegister<blocksize, output_tile>;
      block = dim3(blocksize, blocksize, 1);
      grid = dim3((N + output_tile - 1) / output_tile,
                  (N + output_tile - 1) / output_tile,
                  (N + output_tile - 1) / output_tile);
      printf("Testing Kernel stencilThreadCoarsenRegister......\n");
    }
    // warm up
    kernel<<<grid, block>>>(in_d, out_d, N);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(start, 0));
    for (int i = 0; i < NTIMES; ++i) {
      kernel<<<grid, block>>>(in_d, out_d, N);
      checkCudaErrors(cudaGetLastError());
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    printf("Elapsed time: %f ms\n", time / NTIMES);

    checkCudaErrors(
        cudaMemcpy(out_h_gpu, out_d, memsize, cudaMemcpyDeviceToHost));

    if (compareResults(out_h, out_h_gpu, N)) {
      printf("right answer\n");
    }
    printf("==================================================\n");
  }

  // printf("gpu result:\n");
  // printfMat(out_h_gpu, N);

  // printf("cpu result:\n");
  // printfMat(out_h, N);

  free(in_h);
  free(out_h);
  free(out_h_gpu);
  checkCudaErrors(cudaFree(in_d));
  checkCudaErrors(cudaFree(out_d));
  return 0;
}
