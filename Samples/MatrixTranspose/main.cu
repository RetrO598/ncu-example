#include <cuda_runtime_api.h>
#include <helper_functions.h>

#include <iostream>
int main() {
  std::cout << "hello" << "\n";
  float *mat1;
  checkCudaErrors(cudaMalloc((void **)&mat1, sizeof(float) * 100));
}