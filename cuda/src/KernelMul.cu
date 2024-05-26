#include <KernelMul.cuh>

__global__ void KernelMul(int num_elements, float* x, float* y, float* result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = index; i < num_elements; i += stride) {
    result[i] = x[i] * y[i];
  }
}
