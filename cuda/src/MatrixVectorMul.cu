#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
  // 2 dimensions.
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= height || col >= width) {
    return;
  }

  atomicAdd(result + row, matrix[row * width + col] * vector[col]);
}

