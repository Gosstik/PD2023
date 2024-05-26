#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= height * width) {
    return;
  }

  int row = tid / width;
  int col = tid % width;

  float* res = (float*) ((char*) result + row * pitch) + col;
  float* a = (float*) ((char*) A + row * pitch) + col;
  float* b = (float*) ((char*) B + row * pitch) + col;
  *res = *a + *b;
}
