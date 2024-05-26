#include <MatrixMul.cuh>

/**
 * @property
 * res_height = blockDim.x * gridDim.x \n
 * res_width = blockDim.y * gridDim.y \n
 * shared_memory.size = 2 * blockDim.x * blockDim.y \n
 * blockDim.x == blockDim.y
 */
__global__
void MatrixMul(int a_height, int a_width, int b_width, float* a, float* b, float* res) {
  // Shared data.
  extern __shared__ float float_shared_data[];
  int b_pad = blockDim.x * blockDim.y;
  int shared_a_i = threadIdx.x * blockDim.y + threadIdx.y;
  int shared_b_i = b_pad + shared_a_i;

  // Indexes of result matrix.
  int res_i = blockIdx.x * blockDim.x + threadIdx.x;
  int res_j = blockIdx.y * blockDim.y + threadIdx.y;

  // Matrix.
  res[res_i * b_width + res_j] = 0.f;

  int iters_count = (a_width + blockDim.y - 1) / blockDim.y;
  for (int d = 0; d < iters_count; ++d) {
    // Copy A.
    float_shared_data[shared_a_i] = a[res_i * a_width + threadIdx.y + d * blockDim.y];
    // Copy B.
    float_shared_data[shared_b_i] = b[blockIdx.x * b_width + res_j + d * blockDim.x];
    // Result.
    float res_f = 0.f;

    __syncthreads();

    // Multiplication inside block.
//    for (int k = 0; k < blockDim.y; ++k) {
//      res_f += float_shared_data[threadIdx.x * blockDim.y + k] *
//               float_shared_data[b_pad + k * blockDim.y + threadIdx.y];
//    }
    for (int k = 0; k < blockDim.y; ++k) {
      res_f += float_shared_data[threadIdx.x * blockDim.y + k] *
          float_shared_data[b_pad + k * blockDim.x + threadIdx.y];
    }

    res[res_i * b_width + res_j] += res_f;

    __syncthreads();
  }
}





//
///**
// * @remarks
// * width_b == blockDim.y * gridDim.y \n
// * gridDim.x * blockDim.x == "result.height" \n
// * gridDim.y * gridDim.y == "result.width" \n
// */
//__global__
//void MatrixMul(int height_a, int width_a, int width_b, float* a, float* b, float* result) {
//  extern __shared__ float float_shared_data[];
//
//  // Compute indexes in source matrices.
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  // int height = blockDim.x * gridDim.x;
//  int width = blockDim.y * gridDim.y;
//
//  if (i >= height_a || j >= width_b) {
//    return;
//  }
//
//  // Compute node in float_shared_data.
//  int shared_width = blockDim.x;
////  int shared_height = blockDim.y;
//
//  int shared_i = threadIdx.x;
//  int shared_j = threadIdx.y;
//
//  float* shared_node = &float_shared_data[shared_i * shared_width + shared_j];
//
//  for (int k = 0; k < width_a; ++k) {
//    *shared_node += a[i * width + k] * b[k * width + j];
//  }
//
//  result[i * width + j] = *shared_node;
//}

//__global__
//void MatrixMul(float* A, float* B, float* C, int mid_size) {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  int height = blockDim.x * gridDim.x;
//  int width = blockDim.y * gridDim.y;
//
//  C[i * width + j] = .0f;
//
//  for (int k = 0; k < mid_size; ++k) {
//    C[i * width + j] += A[i * mid_size + k] * B[k * width + j];
//  }
//}
