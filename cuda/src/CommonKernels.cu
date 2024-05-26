#include <CommonKernels.cuh>



void GPUAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr,
            "GPUassert: %s %s %d\n",
            cudaGetErrorString(code),
            file,
            line);
    if (abort) {
      exit(code);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

size_t Log2(size_t num) {
  if (num < 2) {
    return 0;
  }
  size_t res = 0;
  while (num > 1) {
    ++res;
    num >>= 1;
  }
  return res;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void ScalarMulPerBlockSimple(int num_elements, float* vector1, float* vector2, float *result) {
  extern __shared__ float float_shared_data[];

  unsigned int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_elements) {
    return;
  }

  float_shared_data[tid] = vector1[index] * vector2[index];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float_shared_data[tid] += float_shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = float_shared_data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////

#define WARP_REDUCE_FULL_MASK unsigned(-1) // 0xffffffff

__device__ void WarpReduce(volatile float* shared_data, int tid) {
  shared_data[tid] += shared_data[tid + 32];
  float val = shared_data[tid];
  val += __shfl_down_sync(WARP_REDUCE_FULL_MASK, val, 16);
  val += __shfl_down_sync(WARP_REDUCE_FULL_MASK, val, 8);
  val += __shfl_down_sync(WARP_REDUCE_FULL_MASK, val, 4);
  val += __shfl_down_sync(WARP_REDUCE_FULL_MASK, val, 2);
  val += __shfl_down_sync(WARP_REDUCE_FULL_MASK, val, 1);
  shared_data[tid] = val;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void ScalarMulPerBlockWarpSpecific(int num_elements, float* vector1, float* vector2, float *result) {
  extern __shared__ float float_shared_data[];

  unsigned int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  unsigned int index2 = index + blockDim.x;

  if (index >= num_elements) {
    return;
  }
  if (index2 >= num_elements) {
    float_shared_data[tid] = vector1[index] * vector2[index];
  } else {
    float_shared_data[tid] = vector1[index] * vector2[index] +
                             vector1[index2] * vector2[index2];
  }

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      float_shared_data[tid] += float_shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    WarpReduce(float_shared_data, tid);
  }

  if (tid == 0) {
    result[blockIdx.x] = float_shared_data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * @warning In case num_elements < blockDim.x then float_shared_data fills
 * padding with zeros.
 */
__global__
void SumElementsInFirstBlock(int num_elements, float* vector, float* result) {
  // !!! gridDim.x == 1
  extern __shared__ float float_shared_data[];

  unsigned int tid = threadIdx.x;

  if (tid >= num_elements) {
    float_shared_data[tid] = 0;
    return;
  }

  float_shared_data[tid] = vector[tid];

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float_shared_data[tid] += float_shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[0] = float_shared_data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////

__global__
void ScalarMulSumToOneBlock(int num_elements, float* vector1, float* vector2, float *result) {
  // !!! gridDim.x == block_size
  extern __shared__ float float_shared_data[];

  unsigned int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if (index >= num_elements) {
    return;
  }

  float_shared_data[tid] = vector1[index] * vector2[index];

  __syncthreads();

  for (int i = index + stride; i < num_elements; i += stride) {
    float_shared_data[tid] += vector1[i] * vector2[i];
  }

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float_shared_data[tid] += float_shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = float_shared_data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////

__global__
void VectorSumSquareCoords(int num_elements, float* vector, float* result) {
  // !!! gridDim.x == block_size
  extern __shared__ float float_shared_data[];

  unsigned int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if (index >= num_elements) {
    return;
  }

  float_shared_data[tid] = vector[index] * vector[index];

  __syncthreads();

  for (int i = index + stride; i < num_elements; i += stride) {
    float_shared_data[tid] += vector[i] * vector[i];
  }

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float_shared_data[tid] += float_shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, float_shared_data[0]);
  }
}

////////////////////////////////////////////////////////////////////////////////

float VectorSquare2Norm(int num_elements, float* vector, int block_size) {
  // Host memory.
  float* res_arr = new float[1];

  // Device memory.
  int vec_byte_size = num_elements * sizeof(float);

  float* d_vector;
  cudaMalloc(&d_vector, vec_byte_size);

  float* d_block_res;
  int block_res_size = std::min(num_elements, block_size);
  cudaMalloc(&d_block_res, block_res_size * sizeof(float));

  // Move to device.
  cudaMemcpy(d_vector, vector, vec_byte_size, cudaMemcpyHostToDevice);

  // Function execution.
  VectorSumSquareCoords<<<
  block_size, block_size, block_size * sizeof(float)
  >>>(num_elements, d_vector, d_block_res);

  SumElementsInFirstBlock<<<
  1, block_size, block_size * sizeof(float)
  >>>(block_res_size, d_block_res, d_block_res);

  cudaMemcpy(res_arr, d_block_res, sizeof(float), cudaMemcpyDeviceToHost);

  cudaErrchk(cudaPeekAtLastError());

  float res = res_arr[0];

  // Free resources.
  cudaFree(d_vector);
  cudaFree(d_block_res);

  delete[] res_arr;

  return res;
}

////////////////////////////////////////////////////////////////////////////////
