#include <ScalarMul.cuh>
#include <CommonKernels.cuh>

// Without profiling data.
float ScalarMul(int num_elements,
                float* vector1,
                float* vector2,
                int block_size) {
  // Device memory.
  int vec_byte_size = num_elements * sizeof(float);

  float* d_v1;
  cudaMalloc(&d_v1, vec_byte_size);

  float* d_v2;
  cudaMalloc(&d_v2, vec_byte_size);

  float* d_block_result;
  int d_block_result_size = std::min(block_size, num_elements);
  cudaMalloc(&d_block_result, d_block_result_size * sizeof(float));

  float* d_result;
  cudaMalloc(&d_result, sizeof(float));

  // Move to device.
  cudaMemcpy(d_v1, vector1, vec_byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, vector2, vec_byte_size, cudaMemcpyHostToDevice);

  // Function execution.
  ScalarMulSumToOneBlock<<<
  d_block_result_size, block_size, block_size * sizeof(float)
  >>>(num_elements, d_v1, d_v2, d_block_result);

  cudaDeviceSynchronize();

  SumElementsInFirstBlock<<<
  1, block_size, block_size * sizeof(float)
  >>>(d_block_result_size, d_block_result, d_result);

  float* result = new float[1];
  cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  cudaErrchk(cudaPeekAtLastError());

  // Free resources.
  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_block_result);
  cudaFree(d_result);

  float res = result[0];
  delete[] result;

  return res;
}
