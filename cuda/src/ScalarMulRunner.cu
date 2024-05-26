#include <iostream>

#include <ScalarMulRunner.cuh>
#include <CommonKernels.cuh>

/**
 * @property
 * num_elements <= block_size^2
 */
float ScalarMulTwoReductions(int num_elements,
                             float* vector1,
                             float* vector2,
                             int block_size) {
  size_t log_num_elements = Log2(num_elements);

  // Device memory.
  int vec_byte_size = num_elements * sizeof(float);

  float* d_v1;
  cudaMalloc(&d_v1, vec_byte_size);

  float* d_v2;
  cudaMalloc(&d_v2, vec_byte_size);

  float* d_block_result;
  int reduction_block_count = (num_elements + block_size - 1) / block_size;
  cudaMalloc(&d_block_result, reduction_block_count * sizeof(float));

  float* d_result;
  cudaMalloc(&d_result, sizeof(float));

  // Get reduced_block_size (2^p, where p: 2^p >= reduction_block_count).
  int p_counter = 1;
  int blocks_left = (reduction_block_count - 1) >> 1;
  while (blocks_left > 0) {
    ++p_counter;
    blocks_left >>= 1;
  }
  int reduced_block_size = 1 << p_counter;

  // Move to device.
  cudaMemcpy(d_v1, vector1, vec_byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, vector2, vec_byte_size, cudaMemcpyHostToDevice);

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);

  ScalarMulPerBlockWarpSpecific<<<
  reduction_block_count, block_size, block_size * sizeof(float)
  >>>(num_elements, d_v1, d_v2, d_block_result);
//  ScalarMulPerBlockSimple<<<
//      reduction_block_count, block_size, block_size * sizeof(float)
//      >>>(num_elements, d_v1, d_v2, d_block_result);

  cudaDeviceSynchronize();

  SumElementsInFirstBlock<<<
  1, reduced_block_size, block_size * sizeof(float)
  >>>(reduced_block_size, d_block_result, d_result);

  cudaEventRecord(stop);

  float* result = new float[1];
  cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Put elapsed time.
  cudaEventSynchronize(stop);

  cudaErrchk(cudaPeekAtLastError());

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "ScalarMulTwoReductions, "
            << block_size << ", "
            << "$2^{" << log_num_elements << "}$, "
            << millis << '\n';

  // Free resources.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_block_result);
  cudaFree(d_result);

  float res = result[0];
  delete[] result;

  return res;
}

float ScalarMulSumPlusReduction(int num_elements,
                                float* vector1,
                                float* vector2,
                                int block_size) {
  size_t log_num_elements = Log2(num_elements);

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

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);

  ScalarMulSumToOneBlock<<<
  d_block_result_size, block_size, block_size * sizeof(float)
  >>>(num_elements, d_v1, d_v2, d_block_result);

  cudaDeviceSynchronize();

  SumElementsInFirstBlock<<<
  1, block_size, block_size * sizeof(float)
  >>>(d_block_result_size, d_block_result, d_result);

  cudaEventRecord(stop);

  float* result = new float[1];
  cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  cudaErrchk(cudaPeekAtLastError());

  // Put elapsed time.
  cudaEventSynchronize(stop);

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "ScalarMulSumPlusReduction, "
            << block_size << ", "
            << "$2^{" << log_num_elements << "}$, "
            << millis << '\n';

  // Free resources.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_block_result);
  cudaFree(d_result);

  float res = result[0];
  delete[] result;

  return res;
}
