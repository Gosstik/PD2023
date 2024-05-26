#include <cmath>
#include <iostream>

#include <CosineVector.cuh>
#include <CommonKernels.cuh>
#include <ScalarMul.cuh>

float CosineVector(int num_elements, float* vector1, float* vector2, int block_size) {
  size_t log_num_elements = Log2(num_elements);

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);

  float norm1 = VectorSquare2Norm(num_elements, vector1, block_size);
  float norm2 = VectorSquare2Norm(num_elements, vector2, block_size);
  float scalar_mul = ScalarMul(num_elements, vector1, vector2, block_size);

  cudaDeviceSynchronize();

  float res = 0.f;
  if (scalar_mul == 0.f) {
    res = 0.f;
  } else {
    res = (scalar_mul < 0.f) ? -1.f : 1.f; // Save sign of cosine.
    res *= std::sqrt(scalar_mul / norm1 * scalar_mul / norm2);
  }

  cudaEventRecord(stop);

  // Put elapsed time.
  cudaEventSynchronize(stop);

  cudaErrchk(cudaPeekAtLastError());

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "CosineVector, "
            << block_size << ", "
            << "$2^{" << log_num_elements << "}$, "
            << millis << '\n';

  // Free resources.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return res;
}
