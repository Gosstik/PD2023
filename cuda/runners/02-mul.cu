#include <iostream>

#include "CommonKernels.cuh"
#include "KernelMul.cuh"

void ProfileFunc(size_t block_size, size_t log_num_elements) {
  size_t num_elements = 1UL << log_num_elements;

  // Host memory.
  float* x = new float[num_elements];
  float* y = new float[num_elements];
  float* res = new float[num_elements];

  for (int i = 0; i < num_elements; ++i) {
    x[i] = 2.0f;
    y[i] = 3.0f;
  }

  // Device memory.
  size_t d_arr_size = num_elements * sizeof(float);

  float* d_x = nullptr;
  cudaMalloc(&d_x, d_arr_size);
  float* d_y = nullptr;
  cudaMalloc(&d_y, d_arr_size);
  float* d_result = nullptr;
  cudaMalloc(&d_result, d_arr_size);

  // Move to device.
  cudaMemcpy(d_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, num_elements * sizeof(float), cudaMemcpyHostToDevice);

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);
  KernelMul<<<(num_elements + block_size - 1) / block_size, block_size>>>(num_elements, d_x, d_y, d_result);
  cudaEventRecord(stop);

  cudaMemcpy(res, d_result, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  // Put elapsed time.
  cudaEventSynchronize(stop);

  cudaErrchk(cudaPeekAtLastError());

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "KernelMul, "
            << block_size << ", "
            << "$2^{" << log_num_elements << "}$, "
            << millis << '\n';

  // Validate answer.
  for (int i = 0; i < num_elements; ++i) {
    if (res[i] != x[0] * y[0]) {
      std::cerr << "KernelMul: invalid answer.\n";
      std::cerr << "i = " << i << ", "
                << "res[i] = " << res[i] << ", "
                << "x[0] = " << x[0] << ", "
                << "y[0] = " << y[0] << ", "
                << "block_size = " << block_size << ", "
                << "num_elements = 2^" << log_num_elements << ".\n";
      exit(1);
    }
  }

  // Free resources.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_result);

  delete[] x;
  delete[] y;
  delete[] res;
}

int main(int argc, char** argv) {
//  size_t cuda_device_number = std::strtoull(argv[1], nullptr, 10);
//  size_t block_size = std::strtoull(argv[2], nullptr, 10);
//  size_t num_elements = std::strtoull(argv[3], nullptr, 10);

  size_t cuda_device_number = 3;
  cudaSetDevice(cuda_device_number);

  for (size_t block_size = 32; block_size <= 1024; block_size *= 2) {
    for (size_t log_num_elements = 10; log_num_elements <= 29; ++log_num_elements) {
      ProfileFunc(block_size, log_num_elements);
    }
  }
}
