#include <iostream>

#include "CommonKernels.cuh"
#include "KernelMatrixAdd.cuh"

std::pair<size_t, size_t> GetWidthHeight(size_t num_elements) {
  size_t log_res_dim_len = 1;
  while (num_elements != (1UL << log_res_dim_len)) {
    ++log_res_dim_len;
  }
  size_t width = 1UL << (log_res_dim_len / 2);
  size_t height = num_elements / width;
  return {width, height};
}

void ProfileFunc(size_t block_size, size_t log_num_elements) {
  size_t num_elements = 1UL << log_num_elements;

  auto [width, height] = GetWidthHeight(num_elements);

  // Host memory.
  float* x = new float[num_elements];
  float* y = new float[num_elements];
  float* res = new float[num_elements];

  for (int i = 0; i < num_elements; ++i) {
    x[i] = 2.0f;
    y[i] = 3.0f;
  }

  // Device memory.
  size_t pitch = 0;

  float* d_x = nullptr;
  cudaMallocPitch(&d_x, &pitch, width * sizeof(float), height);
  float* d_y = nullptr;
  cudaMallocPitch(&d_y, &pitch, width * sizeof(float), height);
  float* d_result = nullptr;
  cudaMallocPitch(&d_result, &pitch, width * sizeof(float), height);

  // Move to device.
  cudaMemcpy2D(d_x,
               pitch,
               x,
               width * sizeof(float),
               width * sizeof(float),
               height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_y,
               pitch,
               y,
               width * sizeof(float),
               width * sizeof(float),
               height,
               cudaMemcpyHostToDevice);

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);
  KernelMatrixAdd<<<
  (num_elements + block_size - 1) / block_size, block_size
  >>>(height, width, pitch, d_x, d_y, d_result);
  cudaEventRecord(stop);

  cudaMemcpy2D(res,
               width * sizeof(float),
               d_result,
               pitch,
               width * sizeof(float),
               height,
               cudaMemcpyDeviceToHost);

  // Put elapsed time.
  cudaEventSynchronize(stop);

  cudaErrchk(cudaPeekAtLastError());

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "KernelMatrixAdd, "
            << block_size << ", "
            << height << "*" << width << ", "
            << millis << '\n';

  // Validate answer.
  for (int i = 0; i < num_elements; ++i) {
    if (res[i] != x[0] + y[0]) {
      std::cerr << "KernelMatrixAdd: invalid answer.\n";
      std::cerr << "i = " << i << ", "
                << "res[i] = " << res[i] << ", "
                << "x[0] = " << x[0] << ", "
                << "y[0] = " << y[0] << ", "
                << "block_size = " << block_size << ", "
                << "num_elements = 2^" << num_elements << ".\n";
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
