#include <iostream>

#include "CommonKernels.cuh"
#include "MatrixVectorMul.cuh"

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
  float* a = new float[num_elements];
  float* v = new float[width];
  float* res = new float[height];

  for (int i = 0; i < num_elements; ++i) {
    a[i] = 2.0f;
  }

  for (int i = 0; i < width; ++i) {
    v[i] = 3.0f;
  }

  // Device memory.
  float* d_a = nullptr;
  cudaMalloc(&d_a, num_elements * sizeof(float));
  float* d_v = nullptr;
  cudaMalloc(&d_v, width * sizeof(float));
  float* d_result = nullptr;
  cudaMalloc(&d_result, height * sizeof(float));

  // Move to device.
  cudaMemcpy(d_a, a, num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, height * sizeof(float));

  // Get gridDim and blockDim.
  auto [block_sizes_x, block_sizes_y] = GetWidthHeight(block_size);

  dim3 block_sizes(block_sizes_x, block_sizes_y);
  dim3 num_blocks((height + block_sizes_x - 1) / block_sizes_x,
                  (width + block_sizes_y - 1) / block_sizes_y);

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);

  MatrixVectorMul<<<num_blocks, block_sizes>>>(height, width, d_a, d_v, d_result);
  cudaEventRecord(stop);

  cudaMemcpy(res, d_result, height * sizeof(float), cudaMemcpyDeviceToHost);

  // Put elapsed time.
  cudaEventSynchronize(stop);

  cudaErrchk(cudaPeekAtLastError());

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "MatrixVectorMul, "
            << block_size << ", "
            << height << "*" << width << ", "
            << millis << '\n';

  // Validate answer.
  for (int i = 0; i < height; ++i) {
    if (res[i] != a[0] * v[0] * width) {
      std::cerr << "KernelMatrixAdd: invalid answer.\n";
      std::cerr << "i = " << i << ", "
                << "res[i] = " << res[i] << ", "
                << "a[0] = " << a[0] << ", "
                << "v[0] = " << v[0] << ", "
                << "matrix size = " << height << "*" << width << ", "
                << "block_size = " << block_size << ", "
                << "num_elements = 2^" << log_num_elements << ".\n";
      exit(1);
    }
  }

  // Free resources.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_a);
  cudaFree(d_v);
  cudaFree(d_result);

  delete[] a;
  delete[] v;
  delete[] res;
}

int main(int argc, char** argv) {
//  size_t cuda_device_number = std::strtoull(argv[1], nullptr, 10);
//  size_t block_size = std::strtoull(argv[2], nullptr, 10);
//  size_t num_elements = std::strtoull(argv[3], nullptr, 10);

  size_t cuda_device_number = 3;
  cudaSetDevice(cuda_device_number);

  for (size_t block_size = 32; block_size <= 1024; block_size *= 2) {
    for (size_t log_num_elements = 10; log_num_elements <= 30; ++log_num_elements) {
      ProfileFunc(block_size, log_num_elements);
    }
  }
}
