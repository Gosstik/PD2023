#include <iostream>

#include "MatrixMul.cuh"
#include "CommonKernels.cuh"

std::pair<size_t, size_t> GetWidthHeight(size_t res_dim_len) {
  size_t log_res_dim_len = 1;
  while (res_dim_len != (1 << log_res_dim_len)) {
    ++log_res_dim_len;
  }
  size_t block_dim_len = 1 << (log_res_dim_len / 2);
  size_t grid_dim_len = res_dim_len / block_dim_len;
  return {block_dim_len, grid_dim_len};
}

void ProfileFunc(size_t block_size, size_t log_num_elements) {
  size_t num_elements = 1UL << log_num_elements;

  size_t matrix_dim = GetWidthHeight(num_elements).first;

  size_t a_width = matrix_dim;
  size_t a_height = matrix_dim;
  size_t a_num_elems = a_width * a_height;

  size_t b_width = matrix_dim;
  size_t b_height = a_width;
  size_t b_num_elems = b_width * b_height;

  size_t res_width = b_width;
  size_t res_height = a_height;
  size_t res_num_elems = a_height * b_width;

  // Host memory.
  float* a = new float[a_num_elems];
  float* b = new float[b_num_elems];
  float* res = new float[res_num_elems];

  for (int i = 0; i < a_num_elems; ++i) {
    a[i] = 2.0f;
  }

  for (int i = 0; i < b_num_elems; ++i) {
    b[i] = 3.0f;
  }

  // Device memory.
  float* d_a = nullptr;
  cudaMalloc(&d_a, a_num_elems * sizeof(float));
  float* d_b = nullptr;
  cudaMalloc(&d_b, b_num_elems * sizeof(float));
  float* d_res = nullptr;
  cudaMalloc(&d_res, res_num_elems * sizeof(float));

  // Move to device.
  cudaMemcpy(d_a, a, a_num_elems * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, b_num_elems * sizeof(float), cudaMemcpyHostToDevice);

  // Get gridDim and blockDim.
  auto [block_dim_x, block_dim_y] = GetWidthHeight(block_size);
  size_t grid_dim_x = res_height / block_dim_x;
  size_t grid_dim_y = res_width / block_dim_y;

  dim3 block_sizes(block_dim_x, block_dim_y);
  dim3 num_blocks(grid_dim_x, grid_dim_y);
  size_t shared_memory_bytes = 2 * block_sizes.x * block_sizes.y * sizeof(float);

  // Events.
  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // Function execution.
  cudaEventRecord(start);

  MatrixMul<<<
  num_blocks, block_sizes, shared_memory_bytes
  >>>(a_height, a_width, b_width, d_a, d_b, d_res);
  cudaEventRecord(stop);

  cudaMemcpy(res, d_res, res_num_elems * sizeof(float), cudaMemcpyDeviceToHost);

  // Put elapsed time.
  cudaEventSynchronize(stop);

  cudaErrchk(cudaPeekAtLastError());

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  std::cout << "MatrixMul, "
            << block_size << ", "
            << a_height << "*" << b_height << "*" << b_width << ", "
            << millis << '\n';

  // Validate answer.
  for (int i = 0; i < a_height; ++i) {
    if (res[i] != a[0] * b[0] * a_width) {
      std::cerr << "MatrixMul: invalid answer.\n";
      std::cerr << "i = " << i << ", "
                << "res[i] = " << res[i] << ", "
                << "correct = " << a[0] * b[0] * a_width << ", "
                << "a[0] = " << a[0] << ", "
                << "b[0] = " << b[0] << ", "
                << "a_height = " << a_height << ", "
                << "a_width = " << a_width << ", "
                << "b_width = " << b_width << ".\n";
      exit(1);
    }
  }

  // Free resources.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_res);

  delete[] a;
  delete[] b;
  delete[] res;
}

int main(int argc, char** argv) {
//  size_t cuda_device_number = std::strtoull(argv[1], nullptr, 10);
//  size_t block_size = std::strtoull(argv[2], nullptr, 10);
//  size_t num_elements = std::strtoull(argv[3], nullptr, 10);

  size_t cuda_device_number = 3;
  cudaSetDevice(cuda_device_number);

  for (size_t block_size = 64; block_size <= 1024; block_size *= 4) {
    for (size_t log_num_elements = 10; log_num_elements <= 28; log_num_elements += 2) {
      ProfileFunc(block_size, log_num_elements);
    }
  }
}
