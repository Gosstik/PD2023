#include <iostream>

#include <ScalarMulRunner.cuh>

void ProfileFunc(size_t block_size, size_t log_num_elements) {
  size_t num_elements = 1UL << log_num_elements;

  // Host memory.
  float* v1 = new float[num_elements];
  float* v2 = new float[num_elements];

  for (int i = 0; i < num_elements; ++i) {
    v1[i] = 2.0f;
    v2[i] = 2.0f;
  }

  // ScalarMulTwoReductions.
  float result = ScalarMulTwoReductions(num_elements, v1, v2, block_size);

  // Validate answer.
  if (result != v1[0] * v2[0] * num_elements) {
    std::cerr << "ScalarMulTwoReductions: invalid answer.\n";
    std::cerr << "result = " << result
              << ", v1[0] = " << v1[0]
              << ", v2[0] = " << v2[0]
              << ", num_elements = 2^" << log_num_elements << ".\n";
    exit(1);
  }

  delete[] v1;
  delete[] v2;
}

void ProfileSumPlusReduction(size_t block_size, size_t log_num_elements) {
  size_t num_elements = 1UL << log_num_elements;

  // Host memory.
  float* v1 = new float[num_elements];
  float* v2 = new float[num_elements];

  for (int i = 0; i < num_elements; ++i) {
    v1[i] = 2.0f;
    v2[i] = 2.0f;
  }

  // ScalarMulSumPlusReduction.
  float result = ScalarMulSumPlusReduction(num_elements, v1, v2, block_size);

  // Validate answer.
  if (result != v1[0] * v2[0] * num_elements) {
    std::cerr << "ScalarMulSumPlusReduction: invalid answer.\n";
    std::cerr << "result = " << result
              << ", v1[0] = " << v1[0]
              << ", v2[0] = " << v2[0]
              << ", num_elements = 2^" << log_num_elements << ".\n";
    exit(1);
  }

  delete[] v1;
  delete[] v2;
}

int main(int argc, char** argv) {
//  size_t cuda_device_number = std::strtoull(argv[1], nullptr, 10);
//  size_t block_size = std::strtoull(argv[2], nullptr, 10);
//  size_t num_elements = std::strtoull(argv[3], nullptr, 10);

  size_t cuda_device_number = 3;
  cudaSetDevice(cuda_device_number);

  for (size_t block_size = 32; block_size <= 1024; block_size *= 2) {
    for (size_t log_num_elements = 10; log_num_elements <= 28; ++log_num_elements) {
      if (block_size * block_size >= (1UL << log_num_elements)) {
        ProfileFunc(block_size, log_num_elements);
      }
      ProfileSumPlusReduction(block_size, log_num_elements);
    }
  }
}
