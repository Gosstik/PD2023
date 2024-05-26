#pragma once


float ScalarMulTwoReductions(int num_elements, float* vector1, float* vector2, int block_size);

float ScalarMulSumPlusReduction(int num_elements, float* vector1, float* vector2, int block_size);

