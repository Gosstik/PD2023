#pragma once


__global__
void MatrixMul(int a_height, int a_width, int b_width, float *a, float *b, float *res);
