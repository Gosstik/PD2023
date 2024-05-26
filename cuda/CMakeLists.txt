################################################################################

# For local environment.

#cmake_minimum_required(VERSION 3.10)
#project(CudaTasks)
#
## --------------------------------------------------------------------
#
#option(DEVELOPMENT_MODE "Enable building local_tests." OFF)
#
## --------------------------------------------------------------------
#
#find_package(CUDA REQUIRED)
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")
#
## --------------------------------------------------------------------
#
## Uncomment next lines to save binary files in special dirs.
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#
## --------------------------------------------------------------------
#
#include_directories(include)
#
#cuda_add_library(common_kernels src/CommonKernels.cu STATIC)
## You can add files to common_kernels library; this library will be compiled with tests
#
#cuda_add_executable(01-add runners/01-add.cu src/KernelAdd.cu)
#cuda_add_executable(02-mul runners/02-mul.cu src/KernelMul.cu)
#cuda_add_executable(03-matrix-add runners/03-matrix-add.cu src/KernelMatrixAdd.cu)
#cuda_add_executable(04-matrix-vector-mul runners/04-matrix-vector-mul.cu src/MatrixVectorMul.cu)
#cuda_add_executable(05-scalar-mul runners/05-scalar-mul.cu src/ScalarMulRunner.cu src/ScalarMul.cu)
#cuda_add_executable(
#    06-cosine-vector runners/06-cosine-vector.cu src/CosineVector.cu
#    src/ScalarMulRunner.cu
#    src/ScalarMul.cu
#)
#cuda_add_executable(07-matrix-mul runners/07-matrix-mul.cu src/MatrixMul.cu)
#cuda_add_executable(08-filter runners/08-filter.cu src/Filter.cu)
#
#
#target_link_libraries(01-add common_kernels)
#target_link_libraries(02-mul common_kernels)
#target_link_libraries(03-matrix-add common_kernels)
#target_link_libraries(04-matrix-vector-mul common_kernels)
#target_link_libraries(05-scalar-mul common_kernels)
#target_link_libraries(06-cosine-vector common_kernels)
#target_link_libraries(07-matrix-mul common_kernels)
#target_link_libraries(08-filter common_kernels)
#
#add_subdirectory(tests)
#
#if (DEVELOPMENT_MODE)
#    include_directories("local_tests")
#    add_subdirectory(local_tests)
#endif()

################################################################################

# Try to make CI and local environment friends.

#set_target_properties(common_kernels PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(01-add PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(02-mul PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(03-matrix-add PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(04-matrix-vector-mul PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(05-scalar-mul PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(06-cosine-vector PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(07-matrix-mul PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#set_target_properties(08-filter PROPERTIES CUDA_ARCHITECTURES "35;50;72")

################################################################################

# For CI.

cmake_minimum_required(VERSION 3.10)

project(CudaTasks LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")

add_library(common_kernels STATIC src/CommonKernels.cu)
# You can add files to common_kernels library; this library will be compiled with tests

add_executable(01-add runners/01-add.cu src/KernelAdd.cu)
add_executable(02-mul runners/02-mul.cu src/KernelMul.cu)
add_executable(03-matrix-add runners/03-matrix-add.cu src/KernelMatrixAdd.cu)
add_executable(04-matrix-vector-mul runners/04-matrix-vector-mul.cu src/MatrixVectorMul.cu)
add_executable(05-scalar-mul runners/05-scalar-mul.cu src/ScalarMulRunner.cu src/ScalarMul.cu)
add_executable(06-cosine-vector runners/06-cosine-vector.cu src/CosineVector.cu src/ScalarMulRunner.cu src/ScalarMul.cu)
add_executable(07-matrix-mul runners/07-matrix-mul.cu src/MatrixMul.cu)
add_executable(08-filter runners/08-filter.cu src/Filter.cu)


target_link_libraries(01-add common_kernels)
target_link_libraries(02-mul common_kernels)
target_link_libraries(03-matrix-add common_kernels)
target_link_libraries(04-matrix-vector-mul common_kernels)
target_link_libraries(05-scalar-mul common_kernels)
target_link_libraries(06-cosine-vector common_kernels)
target_link_libraries(07-matrix-mul common_kernels)
target_link_libraries(08-filter common_kernels)

add_subdirectory(tests)
