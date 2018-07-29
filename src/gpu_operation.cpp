#include "gpu_operation.h"
using namespace cuda_test;

__global__
void add_gpu(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(int i = index; i < n; i+=stride)
    y[i] += x[i];
}

__global__
void subtract_gpu(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(int i = index; i < n; i+=stride)
    y[i] -= x[i];
}

__global__
void multiply_gpu(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(int i = index; i < n; i+=stride)
    y[i] *= x[i];
}

GpuOperation::GpuOperation() {
  v1_ = NULL;
  v2_ = NULL;
  size_ = 0;
  blocksize_ = 0;
  numblocks_ = 0;
}

GpuOperation::~GpuOperation() {
  cudaFree(v1_);
  cudaFree(v2_);
}

void GpuOperation::init(const int size, const int blocksize) {
  size_ = size;
  blocksize_ = blocksize;
  numblocks_ = (size_ + blocksize_-1) / blocksize_;
  cudaMallocManaged(&v1_, size_ * sizeof(float));
  cudaMallocManaged(&v2_, size_ * sizeof(float));
  // initialize x and y arrays on the host
  for (int i = 0; i < size_; i++) {
    v1_[i] = 1.0f;
    v2_[i] = 2.0f;
  }  
}

void GpuOperation::add() {
  add_gpu<<< numblocks_, blocksize_ >>>(size_, v1_, v2_);
  cudaDeviceSynchronize();
}

void GpuOperation::subtract() {
  subtract_gpu<<< numblocks_, blocksize_ >>>(size_, v1_, v2_);
  cudaDeviceSynchronize();
}

void GpuOperation::multiply() {
  multiply_gpu<<< numblocks_, blocksize_ >>>(size_, v1_, v2_);
  cudaDeviceSynchronize();
}
