#include "gpu_operation.h"

using namespace cuda_test;

int main(void) {

  GpuOperation gpu_operation;
  gpu_operation.init(10, 256);
  gpu_operation.print();
  gpu_operation.add();
  gpu_operation.print();
}
