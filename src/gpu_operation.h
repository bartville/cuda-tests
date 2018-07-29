#pragma once

#include <iostream>

namespace cuda_test {

  class GpuOperation {
  public:
    GpuOperation();
    ~GpuOperation();

    void init(const int size, const int blocksize);
    void add();
    void subtract();
    void multiply();

    const void print() const;
    
  private:
    float* v1_;
    float* v2_;
    int size_;
    int blocksize_;
    int numblocks_;
  };

}
