# Specify binary name and source file to build it from
cuda_add_executable(tensor_ops
  tensor_ops.cu
  )

cuda_add_library(gpu_operation_library SHARED
  gpu_operation.cu
#  gpu_operation.cpp
)
