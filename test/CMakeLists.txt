cuda_add_executable(gpu_operation
  gpu_operation.cu
  )

target_link_libraries(gpu_operation
  gpu_operation_library
)
