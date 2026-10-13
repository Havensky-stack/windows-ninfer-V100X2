cmake_minimum_required(VERSION 3.28)
project(ninfer_windows_toolchain_probe LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)
add_executable(capture_probe ../../tp2/capture_probe.cu)
set_target_properties(capture_probe PROPERTIES
  CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 70-real
  CUDA_RUNTIME_LIBRARY Static
  MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
target_compile_options(capture_probe PRIVATE
  -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zc:preprocessor)
target_link_options(capture_probe PRIVATE /NODEFAULTLIB:LIBCMT)
target_link_libraries(capture_probe PRIVATE CUDA::cuda_driver CUDA::nvtx3)
