cmake_minimum_required(VERSION 3.24)

project(sihps_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CC 8.9 measured directly on the target dev GPU (RTX 3050 Laptop) via a
# device-query smoke test -- see docs/architecture/CPU_GPU.md.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SIHPS_BUILD_TESTS "Build unit tests" ON)
option(SIHPS_BUILD_BENCHMARKS "Build benchmarks" ON)

find_package(CUDAToolkit REQUIRED)

# Optional by design. Every parallel loop in this codebase is written so
# that dropping the pragmas changes nothing but the wall clock
# (src/parallel/Parallel.hpp), so a toolchain without OpenMP still builds
# and still produces bit-identical results.
find_package(OpenMP)

include(cmake/CompilerWarnings.cmake)

add_subdirectory(src)

if(SIHPS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SIHPS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
