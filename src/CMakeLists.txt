add_library(sihps_core STATIC
  memory/MemoryArena.cpp
  sparse/CSRMatrix.cpp
  sparse/CSCMatrix.cpp
  sparse/Convert.cpp
  cuda/CudaStream.cpp
  cuda/CudaEvent.cpp
  cuda/CudaDevice.cpp
  cuda/CusparseHandle.cpp
  cuda/GpuSpMV.cpp
  cuda/GpuPricer.cpp
  cuda/PricingKernels.cu
  cuda/PdlpKernels.cu
  cuda/GpuPdlp.cpp
  io/MpsReader.cpp
  io/NetlibReference.cpp
  lp/BasisFactorization.cpp
  lp/LpProblem.cpp
  lp/LpSolver.cpp
  lp/Presolve.cpp
  lp/Scaling.cpp
  lp/Simplex.cpp
)

target_include_directories(sihps_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(sihps_core PUBLIC cxx_std_17)
target_link_libraries(sihps_core PUBLIC CUDA::cudart CUDA::cusparse)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sihps_core PUBLIC OpenMP::OpenMP_CXX)
endif()
sihps_set_warnings(sihps_core)
