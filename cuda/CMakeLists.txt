cmake_minimum_required(VERSION 3.24)
project(sankhya_cuda LANGUAGES CXX CUDA)

add_library(sankhya_cuda SHARED src/sankhya_cuda.cu src/sankhya_cuda_lp.cu)
target_compile_features(sankhya_cuda PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(sankhya_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(sankhya_cuda PRIVATE SANKHYA_CUDA_BUILD)
set_target_properties(sankhya_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

option(SANKHYA_CUDA_BUILD_TESTS "Build CUDA smoke tests" ON)
if(SANKHYA_CUDA_BUILD_TESTS)
    add_executable(sankhya_cuda_smoke tests/spmv_smoke.cu)
    target_link_libraries(sankhya_cuda_smoke PRIVATE sankhya_cuda)
    target_include_directories(sankhya_cuda_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME sankhya_cuda_smoke COMMAND sankhya_cuda_smoke)
    add_executable(sankhya_cuda_pdhg_smoke tests/pdhg_smoke.cu)
    target_link_libraries(sankhya_cuda_pdhg_smoke PRIVATE sankhya_cuda)
    target_include_directories(sankhya_cuda_pdhg_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME sankhya_cuda_pdhg_smoke COMMAND sankhya_cuda_pdhg_smoke)
    add_executable(sankhya_cuda_diagonal_qp_smoke tests/diagonal_qp_smoke.cu)
    target_link_libraries(sankhya_cuda_diagonal_qp_smoke PRIVATE sankhya_cuda)
    target_include_directories(sankhya_cuda_diagonal_qp_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME sankhya_cuda_diagonal_qp_smoke COMMAND sankhya_cuda_diagonal_qp_smoke)
endif()

add_executable(sankhya_pdhg ../apps/sankhya_pdhg.cu)
target_link_libraries(sankhya_pdhg PRIVATE sankhya_core sankhya_cuda)
target_include_directories(sankhya_pdhg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../c/include)

add_executable(sankhya_qp_cuda ../apps/sankhya_qp_cuda.cu)
target_link_libraries(sankhya_qp_cuda PRIVATE sankhya_core sankhya_cuda)
target_include_directories(sankhya_qp_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../c/include)
add_test(NAME sankhya_cuda_general_qp_smoke
    COMMAND sankhya_qp_cuda
        ${CMAKE_SOURCE_DIR}/bench/smoke/tiny_qp.qps
        --iterations 100000 --tolerance 1e-5)
