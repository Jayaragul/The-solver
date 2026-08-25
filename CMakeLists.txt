cmake_minimum_required(VERSION 3.24)
project(sankhya LANGUAGES C CXX)

option(SANKHYA_ENABLE_CUDA "Build optional CUDA kernels" OFF)
option(SANKHYA_BUILD_TESTS "Build native smoke tests" ON)

add_library(sankhya_core STATIC
    c/src/sankhya_model.c
    c/src/sankhya_lu.c
    c/src/sankhya_mps.c
    c/src/sankhya_verify.c
    c/src/sk_lu.c
    c/src/sk_core.c
    c/src/sk_solve.c
    c/src/sk_simplex.c
    c/src/sk_scale.c
    c/src/sk_milp.c)
target_include_directories(sankhya_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/c/include)
target_compile_features(sankhya_core PUBLIC c_std_11)

add_executable(sk_mip tools/sk_mip.c)
target_link_libraries(sk_mip PRIVATE sankhya_core)

add_executable(sk_lp tools/sk_lp.c)
target_link_libraries(sk_lp PRIVATE sankhya_core)

add_executable(sk_qp tools/sk_qp.c)
target_link_libraries(sk_qp PRIVATE sankhya_core)

add_executable(sankhya_solve apps/sankhya_solve.c)
target_link_libraries(sankhya_solve PRIVATE sankhya_core)

# libm is a separate object on Unix toolchains; MSVC folds it into the CRT.
find_library(SANKHYA_M_LIBRARY m)
if(SANKHYA_M_LIBRARY)
    target_link_libraries(sankhya_core PUBLIC ${SANKHYA_M_LIBRARY})
endif()

# The transpose-SpMV in the native QP path has independent output columns.
# Use OpenMP when the host toolchain provides it; the solver remains fully
# functional and serial when it is unavailable.
find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    target_link_libraries(sankhya_core PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(sankhya_core PUBLIC SANKHYA_HAS_OPENMP=1)
endif()

if(SANKHYA_BUILD_TESTS)
    enable_testing()
    add_test(NAME sk_simplex_unit
        COMMAND sk_lp ${CMAKE_CURRENT_SOURCE_DIR}/bench/smoke/unit_lp.mps --quiet)
    add_test(NAME sk_simplex_tiny
        COMMAND sk_lp ${CMAKE_CURRENT_SOURCE_DIR}/bench/smoke/tiny_lp.mps --quiet)
    add_test(NAME sk_simplex_scaled
        COMMAND sk_lp ${CMAKE_CURRENT_SOURCE_DIR}/bench/smoke/scaled_lp.mps --quiet)
    add_test(NAME sk_qp_cli
        COMMAND sk_qp ${CMAKE_CURRENT_SOURCE_DIR}/bench/smoke/tiny_qp.qps --expect -4 --quiet)
    add_test(NAME sk_milp_unit
        COMMAND sk_mip ${CMAKE_CURRENT_SOURCE_DIR}/bench/smoke/tiny_milp.mps --quiet)
    add_test(NAME sk_milp_knapsack
        COMMAND sk_mip ${CMAKE_CURRENT_SOURCE_DIR}/bench/smoke/knapsack.mps --quiet)
    add_executable(sankhya_model_smoke c/tests/model_smoke.c)
    target_link_libraries(sankhya_model_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_model_smoke COMMAND sankhya_model_smoke)
    add_executable(sankhya_lu_smoke c/tests/lu_smoke.c)
    target_link_libraries(sankhya_lu_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_lu_smoke COMMAND sankhya_lu_smoke)
    add_executable(sankhya_mps_smoke c/tests/mps_smoke.c)
    target_link_libraries(sankhya_mps_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_mps_smoke COMMAND sankhya_mps_smoke)
    add_executable(sankhya_verify_smoke c/tests/verify_smoke.c)
    target_link_libraries(sankhya_verify_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_verify_smoke COMMAND sankhya_verify_smoke)
    add_executable(sk_lu_test c/tests/sk_lu_test.c)
    target_link_libraries(sk_lu_test PRIVATE sankhya_core)
    add_test(NAME sk_lu_test COMMAND sk_lu_test)
    add_executable(sankhya_qps_smoke c/tests/qps_smoke.c)
    target_link_libraries(sankhya_qps_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_qps_smoke COMMAND sankhya_qps_smoke)
    add_executable(sankhya_lp_solve_smoke c/tests/lp_solve_smoke.c)
    target_link_libraries(sankhya_lp_solve_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_lp_solve_smoke COMMAND sankhya_lp_solve_smoke)
    add_executable(sankhya_milp_branch_smoke c/tests/milp_branch_smoke.c)
    target_link_libraries(sankhya_milp_branch_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_milp_branch_smoke COMMAND sankhya_milp_branch_smoke)
    add_executable(sankhya_presolve_smoke c/tests/presolve_smoke.c)
    target_link_libraries(sankhya_presolve_smoke PRIVATE sankhya_core)
    add_test(NAME sankhya_presolve_smoke COMMAND sankhya_presolve_smoke)
endif()

if(SANKHYA_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
