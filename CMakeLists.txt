cmake_minimum_required(VERSION 3.24)

# Two solver cores live in this tree and are built side by side:
#
#   sankhya  - the C core in c/ (revised simplex, sparse LU, branch and bound).
#              No external dependency; always built.
#   sihps    - the C++/CUDA core in src/ with its own benchmark suite in
#              benchmarks/.  Its kernels are compiled unconditionally into
#              sihps_core, so it is configured only when a CUDA toolchain is
#              actually present.
#
# CUDA is detected rather than required.  The Netlib and MIPLIB harnesses that
# the project is judged on are CPU paths, and a machine without nvcc must still
# be able to configure, build and run them.
project(the_solver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SANKHYA_ENABLE_CUDA "Build the optional CUDA kernels under cuda/" OFF)
option(SANKHYA_BUILD_TESTS "Build native smoke tests" ON)
option(SIHPS_BUILD_TESTS "Build sihps unit tests" ON)
option(SIHPS_BUILD_BENCHMARKS "Build sihps benchmarks" ON)
option(SIHPS_REQUIRE_CUDA "Fail configuration if no CUDA toolchain is found" OFF)

# --------------------------------------------------------------- sankhya (C)

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

# libm is a separate object on Unix toolchains; MSVC folds it into the CRT.
find_library(SANKHYA_M_LIBRARY m)
if(SANKHYA_M_LIBRARY)
    target_link_libraries(sankhya_core PUBLIC ${SANKHYA_M_LIBRARY})
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(sankhya_core PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(sankhya_core PUBLIC SANKHYA_HAS_OPENMP=1)
endif()

add_executable(sk_mip tools/sk_mip.c)
target_link_libraries(sk_mip PRIVATE sankhya_core)

add_executable(sk_lp tools/sk_lp.c)
target_link_libraries(sk_lp PRIVATE sankhya_core)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/sk_qp.c)
    add_executable(sk_qp tools/sk_qp.c)
    target_link_libraries(sk_qp PRIVATE sankhya_core)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/sk_bench.c)
    add_executable(sk_bench tools/sk_bench.c)
    target_link_libraries(sk_bench PRIVATE sankhya_core)
endif()

if(SANKHYA_BUILD_TESTS)
    enable_testing()
    file(GLOB SANKHYA_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/c/tests/*.c)
    foreach(test_source ${SANKHYA_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(sankhya_${test_name} ${test_source})
        target_link_libraries(sankhya_${test_name} PRIVATE sankhya_core)
        add_test(NAME sankhya_${test_name} COMMAND sankhya_${test_name})
    endforeach()
endif()

if(SANKHYA_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()

# ----------------------------------------------------------- sihps (C++/CUDA)

include(CheckLanguage)
check_language(CUDA)
set(SIHPS_CUDA_AVAILABLE OFF)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Ampere GA107 (RTX 3050 Laptop) is compute capability 8.6.  Both 86 and 89
    # are emitted so a build made on the dev laptop also runs on Ada parts.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 89)
    endif()
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        set(SIHPS_CUDA_AVAILABLE ON)
    endif()
endif()

if(SIHPS_CUDA_AVAILABLE)
    find_package(Git QUIET)
    set(SIHPS_GIT_COMMIT "unknown")
    set(SIHPS_GIT_DIRTY "unknown")
    if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE SIHPS_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE SIHPS_GIT_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(SIHPS_GIT_STATUS STREQUAL "")
            set(SIHPS_GIT_DIRTY "clean")
        else()
            set(SIHPS_GIT_DIRTY "dirty")
        endif()
        if(SIHPS_GIT_COMMIT STREQUAL "")
            set(SIHPS_GIT_COMMIT "unknown")
        endif()
    endif()
    message(STATUS "SIHPS build stamp: ${SIHPS_GIT_COMMIT} (${SIHPS_GIT_DIRTY})")

    include(cmake/CompilerWarnings.cmake)
    add_subdirectory(src)
    if(SIHPS_BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()
    if(SIHPS_BUILD_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()
elseif(SIHPS_REQUIRE_CUDA)
    message(FATAL_ERROR
        "SIHPS_REQUIRE_CUDA is ON but no CUDA toolchain was found. "
        "Install the CUDA toolkit or configure with -DSIHPS_REQUIRE_CUDA=OFF.")
else()
    message(STATUS
        "No CUDA toolchain found: the sihps C++/CUDA core and its benchmarks "
        "are skipped. The sankhya C core still builds. Set SIHPS_REQUIRE_CUDA=ON "
        "to make this an error.")
endif()
