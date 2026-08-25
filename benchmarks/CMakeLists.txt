add_executable(bench_spmv bench_spmv.cpp)
target_link_libraries(bench_spmv PRIVATE sihps_core)
sihps_set_warnings(bench_spmv)

add_executable(bench_netlib bench_netlib.cpp)
target_link_libraries(bench_netlib PRIVATE sihps_core)
sihps_set_warnings(bench_netlib)

add_executable(bench_lp_solve bench_lp_solve.cpp)
target_link_libraries(bench_lp_solve PRIVATE sihps_core)
sihps_set_warnings(bench_lp_solve)

add_executable(validate_netlib validate_netlib.cpp)
target_link_libraries(validate_netlib PRIVATE sihps_core)
sihps_set_warnings(validate_netlib)

add_executable(validate_infeasible validate_infeasible.cpp)
target_link_libraries(validate_infeasible PRIVATE sihps_core)
sihps_set_warnings(validate_infeasible)

add_executable(bench_pricing_rule bench_pricing_rule.cpp)
target_link_libraries(bench_pricing_rule PRIVATE sihps_core)
sihps_set_warnings(bench_pricing_rule)

# Writes each parsed LpProblem out for the external reference-solver
# comparison driven by benchmarks/compare_reference.py. No reference solver
# is linked here -- see export_lp.cpp's header comment.
add_executable(export_lp export_lp.cpp)
target_link_libraries(export_lp PRIVATE sihps_core)
sihps_set_warnings(export_lp)

add_executable(debug_one debug_one.cpp)
target_link_libraries(debug_one PRIVATE sihps_core)
sihps_set_warnings(debug_one)

add_executable(bench_lp_algorithm bench_lp_algorithm.cpp)
target_link_libraries(bench_lp_algorithm PRIVATE sihps_core)
sihps_set_warnings(bench_lp_algorithm)

add_executable(bench_pricing_backend bench_pricing_backend.cpp)
target_link_libraries(bench_pricing_backend PRIVATE sihps_core)
sihps_set_warnings(bench_pricing_backend)

add_executable(bench_parallel bench_parallel.cpp)
target_link_libraries(bench_parallel PRIVATE sihps_core)
sihps_set_warnings(bench_parallel)

add_executable(bench_gpu_latency bench_gpu_latency.cpp)
target_link_libraries(bench_gpu_latency PRIVATE sihps_core)
sihps_set_warnings(bench_gpu_latency)
target_compile_definitions(bench_gpu_latency PRIVATE SIHPS_DATA_ROOT="${CMAKE_SOURCE_DIR}/data")

add_executable(bench_spmv_algorithm bench_spmv_algorithm.cpp)
target_link_libraries(bench_spmv_algorithm PRIVATE sihps_core)
sihps_set_warnings(bench_spmv_algorithm)

add_executable(bench_pdlp bench_pdlp.cpp)
target_link_libraries(bench_pdlp PRIVATE sihps_core)
sihps_set_warnings(bench_pdlp)

add_executable(profile_simplex profile_simplex.cpp)
target_link_libraries(profile_simplex PRIVATE sihps_core)
sihps_set_warnings(profile_simplex)

add_executable(diag_first_order diag_first_order.cpp)
target_link_libraries(diag_first_order PRIVATE sihps_core)
sihps_set_warnings(diag_first_order)

add_executable(validate_crossmethod validate_crossmethod.cpp)
target_link_libraries(validate_crossmethod PRIVATE sihps_core)
sihps_set_warnings(validate_crossmethod)
