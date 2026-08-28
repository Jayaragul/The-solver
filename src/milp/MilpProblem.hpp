#pragma once

#include "../io/MpsModel.hpp"
#include "../lp/LpProblem.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

// The MILP model owns the continuous relaxation and the integrality contract
// separately. The objective remains in the caller's natural sense; the MILP
// solver normalizes maximization in its private LP workspace. The LP engine
// never sees this metadata; that prevents an LP
// relaxation from accidentally treating an integer variable as fixed or
// rounded before its bound has been certified by branch-and-bound.
struct MilpProblem {
    LpProblem relaxation;
    std::vector<VariableType> variable_types;
    bool maximize = false;

    std::int32_t n_rows() const { return relaxation.n_rows(); }
    std::int32_t n_cols() const { return relaxation.n_cols(); }
};

MilpProblem milp_problem_from_mps(const MpsModel& model);

// Validates dimensions, bounds, and the supported variable-type contract.
// Throws std::invalid_argument on malformed models rather than allowing a
// search to report a result for a model it did not actually solve.
void validate_milp_problem(const MilpProblem& problem);

} // namespace sihps
