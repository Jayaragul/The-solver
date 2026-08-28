#include "MilpProblem.hpp"

#include <cmath>
#include <stdexcept>

namespace sihps {

MilpProblem milp_problem_from_mps(const MpsModel& model) {
    MilpProblem problem;
    problem.relaxation = lp_problem_from_mps(model);
    problem.variable_types = model.col_types;
    problem.maximize = model.objective_sense == ObjectiveSense::MAXIMIZE;
    if (problem.variable_types.empty()) {
        problem.variable_types.assign(static_cast<std::size_t>(problem.n_cols()),
                                      VariableType::CONTINUOUS);
    }
    validate_milp_problem(problem);
    return problem;
}

void validate_milp_problem(const MilpProblem& problem) {
    const auto n = static_cast<std::size_t>(problem.n_cols());
    const auto m = static_cast<std::size_t>(problem.n_rows());
    const LpProblem& lp = problem.relaxation;
    if (problem.variable_types.size() != n || lp.obj.size() != n || lp.lower.size() != n ||
        lp.upper.size() != n || lp.rhs.size() != m || lp.row_types.size() != m ||
        lp.slack_lower.size() != m || lp.slack_upper.size() != m) {
        throw std::invalid_argument("MilpProblem: inconsistent dimension metadata");
    }
    for (std::size_t j = 0; j < n; ++j) {
        if (std::isnan(lp.lower[j]) || std::isnan(lp.upper[j]) || lp.lower[j] > lp.upper[j]) {
            throw std::invalid_argument("MilpProblem: invalid variable bounds");
        }
        if (problem.variable_types[j] == VariableType::BINARY &&
            (lp.upper[j] < 0.0 || lp.lower[j] > 1.0)) {
            throw std::invalid_argument("MilpProblem: binary variable has no [0,1] overlap");
        }
    }
}

} // namespace sihps
