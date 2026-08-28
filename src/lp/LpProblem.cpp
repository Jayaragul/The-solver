#include "LpProblem.hpp"

#include <cmath>

namespace sihps {

void apply_default_row_bounds(LpProblem& problem) {
    const auto m = problem.row_types.size();
    problem.slack_lower.assign(m, 0.0);
    problem.slack_upper.assign(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        switch (problem.row_types[i]) {
            case 'L':
                problem.slack_lower[i] = 0.0;
                problem.slack_upper[i] = kInfinity;
                break;
            case 'G':
                problem.slack_lower[i] = -kInfinity;
                problem.slack_upper[i] = 0.0;
                break;
            case 'E':
            default:
                problem.slack_lower[i] = 0.0;
                problem.slack_upper[i] = 0.0;
                break;
        }
    }
}

LpProblem lp_problem_from_mps(const MpsModel& model) {
    LpProblem problem;
    problem.A = CSRMatrix::from_triplets(model.n_rows, model.n_cols, model.constraint_triplets);
    problem.obj = model.obj;
    problem.rhs = model.rhs;
    problem.row_types = model.row_types;
    problem.lower = model.col_lower;
    problem.upper = model.col_upper;

    apply_default_row_bounds(problem);

    // RANGES narrows the plain row-type bounds. With Ax + s = rhs (s is
    // the slack), a range r on row i restricts Ax to an interval anchored
    // at rhs; since s = rhs - Ax, that interval maps directly onto s's
    // bounds (ESTABLISHED MPS convention -- see e.g. the RANGES section
    // semantics documented for standard MPS readers):
    //   L row: Ax in [rhs-|r|, rhs]      -> s in [0, |r|]
    //   G row: Ax in [rhs, rhs+|r|]      -> s in [-|r|, 0]
    //   E row, r >= 0: Ax in [rhs, rhs+r]  -> s in [-r, 0]
    //   E row, r <  0: Ax in [rhs+r, rhs]  -> s in [0, -r]
    for (std::int32_t i = 0; i < model.n_rows; ++i) {
        if (!model.has_range[static_cast<std::size_t>(i)]) continue;
        double r = model.row_range[static_cast<std::size_t>(i)];
        char t = model.row_types[static_cast<std::size_t>(i)];
        double abs_r = std::fabs(r);
        if (t == 'L') {
            problem.slack_lower[static_cast<std::size_t>(i)] = 0.0;
            problem.slack_upper[static_cast<std::size_t>(i)] = abs_r;
        } else if (t == 'G') {
            problem.slack_lower[static_cast<std::size_t>(i)] = -abs_r;
            problem.slack_upper[static_cast<std::size_t>(i)] = 0.0;
        } else { // 'E'
            if (r >= 0.0) {
                problem.slack_lower[static_cast<std::size_t>(i)] = -r;
                problem.slack_upper[static_cast<std::size_t>(i)] = 0.0;
            } else {
                problem.slack_lower[static_cast<std::size_t>(i)] = 0.0;
                problem.slack_upper[static_cast<std::size_t>(i)] = -r;
            }
        }
    }

    return problem;
}

} // namespace sihps
