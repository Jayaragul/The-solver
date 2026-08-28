#pragma once

#include "../io/MpsModel.hpp"
#include "../sparse/CSRMatrix.hpp"

#include <limits>
#include <vector>

namespace sihps {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// A linear program in general row-bound form: minimize obj^T x subject to
// Ax {<=,>=,=} rhs (per row_types, narrowed by RANGES where present) and
// lower <= x <= upper. This is the natural, unaugmented form -- Simplex
// (Simplex.hpp) converts it internally into the bounded-slack equality
// form the revised simplex method actually operates on
// (docs/architecture/LP.md).
struct LpProblem {
    CSRMatrix A;
    std::vector<double> obj;
    std::vector<double> rhs;
    std::vector<char> row_types; // 'L', 'G', 'E' -- kept for reference/diagnostics
    std::vector<double> lower;   // structural variable lower bounds, size n_cols
    std::vector<double> upper;   // structural variable upper bounds, size n_cols

    // Precomputed slack bounds for the augmented equality system
    // (Ax + s = rhs), already folding in RANGES where present -- Simplex
    // consumes these directly rather than re-deriving them from
    // row_types, so RANGES-aware and RANGES-free problems (including
    // hand-constructed test problems) go through one code path.
    std::vector<double> slack_lower; // size n_rows
    std::vector<double> slack_upper; // size n_rows

    std::int32_t n_rows() const { return A.rows(); }
    std::int32_t n_cols() const { return A.cols(); }
};

// Fills slack_lower/slack_upper from row_types alone (the plain-row-type
// bounds, ignoring RANGES): 'L' -> [0, +inf), 'G' -> (-inf, 0], 'E' ->
// [0, 0]. Called as the base case by lp_problem_from_mps before RANGES
// adjustments are applied, and usable directly by hand-constructed test
// problems that have no RANGES to account for.
void apply_default_row_bounds(LpProblem& problem);

// Builds an LpProblem from a parsed MPS model, including BOUNDS and
// RANGES (src/io/MpsReader.hpp).
LpProblem lp_problem_from_mps(const MpsModel& model);

} // namespace sihps
