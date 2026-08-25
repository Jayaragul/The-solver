#pragma once

#include "../sparse/Triplet.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sihps {

// Integrality metadata is kept in the parsed model so it cannot be lost
// when an MPS instance is converted into the LP relaxation used by MILP.
enum class VariableType { CONTINUOUS, INTEGER, BINARY };
enum class ObjectiveSense { MINIMIZE, MAXIMIZE };

// Result of parsing an MPS file: everything needed to build a bounded LP
// -- constraint matrix, objective, row types, RANGES, and column BOUNDS.
// Row/column indices are assigned in first-appearance order within the
// file, which is the conventional MPS ordering.
struct MpsModel {
    std::string name;
    std::string objective_row_name;
    ObjectiveSense objective_sense = ObjectiveSense::MINIMIZE;

    std::int32_t n_rows = 0; // constraint rows only; the objective row is excluded
    std::int32_t n_cols = 0;

    std::vector<std::string> row_names; // size n_rows
    std::vector<std::string> col_names; // size n_cols
    std::vector<char> row_types;        // size n_rows; one of 'L', 'G', 'E'

    std::vector<double> obj; // size n_cols
    std::vector<double> rhs; // size n_rows

    // RANGES: per-row range value and whether one was specified. When
    // has_range[i] is true, row i's allowed Ax range is narrower than the
    // plain row_types[i] would imply -- see MpsReader.cpp for the exact
    // per-type formula (ESTABLISHED MPS convention).
    std::vector<double> row_range;  // size n_rows, meaningful only where has_range is true
    std::vector<bool> has_range;    // size n_rows

    // BOUNDS: per-column [lower, upper]. Defaults to the MPS convention
    // of [0, +infinity) for a column with no BOUNDS entry.
    std::vector<double> col_lower; // size n_cols
    std::vector<double> col_upper; // size n_cols
    std::vector<VariableType> col_types; // size n_cols; defaults to CONTINUOUS

    std::vector<Triplet> constraint_triplets; // rows/cols index into the above
};

} // namespace sihps
