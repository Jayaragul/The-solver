#pragma once

#include "MpsModel.hpp"

#include <string>

namespace sihps {

// MPS reader: parses NAME, ROWS, COLUMNS, RHS, RANGES, and BOUNDS using
// free-format whitespace tokenization (the de facto standard for reading
// the classic Netlib LP set -- these files do not rely on fixed-column
// parsing). ENDATA is recognized as a section header (so its absence of a
// body is not misparsed) but carries no data.
//
// Per MPS convention, only the first N-type row in ROWS is treated as the
// objective; any subsequent N rows are "free rows" and are intentionally
// excluded from both the constraint matrix and the objective.
//
// BOUNDS types supported: UP, LO, FX, FR, MI, PL, BV (ESTABLISHED MPS
// convention). One documented convention is applied: if a column's only
// bound entry is a negative UP value with no explicit LO, its lower bound
// is set to -infinity rather than left at the default 0 (which would
// otherwise produce an inverted, empty [0, negative] range) -- this
// matches the interpretation used by several established MPS readers,
// not an invented rule.
//
// Throws std::runtime_error on malformed input: an unreadable file, an
// unrecognized row/bound type, or a COLUMNS/RHS/RANGES/BOUNDS entry
// referencing a name that was never declared in ROWS/COLUMNS. This
// project does not silently tolerate malformed model input
// (docs/architecture/SYSTEM.md \S2.2).
MpsModel read_mps_file(const std::string& path);

} // namespace sihps
