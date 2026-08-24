#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace sihps {

// Published reference optimal objective values for the Netlib LP test
// set, parsed at RUNTIME from Netlib's own index file
// (https://www.netlib.org/lp/data/readme, saved as
// data/netlib_readme.txt). Nothing is hardcoded in this project's source:
// the expected values come from the authoritative distribution, so they
// cannot drift out of sync with it and cannot be quietly tuned to make
// this solver look correct.
//
// Two tables in that file are read:
//
//   1. PROBLEM SUMMARY TABLE -- the primary value per instance. Netlib
//      states these were computed with MINOS 5.3 in VAX double precision.
//
//   2. The "CPLEX(Sparc) / MINOS(MIPS)" table -- later IEEE-double
//      recomputations that DISAGREE with the primary table on a number of
//      instances (e.g. SCAGR7, FFFFF800, FORPLAN, SCRS8, PILOT4).
//
// Because the published sources genuinely disagree for those instances,
// an answer is accepted if it matches ANY published value for that
// instance, and the matched source is reported. Silently comparing only
// against the primary column would mark a solver WRONG for agreeing with
// CPLEX, which would be a false negative rather than a real check.
struct ReferenceValue {
    double value;
    std::string source; // "summary", "cplex", or "minos"
};

class NetlibReference {
public:
    // Throws std::runtime_error if the file cannot be read.
    static NetlibReference load(const std::string& readme_path);

    // Instance names are matched case-insensitively (the file uses
    // uppercase, the .mps filenames are lowercase).
    const std::vector<ReferenceValue>* find(const std::string& instance_name) const;

    std::size_t size() const noexcept { return refs_.size(); }

private:
    std::unordered_map<std::string, std::vector<ReferenceValue>> refs_;
};

} // namespace sihps
