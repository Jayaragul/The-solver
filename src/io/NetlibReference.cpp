#include "NetlibReference.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sihps {
namespace {

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

bool parse_integer(const std::string& s, long& out) {
    if (s.empty()) return false;
    try {
        std::size_t consumed = 0;
        out = std::stol(s, &consumed);
        return consumed == s.size();
    } catch (...) {
        return false;
    }
}

// Reference optimal values in this file are always written in E notation
// (e.g. -4.6475314286E+02), which distinguishes them unambiguously from
// the plain-integer row/column/nonzero/byte counts on the same line.
bool parse_e_notation(const std::string& s, double& out) {
    if (s.find('E') == std::string::npos && s.find('e') == std::string::npos) return false;
    try {
        std::size_t consumed = 0;
        out = std::stod(s, &consumed);
        return consumed == s.size();
    } catch (...) {
        return false;
    }
}

} // namespace

NetlibReference NetlibReference::load(const std::string& readme_path) {
    std::ifstream in(readme_path);
    if (!in) {
        throw std::runtime_error("NetlibReference: cannot open " + readme_path);
    }

    NetlibReference ref;
    std::string line;
    bool in_alternates = false;

    while (std::getline(in, line)) {
        if (line.find("CPLEX") != std::string::npos && line.find("MINOS") != std::string::npos) {
            in_alternates = true;
            continue;
        }
        if (in_alternates && line.find("The above") != std::string::npos) {
            in_alternates = false;
            continue;
        }

        auto tokens = tokenize(line);
        if (tokens.size() < 2) continue;
        const std::string name = to_upper(tokens[0]);

        if (in_alternates) {
            // NAME  <cplex value>  [<minos value>]   -- either may be absent
            // (blank column) or replaced by "**".
            std::vector<double> values;
            for (std::size_t k = 1; k < tokens.size(); ++k) {
                double v = 0.0;
                if (parse_e_notation(tokens[k], v)) values.push_back(v);
            }
            if (values.empty()) continue;
            // Column position cannot be recovered reliably from
            // whitespace-tokenized text when one column is blank, so the
            // first value is attributed to cplex and a second to minos
            // only when two are present on the line.
            ref.refs_[name].push_back({values[0], values.size() >= 2 ? "cplex" : "cplex-or-minos"});
            if (values.size() >= 2) {
                ref.refs_[name].push_back({values[1], "minos"});
            }
            continue;
        }

        // PROBLEM SUMMARY TABLE row:
        //   NAME  rows  cols  nonzeros  bytes  [B|R|BR]  <optimal value>  [**]
        if (tokens.size() < 6) continue;
        long dummy = 0;
        bool four_ints = true;
        for (std::size_t k = 1; k <= 4; ++k) {
            if (!parse_integer(tokens[k], dummy)) {
                four_ints = false;
                break;
            }
        }
        if (!four_ints) continue;

        for (std::size_t k = 5; k < tokens.size(); ++k) {
            double v = 0.0;
            if (parse_e_notation(tokens[k], v)) {
                ref.refs_[name].push_back({v, "summary"});
                break;
            }
        }
    }

    if (ref.refs_.empty()) {
        throw std::runtime_error("NetlibReference: parsed no reference values from " + readme_path);
    }
    return ref;
}

const std::vector<ReferenceValue>* NetlibReference::find(const std::string& instance_name) const {
    auto it = refs_.find(to_upper(instance_name));
    return (it == refs_.end()) ? nullptr : &it->second;
}

} // namespace sihps
