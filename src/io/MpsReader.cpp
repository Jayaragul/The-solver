#include "MpsReader.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace sihps {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

bool is_section_header(const std::string& line, const std::string& first_token) {
    if (line.empty() || line[0] == ' ' || line[0] == '\t') {
        return false; // data lines are indented in standard MPS layout
    }
    static const std::unordered_set<std::string> keywords = {
        "NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA", "OBJSENSE"};
    return keywords.count(first_token) > 0;
}

} // namespace

MpsModel read_mps_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("MpsReader: cannot open file: " + path);
    }

    MpsModel model;
    std::unordered_map<std::string, std::int32_t> row_index; // L/G/E rows only
    std::unordered_set<std::string> declared_rows;            // every row, including extra N rows
    std::unordered_map<std::string, std::int32_t> col_index;
    std::vector<bool> has_explicit_lower; // tracks LO/FX/FR/MI/BV, for the negative-UP convention
    std::string section;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '*') {
            continue;
        }
        auto tokens = tokenize(line);
        if (tokens.empty()) {
            continue;
        }

        if (is_section_header(line, tokens[0])) {
            section = tokens[0];
            if (section == "NAME" && tokens.size() > 1) {
                model.name = tokens[1];
            }
            continue;
        }

        if (section == "ROWS") {
            if (tokens.size() < 2) {
                throw std::runtime_error("MpsReader: malformed ROWS line: " + line);
            }
            const std::string& type = tokens[0];
            const std::string& name = tokens[1];
            declared_rows.insert(name);

            if (type == "N") {
                if (model.objective_row_name.empty()) {
                    model.objective_row_name = name;
                }
                continue; // objective row, or a subsequent dropped free row
            }
            if (type.size() != 1 || (type[0] != 'L' && type[0] != 'G' && type[0] != 'E')) {
                throw std::runtime_error("MpsReader: unknown row type '" + type + "' for row " +
                                          name);
            }
            row_index[name] = model.n_rows++;
            model.row_names.push_back(name);
            model.row_types.push_back(type[0]);
            model.row_range.push_back(0.0);
            model.has_range.push_back(false);

        } else if (section == "COLUMNS") {
            if (tokens.size() < 3) {
                throw std::runtime_error("MpsReader: malformed COLUMNS line: " + line);
            }
            const std::string& col_name = tokens[0];
            std::int32_t c;
            auto cit = col_index.find(col_name);
            if (cit == col_index.end()) {
                c = model.n_cols++;
                col_index[col_name] = c;
                model.col_names.push_back(col_name);
                model.obj.push_back(0.0);
                model.col_lower.push_back(0.0);
                model.col_upper.push_back(kInf);
                has_explicit_lower.push_back(false);
            } else {
                c = cit->second;
            }

            for (std::size_t k = 1; k + 1 < tokens.size(); k += 2) {
                const std::string& row_name = tokens[k];
                const double value = std::stod(tokens[k + 1]);

                if (row_name == model.objective_row_name) {
                    model.obj[static_cast<std::size_t>(c)] = value;
                    continue;
                }
                auto rit = row_index.find(row_name);
                if (rit != row_index.end()) {
                    model.constraint_triplets.push_back({rit->second, c, value});
                    continue;
                }
                if (declared_rows.count(row_name)) {
                    continue; // a legitimately-dropped extra free (N) row
                }
                throw std::runtime_error("MpsReader: COLUMNS references undeclared row: " +
                                          row_name);
            }

        } else if (section == "RHS") {
            if (model.rhs.empty() && model.n_rows > 0) {
                model.rhs.assign(static_cast<std::size_t>(model.n_rows), 0.0);
            }
            for (std::size_t k = 1; k + 1 < tokens.size(); k += 2) {
                const std::string& row_name = tokens[k];
                const double value = std::stod(tokens[k + 1]);

                if (row_name == model.objective_row_name) {
                    continue; // objective constant shift; not used by anything yet
                }
                auto rit = row_index.find(row_name);
                if (rit != row_index.end()) {
                    model.rhs[static_cast<std::size_t>(rit->second)] = value;
                    continue;
                }
                if (declared_rows.count(row_name)) {
                    continue; // RHS entry for a dropped free row
                }
                throw std::runtime_error("MpsReader: RHS references undeclared row: " + row_name);
            }

        } else if (section == "RANGES") {
            for (std::size_t k = 1; k + 1 < tokens.size(); k += 2) {
                const std::string& row_name = tokens[k];
                const double value = std::stod(tokens[k + 1]);

                if (row_name == model.objective_row_name) {
                    continue;
                }
                auto rit = row_index.find(row_name);
                if (rit == row_index.end()) {
                    if (declared_rows.count(row_name)) continue;
                    throw std::runtime_error("MpsReader: RANGES references undeclared row: " +
                                              row_name);
                }
                model.row_range[static_cast<std::size_t>(rit->second)] = value;
                model.has_range[static_cast<std::size_t>(rit->second)] = true;
            }

        } else if (section == "BOUNDS") {
            if (tokens.size() < 3) {
                throw std::runtime_error("MpsReader: malformed BOUNDS line: " + line);
            }
            const std::string& type = tokens[0];
            const std::string& col_name = tokens[2];
            auto cit = col_index.find(col_name);
            if (cit == col_index.end()) {
                throw std::runtime_error("MpsReader: BOUNDS references undeclared column: " +
                                          col_name);
            }
            const auto c = static_cast<std::size_t>(cit->second);
            const double value = (tokens.size() >= 4) ? std::stod(tokens[3]) : 0.0;

            if (type == "UP") {
                model.col_upper[c] = value;
                if (value < 0.0 && !has_explicit_lower[c]) {
                    // Established convention: an UP-only negative bound
                    // with no explicit LO would otherwise leave the
                    // default lower bound (0) above the upper bound.
                    model.col_lower[c] = -kInf;
                }
            } else if (type == "LO") {
                model.col_lower[c] = value;
                has_explicit_lower[c] = true;
            } else if (type == "FX") {
                model.col_lower[c] = value;
                model.col_upper[c] = value;
                has_explicit_lower[c] = true;
            } else if (type == "FR") {
                model.col_lower[c] = -kInf;
                model.col_upper[c] = kInf;
                has_explicit_lower[c] = true;
            } else if (type == "MI") {
                model.col_lower[c] = -kInf;
                has_explicit_lower[c] = true;
            } else if (type == "PL") {
                model.col_upper[c] = kInf;
            } else if (type == "BV") {
                model.col_lower[c] = 0.0;
                model.col_upper[c] = 1.0;
                has_explicit_lower[c] = true;
            } else {
                throw std::runtime_error("MpsReader: unknown BOUNDS type: " + type);
            }
        }
        // ENDATA carries no data.
    }

    if (model.rhs.empty() && model.n_rows > 0) {
        model.rhs.assign(static_cast<std::size_t>(model.n_rows), 0.0);
    }

    return model;
}

} // namespace sihps
