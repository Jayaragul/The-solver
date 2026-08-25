// Ground-truthed against the actual contents of data/netlib_lp/feasible/
// afiro.mps (read directly, not recalled from memory -- prompt.md's
// anti-fabrication discipline applies to test fixtures too): 27 rows of
// type E/L, one N row named COST (the objective), and 32 distinct column
// names appearing in COLUMNS-section position 0.

#include "../test_framework.hpp"
#include "io/MpsReader.hpp"
#include "sparse/CSRMatrix.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

using sihps::CSRMatrix;
using sihps::read_mps_file;

namespace {
std::string afiro_path() {
    return std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/afiro.mps";
}
} // namespace

SIHPS_TEST(mps_reader_parses_afiro_dimensions_correctly) {
    auto model = read_mps_file(afiro_path());
    SIHPS_ASSERT_EQ(model.name, std::string("AFIRO"));
    SIHPS_ASSERT_EQ(model.objective_row_name, std::string("COST"));
    SIHPS_ASSERT_EQ(model.n_rows, 27);
    SIHPS_ASSERT_EQ(model.n_cols, 32);
    SIHPS_ASSERT_EQ(model.row_names.size(), std::size_t{27});
    SIHPS_ASSERT_EQ(model.col_names.size(), std::size_t{32});
    SIHPS_ASSERT_EQ(model.row_types.size(), std::size_t{27});
    SIHPS_ASSERT_EQ(model.obj.size(), std::size_t{32});
    SIHPS_ASSERT_EQ(model.rhs.size(), std::size_t{27});
}

SIHPS_TEST(mps_reader_afiro_row_types_are_only_l_or_e) {
    auto model = read_mps_file(afiro_path());
    int l_count = 0, e_count = 0;
    for (char t : model.row_types) {
        SIHPS_ASSERT_TRUE(t == 'L' || t == 'E');
        if (t == 'L') ++l_count;
        if (t == 'E') ++e_count;
    }
    // Counted directly from the raw file: 19 'L' rows, 8 'E' rows.
    SIHPS_ASSERT_EQ(l_count, 19);
    SIHPS_ASSERT_EQ(e_count, 8);
}

SIHPS_TEST(mps_reader_produces_a_structurally_valid_csr_matrix) {
    auto model = read_mps_file(afiro_path());
    CSRMatrix a = CSRMatrix::from_triplets(model.n_rows, model.n_cols, model.constraint_triplets);
    a.validate(); // throws on any structural violation

    std::vector<double> x(static_cast<std::size_t>(model.n_cols), 1.0);
    std::vector<double> y(static_cast<std::size_t>(model.n_rows));
    a.multiply(x.data(), y.data()); // must not crash/throw
}

SIHPS_TEST(mps_reader_afiro_specific_known_coefficient) {
    // Spot-check one exact entry read straight from the file's first
    // COLUMNS line: "X01  X48  .301  R09  -1." -- column X01, row X48,
    // value 0.301.
    auto model = read_mps_file(afiro_path());
    std::int32_t col_x01 = -1, row_x48 = -1;
    for (std::size_t i = 0; i < model.col_names.size(); ++i) {
        if (model.col_names[i] == "X01") col_x01 = static_cast<std::int32_t>(i);
    }
    for (std::size_t i = 0; i < model.row_names.size(); ++i) {
        if (model.row_names[i] == "X48") row_x48 = static_cast<std::int32_t>(i);
    }
    SIHPS_ASSERT_TRUE(col_x01 >= 0);
    SIHPS_ASSERT_TRUE(row_x48 >= 0);

    bool found = false;
    for (const auto& t : model.constraint_triplets) {
        if (t.row == row_x48 && t.col == col_x01) {
            SIHPS_ASSERT_NEAR(t.value, 0.301, 1e-12);
            found = true;
        }
    }
    SIHPS_ASSERT_TRUE(found);
}

SIHPS_TEST(mps_reader_undeclared_row_reference_throws) {
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "sihps_malformed_test.mps";
    {
        std::ofstream out(tmp);
        out << "NAME          BAD\n"
               "ROWS\n"
               " N  COST\n"
               " L  R1\n"
               "COLUMNS\n"
               "    X1        COST             1.0        R1               1.0\n"
               "    X1        GHOST            1.0\n" // GHOST was never declared in ROWS
               "ENDATA\n";
    }
    SIHPS_ASSERT_THROWS(read_mps_file(tmp.string()));
    fs::remove(tmp);
}

SIHPS_TEST(mps_reader_missing_file_throws) {
    SIHPS_ASSERT_THROWS(read_mps_file("this/path/does/not/exist.mps"));
}
