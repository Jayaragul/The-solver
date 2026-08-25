#include "sankhya_lp.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

typedef struct { char* name; char type; } RowDef;
typedef struct { size_t row; size_t column; double value; } Entry;
typedef struct { char* name; double value; } Pair;
typedef struct { char* type; char* column; double value; int has_value; } BoundDef;

static char* copy_string(const char* source) {
    size_t length;
    char* result;
    if (source == NULL) return NULL;
    length = strlen(source);
    result = (char*)malloc(length + 1);
    if (result != NULL) memcpy(result, source, length + 1);
    return result;
}

static void strip_quotes(char* token) {
    size_t length;
    if (token == NULL) return;
    length = strlen(token);
    if (length >= 2 && ((token[0] == '\'' && token[length - 1] == '\'') ||
                        (token[0] == '"' && token[length - 1] == '"'))) {
        memmove(token, token + 1, length - 2);
        token[length - 2] = '\0';
    }
}

static int find_name(char** names, size_t count, const char* name) {
    size_t i;
    for (i = 0; i < count; ++i) if (strcmp(names[i], name) == 0) return (int)i;
    return -1;
}

static int add_name(char*** names, size_t* count, size_t* capacity, const char* name) {
    char** expanded;
    if (find_name(*names, *count, name) >= 0) return find_name(*names, *count, name);
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 16 : *capacity * 2;
        expanded = (char**)realloc(*names, next * sizeof(char*));
        if (expanded == NULL) return -1;
        *names = expanded; *capacity = next;
    }
    (*names)[*count] = copy_string(name);
    if ((*names)[*count] == NULL) return -1;
    return (int)(*count)++;
}

static void free_names(char** names, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) free(names[i]);
    free(names);
}

static int add_entry(Entry** entries, size_t* count, size_t* capacity, Entry value) {
    Entry* expanded;
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 64 : *capacity * 2;
        expanded = (Entry*)realloc(*entries, next * sizeof(Entry));
        if (expanded == NULL) return 0;
        *entries = expanded; *capacity = next;
    }
    (*entries)[(*count)++] = value;
    return 1;
}

static int add_pair(Pair** pairs, size_t* count, size_t* capacity, const char* name, double value) {
    Pair* expanded;
    size_t i;
    for (i = 0; i < *count; ++i) if (strcmp((*pairs)[i].name, name) == 0) return 1;
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 16 : *capacity * 2;
        expanded = (Pair*)realloc(*pairs, next * sizeof(Pair));
        if (expanded == NULL) return 0;
        *pairs = expanded; *capacity = next;
    }
    (*pairs)[*count].name = copy_string(name);
    if ((*pairs)[*count].name == NULL) return 0;
    (*pairs)[(*count)++].value = value;
    return 1;
}

static int entry_compare(const void* left, const void* right) {
    const Entry* a = (const Entry*)left;
    const Entry* b = (const Entry*)right;
    if (a->column < b->column) return -1;
    if (a->column > b->column) return 1;
    if (a->row < b->row) return -1;
    if (a->row > b->row) return 1;
    return 0;
}

void sankhya_lp_model_init(SankhyaLPModel* model) {
    if (model != NULL) memset(model, 0, sizeof(*model));
}

void sankhya_lp_model_destroy(SankhyaLPModel* model) {
    size_t i;
    if (model == NULL) return;
    sankhya_csc_destroy(&model->A);
    free(model->objective); free(model->row_lower); free(model->row_upper);
    free(model->column_lower); free(model->column_upper); free(model->variable_type);
    if (model->row_names != NULL) for (i = 0; i < model->A.rows; ++i) free(model->row_names[i]);
    if (model->column_names != NULL) for (i = 0; i < model->A.columns; ++i) free(model->column_names[i]);
    free(model->row_names); free(model->column_names);
    sankhya_lp_model_init(model);
}

SankhyaStatus sankhya_lp_read_mps(const char* path, SankhyaLPModel* model) {
    enum { NONE, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADRATIC } section = NONE;
    FILE* file;
    char line[8192];
    char model_name[256] = "";
    RowDef* rows = NULL; size_t row_count = 0, row_capacity = 0;
    char** columns = NULL; size_t column_count = 0, column_capacity = 0;
    Entry* entries = NULL; size_t entry_count = 0, entry_capacity = 0;
    Pair* rhs = NULL; size_t rhs_count = 0, rhs_capacity = 0;
    Pair* ranges = NULL; size_t range_count = 0, range_capacity = 0;
    BoundDef* bounds = NULL; size_t bound_count = 0, bound_capacity = 0;
    double* raw_objective = NULL; unsigned char* raw_type = NULL; size_t raw_capacity = 0;
    char* objective_name = NULL; int integer_mode = 0; int maximize = 0;
    SankhyaLPModel candidate; size_t i; int success = 0;
    if (path == NULL || model == NULL) return SANKHYA_INVALID_ARGUMENT;
    file = fopen(path, "r");
    if (file == NULL) return SANKHYA_INVALID_ARGUMENT;
    sankhya_lp_model_init(&candidate);
    while (fgets(line, sizeof(line), file) != NULL) {
        char* token[8]; size_t token_count = 0; char* current; int is_header;
        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r') continue;
        /* MPS separates a section header from a data record by column 1:
           headers begin there, data records are indented.  Without this test a
           data line whose RHS/RANGES/BOUNDS set is itself named "RHS",
           "RANGES" or "BOUNDS" - the overwhelmingly common convention, and
           what most of Netlib uses - is mistaken for a second section header
           and silently discarded, leaving every right-hand side at zero. */
        is_header = (line[0] != ' ' && line[0] != '\t');
        current = strtok(line, " \t\r\n");
        while (current != NULL && token_count < 8) { strip_quotes(current); token[token_count++] = current; current = strtok(NULL, " \t\r\n"); }
        if (token_count == 0) continue;
        if (is_header && strcasecmp(token[0], "NAME") == 0) { if (token_count > 1) strncpy(model_name, token[1], sizeof(model_name) - 1); continue; }
        if (is_header && strcasecmp(token[0], "OBJSENSE") == 0) { section = NONE; continue; }
        if (strcasecmp(token[0], "MAX") == 0 && section == NONE) { maximize = 1; continue; }
        if (strcasecmp(token[0], "MIN") == 0 && section == NONE) { maximize = 0; continue; }
        if (is_header && strcasecmp(token[0], "ROWS") == 0) { section = ROWS; continue; }
        if (is_header && strcasecmp(token[0], "COLUMNS") == 0) { section = COLUMNS; continue; }
        if (is_header && strcasecmp(token[0], "RHS") == 0) { section = RHS; continue; }
        if (is_header && strcasecmp(token[0], "RANGES") == 0) { section = RANGES; continue; }
        if (is_header && strcasecmp(token[0], "BOUNDS") == 0) { section = BOUNDS; continue; }
        /* The public sk_read_mps bridge performs the second QPS pass.  This
           LP reader must nevertheless leave BOUNDS mode before it encounters
           a quadratic section. */
        if (strcasecmp(token[0], "QUADOBJ") == 0 || strcasecmp(token[0], "QMATRIX") == 0 ||
            strcasecmp(token[0], "QSECTION") == 0) { section = QUADRATIC; continue; }
        if (is_header && strcasecmp(token[0], "ENDATA") == 0) break;
        if (section == ROWS) {
            RowDef* expanded;
            if (token_count < 2 || (token[0][0] != 'N' && token[0][0] != 'L' && token[0][0] != 'G' && token[0][0] != 'E')) goto parse_failure;
            if (token[0][0] == 'N') { if (objective_name == NULL) objective_name = copy_string(token[1]); continue; }
            if (row_count == row_capacity) { size_t next = row_capacity ? row_capacity * 2 : 16; expanded = (RowDef*)realloc(rows, next * sizeof(RowDef)); if (!expanded) goto parse_failure; rows = expanded; row_capacity = next; }
            rows[row_count].name = copy_string(token[1]); rows[row_count].type = token[0][0]; if (!rows[row_count].name) goto parse_failure; ++row_count;
        } else if (section == COLUMNS) {
            int column; size_t pair;
            if (token_count >= 3 && strcasecmp(token[1], "MARKER") == 0) { if (strcasecmp(token[2], "INTORG") == 0) integer_mode = 1; if (strcasecmp(token[2], "INTEND") == 0) integer_mode = 0; continue; }
            if (token_count < 3) goto parse_failure;
            column = add_name(&columns, &column_count, &column_capacity, token[0]); if (column < 0) goto parse_failure;
            if ((size_t)column >= raw_capacity) {
                size_t next = raw_capacity ? raw_capacity * 2 : 16;
                double* next_objective; unsigned char* next_type;
                while (next <= (size_t)column) next *= 2;
                next_objective = (double*)realloc(raw_objective, next * sizeof(double));
                next_type = (unsigned char*)realloc(raw_type, next * sizeof(unsigned char));
                if (next_objective == NULL || next_type == NULL) { free(next_objective); free(next_type); goto parse_failure; }
                memset(next_objective + raw_capacity, 0, (next - raw_capacity) * sizeof(double));
                memset(next_type + raw_capacity, 0, (next - raw_capacity) * sizeof(unsigned char));
                raw_objective = next_objective; raw_type = next_type; raw_capacity = next;
            }
            if (integer_mode) raw_type[column] = SANKHYA_INTEGER;
            for (pair = 1; pair + 1 < token_count; pair += 2) {
                Entry entry; int row = -1; size_t row_index; double value = strtod(token[pair + 1], NULL);
                if (!isfinite(value)) goto parse_failure;
                if (objective_name != NULL && strcmp(objective_name, token[pair]) == 0) {
                    raw_objective[column] += value;
                    continue;
                }
                for (row_index = 0; row_index < row_count; ++row_index) if (strcmp(rows[row_index].name, token[pair]) == 0) { row = (int)row_index; break; }
                if (row < 0) goto parse_failure;
                entry.row = (size_t)row; entry.column = (size_t)column; entry.value = value;
                if (!add_entry(&entries, &entry_count, &entry_capacity, entry)) goto parse_failure;
            }
        } else if (section == RHS || section == RANGES) {
            Pair* target = section == RHS ? rhs : ranges; size_t* count = section == RHS ? &rhs_count : &range_count; size_t* capacity = section == RHS ? &rhs_capacity : &range_capacity; size_t pair;
            /* The leading set name is optional in RHS and RANGES, and the
               Netlib distribution omits it.  A record is therefore
               "[set] row value [row value]": an odd token count carries the
               set name, an even one starts straight at the first row.  Getting
               this wrong shifts every field by one and silently drops the whole
               right-hand side, leaving a feasible-looking but wrong model. */
            size_t first = (token_count % 2 == 1) ? 1 : 0;
            for (pair = first; pair + 1 < token_count; pair += 2) if (!add_pair(&target, count, capacity, token[pair], strtod(token[pair + 1], NULL))) goto parse_failure;
            if (section == RHS) rhs = target; else ranges = target;
        } else if (section == BOUNDS) {
            BoundDef* expanded; if (token_count < 3) goto parse_failure;
            if (bound_count == bound_capacity) { size_t next = bound_capacity ? bound_capacity * 2 : 16; expanded = (BoundDef*)realloc(bounds, next * sizeof(BoundDef)); if (!expanded) goto parse_failure; bounds = expanded; bound_capacity = next; }
            bounds[bound_count].type = copy_string(token[0]); bounds[bound_count].column = copy_string(token[2]); bounds[bound_count].has_value = token_count > 3; bounds[bound_count].value = token_count > 3 ? strtod(token[3], NULL) : 0.0; if (!bounds[bound_count].type || !bounds[bound_count].column) goto parse_failure; ++bound_count;
            if (find_name(columns, column_count, token[2]) < 0 && add_name(&columns, &column_count, &column_capacity, token[2]) < 0) goto parse_failure;
        }
    }
    fclose(file); file = NULL;
    if (row_count == 0 || column_count == 0 || objective_name == NULL) goto parse_failure;
    candidate.A.rows = row_count; candidate.A.columns = column_count;
    qsort(entries, entry_count, sizeof(Entry), entry_compare);
    candidate.A.column_offsets = (size_t*)calloc(column_count + 1, sizeof(size_t));
    candidate.A.row_indices = (int*)malloc(entry_count * sizeof(int)); candidate.A.values = (double*)malloc(entry_count * sizeof(double));
    candidate.objective = (double*)calloc(column_count, sizeof(double)); candidate.row_lower = (double*)malloc(row_count * sizeof(double)); candidate.row_upper = (double*)malloc(row_count * sizeof(double)); candidate.column_lower = (double*)calloc(column_count, sizeof(double)); candidate.column_upper = (double*)malloc(column_count * sizeof(double)); candidate.variable_type = (unsigned char*)calloc(column_count, sizeof(unsigned char));
    candidate.row_names = (char**)calloc(row_count, sizeof(char*)); candidate.column_names = (char**)calloc(column_count, sizeof(char*));
    if (!candidate.A.column_offsets || (entry_count && (!candidate.A.row_indices || !candidate.A.values)) || !candidate.objective || !candidate.row_lower || !candidate.row_upper || !candidate.column_lower || !candidate.column_upper || !candidate.variable_type || !candidate.row_names || !candidate.column_names) goto parse_failure;
    for (i = 0; i < row_count; ++i) { candidate.row_names[i] = copy_string(rows[i].name); candidate.row_lower[i] = -INFINITY; candidate.row_upper[i] = INFINITY; if (rows[i].type == 'L') candidate.row_upper[i] = 0.0; if (rows[i].type == 'G') candidate.row_lower[i] = 0.0; if (rows[i].type == 'E') candidate.row_lower[i] = candidate.row_upper[i] = 0.0; }
    for (i = 0; i < column_count; ++i) { candidate.column_names[i] = copy_string(columns[i]); candidate.column_upper[i] = INFINITY; candidate.objective[i] = raw_objective[i]; candidate.variable_type[i] = raw_type[i]; }
    for (i = 0; i < entry_count; ) { size_t begin = i; double value = 0.0; while (i < entry_count && entries[i].column == entries[begin].column && entries[i].row == entries[begin].row) value += entries[i++].value; if (value != 0.0) { candidate.A.column_offsets[entries[begin].column + 1]++; candidate.A.row_indices[candidate.A.nonzeros] = (int)entries[begin].row; candidate.A.values[candidate.A.nonzeros++] = value; } }
    for (i = 0; i < column_count; ++i) candidate.A.column_offsets[i + 1] += candidate.A.column_offsets[i];
    for (i = 0; i < rhs_count; ++i) { int row; for (row = 0; row < (int)row_count; ++row) if (strcmp(rhs[i].name, rows[row].name) == 0) { double base = rhs[i].value; if (rows[row].type == 'L') candidate.row_upper[row] = base; else if (rows[row].type == 'G') candidate.row_lower[row] = base; else candidate.row_lower[row] = candidate.row_upper[row] = base; break; } if (objective_name && strcmp(rhs[i].name, objective_name) == 0) candidate.objective_offset = -rhs[i].value; }
    for (i = 0; i < range_count; ++i) {
        int row;
        for (row = 0; row < (int)row_count; ++row) if (strcmp(ranges[i].name, rows[row].name) == 0) {
            double width = ranges[i].value;
            double base = isfinite(candidate.row_lower[row]) ? candidate.row_lower[row] : candidate.row_upper[row];
            if (rows[row].type == 'L') candidate.row_lower[row] = base - fabs(width);
            else if (rows[row].type == 'G') candidate.row_upper[row] = base + fabs(width);
            else if (width >= 0.0) candidate.row_upper[row] = base + fabs(width);
            else candidate.row_lower[row] = base - fabs(width);
            break;
        }
    }
    for (i = 0; i < bound_count; ++i) { int col = find_name(columns, column_count, bounds[i].column); if (col < 0) continue; if (strcmp(bounds[i].type, "LO") == 0) candidate.column_lower[col] = bounds[i].value; else if (strcmp(bounds[i].type, "UP") == 0) candidate.column_upper[col] = bounds[i].value; else if (strcmp(bounds[i].type, "FX") == 0) candidate.column_lower[col] = candidate.column_upper[col] = bounds[i].value; else if (strcmp(bounds[i].type, "FR") == 0) candidate.column_lower[col] = -INFINITY; else if (strcmp(bounds[i].type, "MI") == 0) candidate.column_lower[col] = -INFINITY; else if (strcmp(bounds[i].type, "BV") == 0) { candidate.column_lower[col] = 0.0; candidate.column_upper[col] = 1.0; candidate.variable_type[col] = SANKHYA_BINARY; } else if (strcmp(bounds[i].type, "LI") == 0) { candidate.column_lower[col] = bounds[i].value; candidate.variable_type[col] = SANKHYA_INTEGER; } else if (strcmp(bounds[i].type, "UI") == 0) { candidate.column_upper[col] = bounds[i].value; candidate.variable_type[col] = SANKHYA_INTEGER; } }
    if (maximize) { for (i = 0; i < column_count; ++i) candidate.objective[i] = -candidate.objective[i]; candidate.objective_offset = -candidate.objective_offset; }
    if (sankhya_csc_validate(&candidate.A) != SANKHYA_OK) goto parse_failure;
    sankhya_lp_model_destroy(model); *model = candidate; success = 1;
    goto cleanup;
parse_failure:
    if (file) fclose(file); sankhya_lp_model_destroy(&candidate);
cleanup:
    if (rows) { for (i = 0; i < row_count; ++i) free(rows[i].name); free(rows); }
    free_names(columns, column_count); free(entries);
    free(raw_objective); free(raw_type);
    if (rhs) { for (i = 0; i < rhs_count; ++i) free(rhs[i].name); free(rhs); }
    if (ranges) { for (i = 0; i < range_count; ++i) free(ranges[i].name); free(ranges); }
    if (bounds) { for (i = 0; i < bound_count; ++i) { free(bounds[i].type); free(bounds[i].column); } free(bounds); }
    free(objective_name);
    return success ? SANKHYA_OK : SANKHYA_INVALID_ARGUMENT;
}

double sankhya_lp_objective(const SankhyaLPModel* model, const double* x) {
    size_t i; double value;
    if (model == NULL || x == NULL) return NAN;
    value = model->objective_offset;
    for (i = 0; i < model->A.columns; ++i) value += model->objective[i] * x[i];
    return value;
}

double sankhya_lp_max_primal_violation(const SankhyaLPModel* model, const double* x) {
    size_t i; double* activity; double maximum = 0.0;
    if (model == NULL || x == NULL) return NAN;
    activity = (double*)calloc(model->A.rows, sizeof(double)); if (!activity) return NAN;
    sankhya_csc_matvec(&model->A, x, activity);
    for (i = 0; i < model->A.rows; ++i) { if (isfinite(model->row_lower[i]) && model->row_lower[i] - activity[i] > maximum) maximum = model->row_lower[i] - activity[i]; if (isfinite(model->row_upper[i]) && activity[i] - model->row_upper[i] > maximum) maximum = activity[i] - model->row_upper[i]; }
    for (i = 0; i < model->A.columns; ++i) { if (isfinite(model->column_lower[i]) && model->column_lower[i] - x[i] > maximum) maximum = model->column_lower[i] - x[i]; if (isfinite(model->column_upper[i]) && x[i] - model->column_upper[i] > maximum) maximum = x[i] - model->column_upper[i]; }
    free(activity); return maximum > 0.0 ? maximum : 0.0;
}
