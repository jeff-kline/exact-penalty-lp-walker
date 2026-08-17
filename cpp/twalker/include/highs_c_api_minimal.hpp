#pragma once

// Minimal declarations from HiGHS' public C API.  The installed highspy
// library uses 32-bit HighsInt (verified with Highs_getSizeofHighsInt()).
extern "C" {
using HighsInt = int;
void *Highs_create(void);
void Highs_destroy(void *highs);
HighsInt Highs_setBoolOptionValue(void *highs, const char *option,
                                  HighsInt value);
HighsInt Highs_setIntOptionValue(void *highs, const char *option,
                                 HighsInt value);
HighsInt Highs_setDoubleOptionValue(void *highs, const char *option,
                                    double value);
HighsInt Highs_setStringOptionValue(void *highs, const char *option,
                                    const char *value);
HighsInt Highs_passLp(
    void *highs, HighsInt num_col, HighsInt num_row, HighsInt num_nz,
    HighsInt a_format, HighsInt sense, double offset, const double *col_cost,
    const double *col_lower, const double *col_upper, const double *row_lower,
    const double *row_upper, const HighsInt *a_start,
    const HighsInt *a_index, const double *a_value);
HighsInt Highs_passModel(
    void *highs, HighsInt num_col, HighsInt num_row, HighsInt num_nz,
    HighsInt q_num_nz, HighsInt a_format, HighsInt q_format, HighsInt sense,
    double offset, const double *col_cost, const double *col_lower,
    const double *col_upper, const double *row_lower, const double *row_upper,
    const HighsInt *a_start, const HighsInt *a_index, const double *a_value,
    const HighsInt *q_start, const HighsInt *q_index, const double *q_value,
    const HighsInt *integrality);
HighsInt Highs_run(void *highs);
HighsInt Highs_getModelStatus(const void *highs);
HighsInt Highs_getIntInfoValue(const void *highs, const char *info,
                               HighsInt *value);
HighsInt Highs_getSolution(const void *highs, double *col_value,
                           double *col_dual, double *row_value,
                           double *row_dual);
HighsInt Highs_getBasis(const void *highs, HighsInt *col_status,
                        HighsInt *row_status);
HighsInt Highs_setBasis(void *highs, const HighsInt *col_status,
                        const HighsInt *row_status);
HighsInt Highs_getBasicVariables(const void *highs,
                                 HighsInt *basic_variables);
HighsInt Highs_getBasisSolve(const void *highs, const double *rhs,
                             double *solution_vector,
                             HighsInt *solution_num_nz,
                             HighsInt *solution_indices);
HighsInt Highs_getBasisTransposeSolve(const void *highs, const double *rhs,
                                      double *solution_vector,
                                      HighsInt *solution_num_nz,
                                      HighsInt *solution_indices);
HighsInt Highs_changeRowsBoundsByRange(void *highs, HighsInt from_row,
                                       HighsInt to_row, const double *lower,
                                       const double *upper);
HighsInt Highs_changeColsBoundsByRange(void *highs, HighsInt from_col,
                                       HighsInt to_col, const double *lower,
                                       const double *upper);
HighsInt Highs_changeColsCostByRange(void *highs, HighsInt from_col,
                                     HighsInt to_col, const double *cost);
HighsInt Highs_changeCoeff(void *highs, HighsInt row, HighsInt col,
                           double value);
}
