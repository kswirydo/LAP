#include "common.h"
#ifndef HIPBLAS_H
#define HIPBLAS_H



void initialize_handles();


void analyze_spmv(const int n, 
                  const int nnz, 
                  int *ia, 
                  int *ja, 
                  double *a, 
                  const double *x, 
                  double *result,
                 char * option);


void initialize_and_analyze_L_and_U_solve(const int n, 
                                              const int nnzL, 
                                              int *lia, 
                                              int *lja, 
                                              double *la,
                                              const int nnzU, 
                                              int *uia, 
                                              int *uja, 
                                              double *ua);


double hip_dot (const int n, const double *v, const double *w);

void hip_scal (const int n, const double alpha, double *v);

void hip_axpy (const int n, const double alpha, const double *x, double *y);

void hip_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const double*al, const double *bet, const char * kind);

void hip_lower_triangular_solve(const int n, const int nnzL, const int *lia, const int *lja, const double *la,const double *diagonal, const double *x, double *result);

void hip_upper_triangular_solve(const int n, const int nnzU, const int *uia, const int *uja, const double *ua, const double *diagonal, const double *x, double *result);

void hip_vec_vec(const int n, const double * x, const double * y, double *res);

void hip_vector_reciprocal(const int n, const double *v, double *res);

void hip_vector_sqrt(const int n, const double *v, double *res);

void hip_vec_copy(const int n, const double *src, double *dest);

void hip_vec_zero(const int n, double *vec);
#endif
