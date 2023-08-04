
#include "common.h"
#ifndef CUDABLAS_H
#define CUDABLAS_H
double cuda_dot (const int n, const double *v, const double *w);

void cuda_scal (const int n, const double alpha, double *v);

void cuda_axpy (const int n, const double alpha, const double *x, double *y);

void cuda_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const  double*al, const double *bet);

void cuda_lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const double  *la,const double * diag, const double *x, double *result);

void cuda_upper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const double *ua, const double * diag,const double *x, double *result);

void cuda_vec_vec(const int n, const double * x, const double * y, double *res);

void cuda_vector_sqrt(const int n, const double *v, double *res);

void cuda_vector_reciprocal(const int n, const double *v, double *res);

void cuda_vec_copy(const int n, const double *src, double *dest);

void cuda_vec_zero(const int n, double *vec);
void initialize_handles();


void initialize_spmv_buffer(const int n, 
                            const int nnz, 
                            int *ia, 
                            int *ja, 
                            double *a, 
                            const double *x, 
                            double *result, 
                            double *al, 
                            double *bet);


void initialize_and_analyze_L_and_U_solve(const int n, 
                                          const int nnzL, 
                                          int *lia, 
                                          int *lja, 
                                          double *la,
                                          const int nnzU, 
                                          int *uia, 
                                          int *uja, 
                                          double *ua);


void initialize_L_and_U_descriptors(const int n, 
                                  const int nnzL, 
                                  int *lia, 
                                  int *lja, 
                                  double *la,
                                  const int nnzU, 
                                  int *uia, 
                                  int *uja, 
                                  double *ua);
void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      double *a);

#endif
