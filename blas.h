#pragma once
double dot (const int n, const double *v, const double *w);
void axpy (const int n, const double alpha, double *x, double *y);
void scal (const int n, const double alpha, double *v);
void csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const  double*al, const double *bet);
void lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const double *la,const double * diag, const double *x, double *result);
void upper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const double *ua,const double * diag, const double *x, double *result);
void vec_vec(const int n, const double * x, double * y, double *res);
void vector_reciprocal(const int n, const double *v, double *res);
void vector_sqrt(const int n, const double *v, double *res);
void vec_copy(const int n, double *src, double *dest);
void vec_zero(const int n, double *vec);
