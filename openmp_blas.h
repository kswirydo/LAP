//prototypes
//
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

double openmp_dot (const int n, const double *v, const double *w);

void openmp_scal (const int n, const double alpha, double *v);

void openmp_axpy (const int n, const double alpha, const double *x, double *y);

void openmp_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const  double*al, const double *bet);

void openmp_lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const double  *la,const double * diag, const double *x, double *result);

void openmp_upper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const double *ua, const double * diag,const double *x, double *result);

void openmp_vec_vec(const int n, const double * x, const double * y, double *res);

void openmp_vector_sqrt(const int n, const double *v, double *res);

void openmp_vector_reciprocal(const int n, const double *v, double *res);

void openmp_vec_copy(const int n, const double *src, double *dest);

void openmp_vec_zero(const int n, double *vec);
