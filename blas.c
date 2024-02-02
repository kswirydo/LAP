#include "simple_blas.h"
#include "cuda_blas.h"
#include "hip_blas.h"
#include "openmp_blas.h"
#include "blas.h"
#include "common.h"

double dot (const int n, const double *v, const double *w){
  double d;
#if NOACC
  d = simple_dot (n, v, w);
#elif  CUDA
  d = cuda_dot (n, v, w);
#elif OPENMP
  d = openmp_dot (n, v, w);
#elif HIP
  d = hip_dot (n, v, w);
#endif
  return d;
}

void axpy (const int n, const double alpha, double *x, double *y){
#if NOACC
  simple_axpy (n, alpha, x, y);
#elif CUDA
  cuda_axpy (n, alpha, x, y);
#elif OPENMP
  openmp_axpy (n, alpha, x, y);
#elif HIP
  hip_axpy (n, alpha, x, y);
#endif
}

void scal (const int n, const double alpha, double *v){
#if NOACC
  simple_scal (n, alpha, v);
#elif CUDA
  cuda_scal (n, alpha, v);
#elif OPENMP
  openmp_scal (n, alpha, v);
#elif HIP
  hip_scal (n, alpha, v);
#endif
}

void csr_matvec(const int n, 
                const int nnz, 
                const int *ia, 
                const int *ja, 
                const double *a, 
                const double *x, 
                double *result, 
                const double *al, 
                const double *bet,
                const char *kind){
#if NOACC
  simple_csr_matvec(n, nnz, ia, ja, a, x, result, al, bet);
#elif CUDA
  cuda_csr_matvec(n, nnz, ia, ja, a, x, result, al, bet);
#elif OPENMP
  openmp_csr_matvec(n, nnz, ia, ja, a, x, result, al, bet);
#elif HIP
  hip_csr_matvec(n, nnz, ia, ja, a, x, result, al, bet, kind);
#endif
}


void lower_triangular_solve(const int n, 
                            const int nnz, 
                            const int *lia, 
                            const int *lja, 
                            const double *la,
                            const double * diag, 
                            const double *x, 
                            double *result){
#if NOACC
  simple_lower_triangular_solve(n, nnz, lia, lja, la, diag, x, result);
#elif CUDA
  cuda_lower_triangular_solve(n, nnz, lia, lja, la, diag, x, result);
#elif OPENMP
  openmp_lower_triangular_solve(n, nnz, lia, lja, la, diag, x, result);
#elif HIP
  hip_lower_triangular_solve(n, nnz, lia, lja, la, diag, x, result);
#endif
}

void upper_triangular_solve(const int n, 
                            const int nnz, 
                            const int *uia, 
                            const int *uja, 
                            const double *ua,
                            const double *diag, 
                            const double *x, 
                            double *result){
#if NOACC
  simple_upper_triangular_solve(n, nnz, uia, uja, ua, diag, x, result);
#elif CUDA
  cuda_upper_triangular_solve(n, nnz, uia, uja, ua, diag, x, result);
#elif OPENMP
  openmp_upper_triangular_solve(n, nnz, uia, uja, ua, diag, x, result);
#elif HIP
  hip_upper_triangular_solve(n, nnz, uia, uja, ua, diag, x, result);
#endif
}

void ichol(const int *ia, const int *ja, double *a, const int nnzA, pdata *prec_data, double *x, double *y){
#if NOACC
  simple_ichol( ia, ja, a, nnzA, prec_data, x, y);
#elif CUDA
  cuda_ichol( ia, ja, a, nnzA, prec_data, x, y);
#elif OPENMP
  openmp_ichol( ia, ja, a, nnzA, prec_data, x, y);
#elif HIP
  hip_ichol( ia, ja, a, nnzA, prec_data, x, y);
#endif
}

void vec_vec(const int n, const double *x, double *y, double *res){
#if NOACC
  simple_vec_vec(n, x, y, res);
#elif CUDA
  cuda_vec_vec(n, x, y, res);
#elif OPENMP
  openmp_vec_vec(n, x, y, res);
#elif HIP
  hip_vec_vec(n, x, y, res);
#endif
}


void vector_reciprocal(const int n, const double *v, double *res){
#if NOACC
  simple_vector_reciprocal(n, v, res);
#elif CUDA
  cuda_vector_reciprocal(n, v, res);
#elif OPENMP
  openmp_vector_reciprocal(n, v, res);
#elif HIP
  hip_vector_reciprocal(n, v, res);
#endif
}


void vector_sqrt(const int n, const double *v, double *res){
#if NOACC
  simple_vector_sqrt(n, v, res);
#elif CUDA
  cuda_vector_sqrt(n, v, res);
#elif OPENMP
  openmp_vector_sqrt(n, v, res);
#elif HIP
  hip_vector_sqrt(n, v, res);
#endif
}


void vec_copy(const int n, double *src, double *dest){
#if NOACC
  simple_vec_copy(n, src, dest);
#elif CUDA
  cuda_vec_copy(n, src, dest);
#elif OPENMP
  openmp_vec_copy(n, src, dest);
#elif HIP
  hip_vec_copy(n, src, dest);
#endif
}

void vec_zero(const int n, double *vec){
#if NOACC
  simple_vec_zero(n, vec);
#elif CUDA
  cuda_vec_zero(n, vec);
#elif OPENMP
  openmp_vec_zero(n, vec);
#elif HIP
  hip_vec_zero(n, vec);
#endif
}
