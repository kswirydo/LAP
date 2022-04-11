#include "simple_blas.h":1

double dot (const int n, const double *v, const double *w){
  double d;
#if NOACC
  d = simple_dot (n,v,w);
#endif
  return d;
}


void axpy (const int n, const double alpha, double *x, double *y){
#if NOACC
  simple_axpy (n, alpha,x, y);
#endif
}

void scal (const int n, const double alpha, double *v){
#if NOACC
  simple_scal (n, alpha,v);
#endif
}

void csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const int *a, const double *x, double *result){
#if NOACC
  simple_csr_matvec(n, nnz, ia, ja, a, x, result);
#endif
}

void lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const int *la, const double *x, double *result){
#if NOACC
  simple_lower_triangular_solve(n, nnz, lia, lja, la, x, result);
#endif
}

void uppper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const int *ua, const double *x, double *result){
#if NOACC
  simple_upper_triangular_solve(n, nnz, uia, uja, ua, x, result);
#endif
}

void vec_vec(const int n, const double * x, const double * y, double *res){
#if NOACC
simple_vec_vec(n, x, const y, res);
#endif
}


void vector_reciprocal(const int n, const double *v, double *res){
#if NOACC
simple_vector_reciprocal(n, v, res);
#endif
}


void vector_sqrt(const int n, const double *v, double *res){
#if NOACC
simple_vector_sqrt(n, v, res);
#endif
}


void vec_copy(const int n, double *src, double *dest){
#if NOACC
simple_vec_copy(n, v, src, dest);
#endif
}