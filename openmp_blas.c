#include "openmp_blas.h"
#include <math.h>


void openmp_scal (const int n, const double alpha, double *v){

  int i;

  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(alpha) map(tofrom:v[0:n])
  for (i=0; i<n; ++i){
    v[i] *= alpha;
  }
}

void openmp_axpy (const int n, const double alpha, const double *x, double *y){
  int i;

  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:x[0:n]) map(tofrom:y[0:n])
  // #pragma omp teams distribute parallel for  schedule(static) private(i) 
  for (i=0; i<n;++i){
    y[i] += alpha*x[i];
  }
}

void openmp_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result,const  double*al, const double *bet){
  double alpha = *al;
  double beta = *bet;
#pragma omp parallel for default(none) firstprivate(result, ia, ja, a, x, alpha, beta, n)
  {
    for (int i=0; i<n; ++i){
      int lb = ia[i];
      int ub = ia[(((i << 1) + 3) >> 1)];
      double s = result[i] * beta;  
#pragma omp simd reduction(+:s)
      for (int j=lb; j<ub; j++){
        int col = ja[j];
        s += (alpha*a[j]*x[col]);
      }
      result[i] = s;
    }
  }  
}

void openmp_lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const double *la,const double *diagonal, const double *x, double *result){
  //compute result = L^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go through each row (starting from 0)
  int i, j, col;
#pragma omp target teams distribute map(to:lia[0:n+1],lja[0:nnz],la[0:nnz],x[0:n], diagonal[0:n])  map(tofrom:result[0:n])
  for (i=0; i<n; ++i){
    double s =0.0;
#pragma omp simd private(j, col) reduction(+:s)
    for ( j=lia[i]; j<lia[i+1]; ++j){
      col = lja[j];

      s += (-1.0)*la[j]*result[col]; 
    }

    result[i] =(s+x[i])/diagonal[i];
  }
}


void openmp_upper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const double *ua, const double *diagonal, const double *x, double *result){
  //compute result = U^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go through each row (starting from the last row)
  int i,j,col;
  double s; 
 //this kind of works but the result is non deterministic 
  // #pragma omp target teams distribute ordered map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  // #pragma omp target teams distribute map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  // #pragma omp target map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
#pragma omp target teams distribute map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  for (i=n-1; i>=0; --i) {
    s=0.0;

    result[i] = 0.0f;
    #pragma omp simd private(j, col) reduction(+:s)
    //map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz],result[0:n]) map(from:s)
    for (j=uia[i]; j<uia[i+1]; ++j){
      col = uja[j];
      s += (-1.0)*ua[j]*result[col];
    }
    //#pragma omp ordered
    result[i] =(s+x[i])/diagonal[i]; 
  }
}

//not std blas but needed and embarassingly parallel 

//simple vec-vec computes an element-wise product (needed for scaling)
//
void openmp_vec_vec(const int n, const double * x, const double * y, double *res){
  int i;
  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:x[0:n], y[0:n]) map(from:res[0:n])
  for (i=0; i<n; ++i){
    res[i] = x[i]*y[i];
  }
}

//vector reciprocal computes 1./d 
//
void openmp_vector_reciprocal(const int n, const double *v, double *res){
  int i;
  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:v[0:n]) map(from:res[0:n])
  for (i=0; i<n; ++i){
    if  (v[i] != 0.0 )res[i] = 1.0f/v[i];
    else res[i] = 0.0f;
  }
}

//vector sqrt takes an sqrt from each vector entry 


void openmp_vector_sqrt(const int n, const double *v, double *res){
  int i;
  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:v[0:n]) map(from:res[0:n])
  for (i=0; i<n; ++i){
    if  (v[i] >= 0.0) res[i] = sqrt(v[i]);
    else res[i] = 0.0f;
  }
}

void openmp_vec_copy(const int n, const double *src, double *dest){
  int i;
  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:src[0:n]) map(from:dest[0:n])
  for (i=0; i<n; ++i){
    dest[i] = src[i];  
  }
}


void openmp_vec_zero(const int n, double *vec){
  int i;
  #pragma omp target teams distribute parallel for  schedule(static) private(i)  map(tofrom:vec[0:n])
  for (i=0; i<n; ++i){
    vec[i] = 0.0f;  
  }
}
double openmp_dot (const int n, const double *v, const double *w){
  double sum = 0.0;
  int i;
  #pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:v[0:n], w[0:n]) reduction(+:sum)
  for (i=0; i<n; ++i){
    sum += v[i]*w[i];
  }
  return sum;
}
//add vec_copy
