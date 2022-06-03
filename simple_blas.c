#include "simple_blas.h"
#include <math.h>
double simple_dot (const int n, const double *v, const double *w){
  double sum = 0.0f;
  for (int i=0; i<n; ++i){
    sum += v[i]*w[i];
  }
  return sum;
}

void simple_scal (const int n, const double alpha, double *v){
  for (int i=0; i<n; ++i){
    v[i] *= alpha;
  }
}

void simple_axpy (const int n, const double alpha, const double *x, double *y){
  for (int i=0; i<n;++i){
    y[i] += alpha*x[i];
  }
}

void simple_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result,const  double*al, const double *bet){
double alpha = *al;
double beta = *bet;
//printf ("alpha = %f, beta = %f\n", alpha, beta);
  //intialize result to 0
/* 
 for (int i=0; i<n; ++i){
    result[i] = 0.0f;
  }
*/
  //go through every row
  for (int i=0; i<n; ++i){
    //go through each column in this row
      result[i] *= beta;  
    for (int j=ia[i]; j<ia[i+1]; j++){
      int col = ja[j];
      result[i] += (alpha*a[j]*x[col]);
    }
  }
}

void simple_lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const double *la,const double *diagonal, const double *x, double *result){
  //compute result = L^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go through each row (starting from 0)
  for (int i=0; i<n; ++i){
    result[i] = x[i];
    for (int j=lia[i]; j<lia[i+1]; ++j){
      int col = lja[j];
      
//printf("this is row %d column %d is nnz, value %f, subtracting %f*%f from x[%d]=%f\n", i, col, la[j], la[j], result[col], i, result[i]);
result[i] -= la[j]*result[col]; 
   }
#if 0
printf("diagonal entry %d: %f \n", i,  la[lia[i+1]-1]);
    result[i] /= la[lia[i+1]-1]; //divide by the diagonal entry
#endif 

//printf("and diving x[%d]=%f by %f \n", i, result[i], diagonal[i]);
result[i] /=diagonal[i];
 }
}


void simple_upper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const double *ua, const double *diagonal, const double *x, double *result){
  //compute result = U^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go through each row (starting from the last row)
  for (int i=n-1; i>=0; --i){
    result[i] = x[i];
    for (int j=uia[i]; j<uia[i+1]; ++j){
      int col = uja[j];
      result[i] -= ua[j]*result[col];
    }
//    result[i] /= ua[uia[i]]; //divide by the diagonal entry
result[i] /=diagonal[i];
  }
}

//not std blas but needed and embarassingly parallel 

//simple vec-vec computes an element-wise product (needed for scaling)
//
void simple_vec_vec(const int n, const double * x, const double * y, double *res){
  for (int i=0; i<n; ++i){
    res[i] = x[i]*y[i];
  }
}

//vector reciprocal computes 1./d 
//
void simple_vector_reciprocal(const int n, const double *v, double *res){

  for (int i=0; i<n; ++i){
    if  (v[i] != 0.0 )res[i] = 1.0f/v[i];
    else res[i] = 0.0f;
  }
}

//vector sqrt takes an sqrt from each vector entry 


void simple_vector_sqrt(const int n, const double *v, double *res){

  for (int i=0; i<n; ++i){
    if  (v[i] >= 0.0) res[i] = sqrt(v[i]);
    else res[i] = 0.0f;
  }
}

void simple_vec_copy(const int n, const double *src, double *dest){

  for (int i=0; i<n; ++i){
    dest[i] = src[i];  
  }
}


void simple_vec_zero(const int n, double *vec){

  for (int i=0; i<n; ++i){
    vec[i] = 0.0f;  
  }
}
//add vec_copy
