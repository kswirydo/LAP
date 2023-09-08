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


void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      double *a, 
                      int *lia,
                      int *lja,
                      double *la)
{
  printf("ICHOL SETUP INPUT \n");  



  // we assume A = (ia, ja, a) is UPPER TRIANGULAR
  // aux variables
  double * Lcol = (double *) calloc (n, sizeof(double));
  int * Lcounts = (int *) calloc (n, sizeof(int));

  for (int k = 0; k < n; ++k) {
if (n>100000) {
if (k %100==0) printf("processing row: %d\n", k);
}
    simple_vec_zero(n, Lcol);
    a[ia[k]] = sqrt(a[ia[k]]);
    for (int i = ia[k]+1; i<ia[k+1]; ++i){
      a[i] = a[i]/a[ia[k]]; 
      Lcol[ja[i]] = a[i];
    }


    for (int j = k+1; j < n ; ++j) {     
      // go through each column
      for (int idx = ia[j]; idx<ia[j+1]; ++idx) {       
        int i = ja[idx];
        if (i >= j ) { // potential match
          a[idx] = a[idx] - Lcol[j]*Lcol[i];
        }
      }
    }
  }
  // now ia has the values, in upper triangular format
  //lia has correct format


  for (int i = 0; i < n; ++i){
    for (int j = ia[i]; j < ia[i+1]; ++j){
      int row = ja[j];
      double val = a[j];
      la[lia[row]+Lcounts[row]] = val;
      Lcounts[row]++;
    } 
  }
#if 0 
  for (int i = 0; i<n; ++i) {
    printf("This is row %d \n", i);
    for (int j = lia[i]; j<lia[i+1]; ++j)
    {
      printf("(%d,  %f) ", lja[j], la[j]);
    }
    printf("\n");
  } 
#endif
}

void simple_ichol(int *ia, int *ja, double *a, int nnzA, pdata* prec_data, double * x, double *y)
{
  // we dont really need A but whatever
  double *la = prec_data->la;
  int *lia = prec_data->lia;
  int *lja = prec_data->lja;
  double *ua = prec_data->ua;
  int *uia = prec_data->uia;
  int *uja = prec_data->uja;
  int n = prec_data->n;  
  //compute result = L^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go through each row (starting from 0)
//printf("norm before L solve: x: %16.16f \n", simple_dot(n, x, x));
  for (int i=0; i<n; ++i){
    prec_data->aux_vec1[i] = x[i];
    for (int j=lia[i]; j<lia[i+1]; ++j){
      int col = lja[j];
if (col!=i){
 //     printf("this is row %d column %d is nnz, value %f, subtracting %f*%f from x[%d]=%f\n", i, col, la[j], la[j], prec_data->aux_vec1[col], i, prec_data->aux_vec1[i]);
      prec_data->aux_vec1[i] -= la[j]*prec_data->aux_vec1[col]; 
}
    }
#if 0
    printf("diagonal entry %d: %f \n", i,  la[lia[i+1]-1]);
    result[i] /= la[lia[i+1]-1]; //divide by the diagonal entry
#endif 

  //  printf("and diving x[%d]=%f by %f, so x[%d] = %f \n", i,  prec_data->aux_vec1[i], la[lia[i+1]-1], i, prec_data->aux_vec1[i]/la[lia[i+1]-1]);
   // printf("diagonal entry %d: %f \n", i,  la[lia[i+1]-1]);
    prec_data->aux_vec1[i] /=la[lia[i+1]-1]; ;
  }
//printf("norm after L solve: %16.16f \n", simple_dot(n, prec_data->aux_vec1, prec_data->aux_vec1));
//for (int i=0; i<10; ++i) printf("vec1[%d] = %f \n", i, prec_data->aux_vec1[i]);
  for (int i=n-1; i>=0; --i){
    y[i] = prec_data->aux_vec1[i];
    for (int j=uia[i]; j<uia[i+1]; ++j){
      int col = uja[j];
if (col!=i){
      y[i] -= ua[j]*y[col];
}
    }
    y[i] /= ua[uia[i]]; //divide by the diagonal entry
  }

}
//add vec_copy
