#include "openmp_blas.h"
#include <math.h>

void openmp_scal (const int n, const double alpha, double *v)
{ 
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(alpha) map(tofrom:v[0:n])
  for (i = 0; i < n; ++i) {
    v[i] *= alpha;
  }
}

void openmp_axpy (const int n, const double alpha, const double *x, double *y){
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:x[0:n]) map(tofrom:y[0:n])
  for (i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

void openmp_csr_matvec(const int n, 
                       const int nnz, 
                       const int *ia, 
                       const int *ja, 
                       const double *a, 
                       const double *x, 
                       double *result,
                       const  double *al,
                       const double *bet) 
{

  double alpha = *al;
  double beta = *bet;
  /* go through every row */
  int i, j, col;
  double s;
#pragma omp target teams distribute parallel for  schedule(static) private(i, s)map(to:a[0:nnz], x[0:n], ia[0:n+1], ja[0:nnz], alpha, beta) map(tofrom:result[0:n])
  for (i = 0; i < n; ++i) {
    /* go through each column in this row */
    s = result[i] * beta;  
#pragma omp simd reduction(+:s)
    for (j = ia[i]; j < ia[i + 1]; j++) {
      col = ja[j];
      s += (alpha * a[j] * x[col]);
    }
    result[i] = s;
  }
}

void openmp_lower_triangular_solve(const int n, 
                                   const int nnz, 
                                   const int *lia, 
                                   const int *lja, 
                                   const double *la,
                                   const double *diagonal, 
                                   const double *x, 
                                   double *result) 
{
  /* compute result = L^{-1}x */ 
  int i, j, col;
  //#pragma omp target teams distribute map(to:lia[0:n+1],lja[0:nnz],la[0:nnz],x[0:n], diagonal[0:n])  map(tofrom:result[0:n])
  for (i = 0; i < n; ++i) {
    double s = 0.0;
    #pragma omp simd private(j, col) reduction(+:s)
    for (j = lia[i]; j < lia[i + 1]; ++j) {
      col = lja[j];
      s += (-1.0) * la[j] * result[col]; 
    }

    result[i] = (s + x[i]) / diagonal[i];
  }
}

void openmp_upper_triangular_solve(const int n, 
                                   const int nnz, 
                                   const int *uia, 
                                   const int *uja, 
                                   const double *ua, 
                                   const double *diagonal, 
                                   const double *x, 
                                   double *result)
{
  /* compute result = U^{-1}x */ 
  /* go through each row (starting from the last row) */
  int i, j, col;
  double s; 
  //this kind of works but the result is non deterministic 
//#pragma omp target teams distribute map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  // #pragma omp target map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  for (i = n - 1; i >= 0; --i) {
    s = 0.0;
    result[i] = 0.0;
#pragma omp simd private( col) reduction(+:s)
    //map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz],result[0:n]) map(from:s)
    for (j = uia[i]; j < uia[i + 1]; ++j){
      col = uja[j];
      s += (-1.0) * ua[j] * result[col];
    }
    //#pragma omp ordered
    result[i] = (s + x[i]) / diagonal[i]; 
  }
}

/* not std blas but needed and embarassingly parallel */

/* simple vec-vec computes an element-wise product (needed for scaling) */
void openmp_vec_vec(const int n, const double *x, const double *y, double *res)
{
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:x[0:n], y[0:n]) map(from:res[0:n])
  for (i = 0; i < n; ++i) {
    res[i] = x[i] * y[i];
  }
}

/* vector reciprocal computes 1./d */
void openmp_vector_reciprocal(const int n, const double *v, double *res)
{
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:v[0:n]) map(from:res[0:n])
  for (i = 0; i < n; ++i) {
    if (v[i] != 0.0) {
      res[i] = 1.0 / v[i];
    } else { 
      res[i] = 0.0;
    }
  }
}

/* vector sqrt takes an sqrt from each vector entry */
void openmp_vector_sqrt(const int n, const double *v, double *res)
{
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:v[0:n]) map(from:res[0:n])
  for (i = 0; i < n; ++i) {
    if  (v[i] >= 0.0) {
      res[i] = sqrt(v[i]);
    } else {
      res[i] = 0.0;
    }
  }
}

void openmp_vec_copy(const int n, const double *src, double *dest)
{
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:src[0:n]) map(from:dest[0:n])
  for (i = 0; i < n; ++i) {
    dest[i] = src[i];  
  }
}

void openmp_vec_zero(const int n, double *vec)
{
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i)  map(tofrom:vec[0:n])
  for (i = 0; i < n; ++i) {
    vec[i] = 0.0;  
  }
}

double openmp_dot(const int n, const double *v, const double *w)
{
  double sum = 0.0;
  int i;
#pragma omp target teams distribute parallel for  schedule(static) private(i) map(to:v[0:n], w[0:n]) reduction(+:sum)
  for (i = 0; i < n; ++i) {
    sum += v[i] * w[i];
  }
  return sum;
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
  for (int i = 0; i < n; ++i) {
    /*   
     *   if (n>100000) {
     *      if (i %100==0) printf("processing row: %d\n", i);
     }*/

    a[ia[i]] = sqrt(a[ia[i]]);
    for (int m = ia[i] + 1; m < ia[i + 1]; ++m){
      a[m] = a[m]/a[ia[i]]; 
    }

    for (int m = ia[i] + 1; m < ia[i + 1]; ++m) {
      for (int k = ia[ja[m]]; k < ia[ja[m] + 1]; ++k) {
        for (int l = m; l < ia[i + 1]; ++l) {
          if (ja[l] == ja[k]){ 
            a[k] -= a[m] * a[l];
          } /* if */
        } /* loop with l */
      } /* loop with k */ 
    } /* loop with m */
  }
  /* at this point, what we have in (ia, ja, a) is CSR format of L^T (so the same as "U").
   * and we need L (also in CSR), so we have to transpose. */

  int *Lcounts = (int *) calloc (nnzA, sizeof(int));
  for (int i = 0; i < n; ++i) {
    for (int j = ia[i]; j < ia[i + 1]; ++j) {
      int row = ja[j];
      double val = a[j];
      la[lia[row] + Lcounts[row]] = val;
      Lcounts[row]++;
    } 
  }
  free(Lcounts); 
}

void openmp_ichol(const int *ia, 
                  const int *ja, 
                  double *a, 
                  const int nnzA, 
                  pdata *prec_data, 
                  double *x, 
                  double *y)
{
  /* we dont really need A but whatever */
  double *la = prec_data->la;
  int *lia = prec_data->lia;
  int *lja = prec_data->lja;
  double *ua = prec_data->ua;
  int *uia = prec_data->uia;
  int *uja = prec_data->uja;
  int n = prec_data->n;  

  /* compute result = L^{-1}x */ 
  for (int i = 0; i < n; ++i) {
    prec_data->aux_vec1[i] = x[i];
    for (int j = lia[i]; j < lia[i + 1]; ++j) {
      int col = lja[j];
      if (col != i){
        prec_data->aux_vec1[i] -= la[j] * prec_data->aux_vec1[col]; 
      }
    }
    prec_data->aux_vec1[i] /= la[lia[i + 1] - 1]; ;
  }

  for (int i = n - 1; i >= 0; --i) {
    y[i] = prec_data->aux_vec1[i];
    for (int j = uia[i]; j < uia[i + 1]; ++j) {
      int col = uja[j];
      if (col != i){
        y[i] -= ua[j] * y[col];
      }
    }
    y[i] /= ua[uia[i]]; /*divide by the diagonal entry*/
  }
}
