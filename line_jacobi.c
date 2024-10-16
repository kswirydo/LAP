#include "common.h"
#include "blas.h"
void line_jacobi(int *ia, int *ja, real_type *a,int nnzA,  pdata* prec_data, real_type *vec_in, real_type *vec_out){
  int n = prec_data->n;
  vec_vec(n, prec_data->d_r, vec_in, vec_out);

}
