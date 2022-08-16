#include <rocsparse.h>
#include <rocblas.h>
#include <hip/hip_runtime_api.h>
#include "hip_blas.h"


static rocblas_handle handle_rocblas;
static  rocsparse_handle  handle_rocsparse;
static void * mv_buffer = NULL;
static void * L_buffer;
static void * U_buffer;

static rocsparse_mat_descr matA=NULL;
static rocsparse_mat_descr descrL, descrU, descrA;
static rocsparse_mat_info  infoL, infoU;
static rocsparse_mat_info  infoA;


void initialize_handles(){

  rocblas_create_handle(&handle_rocblas);
  rocsparse_create_handle(&handle_rocsparse);
  
  rocsparse_create_mat_descr(&(descrL));
  rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower);
  rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero);

  rocsparse_create_mat_descr(&(descrU));
  rocsparse_set_mat_index_base(descrU, rocsparse_index_base_zero);
  rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper);
  
 rocsparse_create_mat_descr(&(descrA));
rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero);
rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general);

rocsparse_create_mat_info(&infoA);
rocsparse_create_mat_info(&infoL);
rocsparse_create_mat_info(&infoU);
}
void analyze_spmv(const int n, 
                  const int nnz, 
                  int *ia, 
                  int *ja, 
                  double *a, 
                  const double *x, 
                  double *result,
                 char * option
                  ){
//no buffer in matvec
  rocsparse_status status_rocsparse;
if (strcmp(option, "A") == 0)
  status_rocsparse = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                               rocsparse_operation_none,
                                               n,
                                               n,
                                               nnz,
                                               descrA,
                                               a,
                                               ia,
                                               ja,
                                               infoA);

if (strcmp(option, "L") == 0)
  status_rocsparse = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                               rocsparse_operation_none,
                                               n,
                                               n,
                                               nnz,
                                               descrL,
                                               a,
                                               ia,
                                               ja,
                                               infoL);

if (strcmp(option, "U") == 0)
  status_rocsparse = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                               rocsparse_operation_none,
                                               n,
                                               n,
                                               nnz,
                                               descrU,
                                               a,
                                               ia,
                                               ja,
                                               infoU);

  if (status_rocsparse!=0)printf("mv analysis status for %s is %d \n", option, status_rocsparse);

}

void initialize_and_analyze_L_and_U_solve(const int n, 
                                          const int nnzL, 
                                          int *lia, 
                                          int *lja, 
                                          double *la,
                                          const int nnzU, 
                                          int *uia, 
                                          int *uja, 
                                          double *ua){

  size_t L_buffer_size;  
  size_t U_buffer_size;  
  rocsparse_status status_rocsparse;
  status_rocsparse = rocsparse_dcsrsv_buffer_size(handle_rocsparse, 
                               rocsparse_operation_none, 
                               n, 
                               nnzL, 
                               descrL,
                               la, 
                               lia, 
                               lja,
                               infoL, 
                               &L_buffer_size);
//printf("buffer size for L %d status %d \n", L_buffer_size, status_rocsparse);
  hipMalloc((void**)&(L_buffer), L_buffer_size);

 status_rocsparse = rocsparse_dcsrsv_buffer_size(handle_rocsparse, 
                               rocsparse_operation_none, 
                               n, 
                               nnzU, 
                               descrU,
                               ua, 
                               uia, 
                               uja,
                               infoU, 
                               &U_buffer_size);
  hipMalloc((void**)&(U_buffer), U_buffer_size);
//printf("buffer size for U %d status %d \n", U_buffer_size, status_rocsparse);
  status_rocsparse = rocsparse_dcsrsv_analysis(handle_rocsparse, 
                                               rocsparse_operation_none,
                                               n,
                                               nnzL,
                                               descrL,
                                               la,
                                               lia,
                                               lja,
                                               infoL,
                                               rocsparse_analysis_policy_reuse,
                                               rocsparse_solve_policy_auto,
                                               L_buffer);
  if (status_rocsparse!=0)printf("status after analysis 1 %d \n", status_rocsparse);
  status_rocsparse = rocsparse_dcsrsv_analysis(handle_rocsparse, 
                                               rocsparse_operation_none, 
                                               n,
                                               nnzU,
                                               descrU,
                                               ua,
                                               uia,
                                               uja,
                                               infoU,
                                               rocsparse_analysis_policy_reuse,
                                               rocsparse_solve_policy_auto,
                                               U_buffer);
  if (status_rocsparse!=0)printf("status after analysis 2 %d \n", status_rocsparse);
}




__global__ void hip_vec_vec_kernel(const int n,
                                   const double *x,
                                   const double *y,
                                   double *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    z[idx] =  x[idx]*y[idx];

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_reciprocal_kernel(const int n,
                                          const double *x,
                                          double *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    if  (x[idx] != 0.0 ){z[idx] = 1.0f/x[idx];}
    else z[idx] = 0.0f;

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_sqrt_kernel(const int n,
                                    const double *x,
                                    double *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    if (x[idx]>0)
      z[idx] =  sqrt(x[idx]);
    else z[idx] =0.0;

    idx += blockDim.x * gridDim.x;
  }
}


__global__ void hip_vec_zero_kernel(const int n,
                                    double *x){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    x[idx] =  0.0;

    idx += blockDim.x * gridDim.x;
  }
}

double hip_dot (const int n, const double *v, const double *w){
  double sum;

  rocblas_ddot (handle_rocblas, 
                n, 
                v, 
                1, 
                w, 
                1, 
                &sum);
  return sum;
}

void hip_scal (const int n, const double alpha, double *v){
  rocblas_dscal(handle_rocblas, 
                n,
                &alpha,
                v, 
                1);

}

void hip_axpy (const int n, const double alpha, const double *x, double *y){
  rocblas_daxpy(handle_rocblas, 
                n,
                &alpha,
                x, 
                1,
                y, 
                1);

}

void hip_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const double*al, const double *bet, const char * kind){
  // y = alpha *A* x + beta *y 
rocsparse_status st;
if (strcmp(kind, "A") == 0)
  st= rocsparse_dcsrmv(handle_rocsparse,
                   rocsparse_operation_none,
                   n,
                   n,
                   nnz,
                   al,
                   descrA,
                   a,
                   ia,
                   ja,
                   infoA,
                   x,
                   bet,
                   result);
if (strcmp(kind, "L") == 0)
  st= rocsparse_dcsrmv(handle_rocsparse,
                   rocsparse_operation_none,
                   n,
                   n,
                   nnz,
                   al,
                   descrL,
                   a,
                   ia,
                   ja,
                   infoL,
                   x,
                   bet,
                   result);
if (strcmp(kind, "U") == 0)
  st= rocsparse_dcsrmv(handle_rocsparse,
                   rocsparse_operation_none,
                   n,
                   n,
                   nnz,
                   al,
                   descrU,
                   a,
                   ia,
                   ja,
                   infoU,
                   x,
                   bet,
                   result);
//printf("status after mv: %d\n", st);
}

void hip_lower_triangular_solve(const int n, const int nnzL, const int *lia, const int *lja, const double *la,const double *diagonal, const double *x, double *result){
  //compute result = L^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go thr
  //d_x3 = L^(-1)dx2
  double one = 1.0;
  rocsparse_dcsrsv_solve(handle_rocsparse, 
                         rocsparse_operation_none,
                         n,
                         nnzL, 
                         &one, 
                         descrL,
                         la,
                         lia,
                         lja,
                         infoL,
                         x,
                         result,
                         rocsparse_solve_policy_auto,
                         L_buffer);
}


void hip_upper_triangular_solve(const int n, const int nnzU, const int *uia, const int *uja, const double *ua, const double *diagonal, const double *x, double *result){
  //compute result = U^{-1}x 
  double one = 1.0;
  rocsparse_dcsrsv_solve(handle_rocsparse, 
                         rocsparse_operation_none,
                         n, 
                         nnzU, 
                         &one, 
                         descrU,
                         ua,
                         uia,
                         uja,
                         infoU,
                         x,
                         result,
                         rocsparse_solve_policy_auto,
                         U_buffer);
}

//not std blas but needed and embarassingly parallel 

//cuda vec-vec computes an element-wise product (needed for scaling)
//
void hip_vec_vec(const int n, const double * x, const double * y, double *res){

  hipLaunchKernelGGL(hip_vec_vec_kernel, dim3(n/1024+1), dim3(1024),0,0,n, x, y, res);
}

//vector reciprocal computes 1./d 
//

void hip_vector_reciprocal(const int n, const double *v, double *res){

  hipLaunchKernelGGL( hip_vec_reciprocal_kernel,dim3(n/1024+1), dim3(1024),0,0,n, v, res);
}

//vector sqrt takes an sqrt from each vector entry 


void hip_vector_sqrt(const int n, const double *v, double *res){

  hipLaunchKernelGGL(hip_vec_sqrt_kernel, dim3(n), dim3(1024), 0,0,n, v, res);
}

void hip_vec_copy(const int n, const double *src, double *dest){

  hipMemcpy(dest, src, sizeof(double) * n, hipMemcpyDeviceToDevice);
}


void hip_vec_zero(const int n, double *vec){

  hipLaunchKernelGGL(hip_vec_zero_kernel,dim3(n), dim3(1024), 0, 0,n, vec);
}

