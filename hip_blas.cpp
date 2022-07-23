#include <rocsparse.h>
#include <rocblas.h>
#include <hip/hip_runtime_api.h>



static cublasHandle_t handle_cublas;
static hipsparseHandle_t handle_hipsparse;
static void * mv_buffer = NULL;
static void * L_buffer;
static void * U_buffer;

static hipsparseSpMatDescr_t matA=NULL;
static hipsparseSpMatDescr_t matL;
static hipsparseSpMatDescr_t matU;
static hipsparseMatDescr_t descrL, descrU;
static  csrsv2Info_t infoL, infoU;
#define policy HIPSPARSE_SOLVE_POLICY_USE_LEVEL 


void initialize_handles(){
  cublasCreate(&handle_cublas);
  hipsparseCreate(&handle_hipsparse);
}
void initialize_spmv_buffer(const int n, 
                            const int nnz, 
                            int *ia, 
                            int *ja, 
                            double *a, 
                            const double *x, 
                            double *result, 
                            double *al, 
                            double *bet){
  hipsparseDnVecDescr_t vecX;
  hipsparseDnVecDescr_t vecY;
  size_t mv_buffer_size;

  hipsparseCreateDnVec(&vecX,
                      n,
                      (void*)x,
                      CUDA_R_64F);
  hipsparseCreateDnVec(&vecY,
                      n,
                      (void *) result,
                      CUDA_R_64F);
  hipsparseStatus_t status_hipsparse;
  status_hipsparse = hipsparseCreateCsr(&matA,
                    n,
                    n,
                    nnz,
                    ia,
                    ja,
                    a,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
printf("matA creation status %d\n", status_hipsparse);  
status_hipsparse = hipsparseSpMV_bufferSize(handle_hipsparse,
                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          al,
                          matA,
                          vecX,
                          bet,
                          vecY,
                          CUDA_R_64F,
                          HIPSPARSE_MV_ALG_DEFAULT,
                          &mv_buffer_size);
printf("mv buffer size %d alpha %f beta %f status %d \n", mv_buffer_size, *al, *bet, status_hipsparse);
  cudaMalloc((void**) mv_buffer, mv_buffer_size);

  hipsparseDestroyDnVec(vecX);
  hipsparseDestroyDnVec(vecY);

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

  hipsparseCreateMatDescr(&(descrL));
  hipsparseSetMatIndexBase(descrL, HIPSPARSE_INDEX_BASE_ZERO);
  hipsparseSetMatFillMode(descrL, HIPSPARSE_FILL_MODE_LOWER);

  hipsparseCreateMatDescr(&(descrU));
  hipsparseSetMatIndexBase(descrU, HIPSPARSE_INDEX_BASE_ZERO);
  hipsparseSetMatFillMode(descrU, HIPSPARSE_FILL_MODE_UPPER);
  hipsparseCreateCsrsv2Info(&infoL);
  hipsparseCreateCsrsv2Info(&infoU);
  int L_buffer_size;  
  int U_buffer_size;  
  hipsparseDcsrsv2_bufferSize(handle_hipsparse, 
                             HIPSPARSE_OPERATION_NON_TRANSPOSE, 
                             n, 
                             nnzL, 
                             descrL,
                             la, 
                             lia, 
                             lja,
                             infoL, 
                             &L_buffer_size);
printf("buffer size L %d\n", L_buffer_size);
  cudaMalloc((void**)&(L_buffer), L_buffer_size);

  hipsparseDcsrsv2_bufferSize(handle_hipsparse, 
                             HIPSPARSE_OPERATION_NON_TRANSPOSE, 
                             n, 
                             nnzU, 
                             descrU,
                             ua, 
                             uia, 
                             uja,
                             infoU, 
                             &U_buffer_size);
printf("buffer size U %d\n", U_buffer_size);
  cudaMalloc((void**)&(U_buffer), U_buffer_size);
  hipsparseStatus_t status_hipsparse;
  status_hipsparse = hipsparseDcsrsv2_analysis(handle_hipsparse, 
                                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                             n,
                                             nnzL,
                                             descrL,
                                             la,
                                             lia,
                                             lja,
                                             infoL,
                                             policy, 
                                             L_buffer);

  status_hipsparse = hipsparseDcsrsv2_analysis(handle_hipsparse, 
                                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                             n,
                                             nnzU,
                                             descrU,
                                             ua,
                                             uia,
                                             uja,
                                             infoU,
                                             policy, 
                                             U_buffer);
}


void initialize_L_and_U_descriptors(const int n, 
                                  const int nnzL, 
                                  int *lia, 
                                  int *lja, 
                                  double *la,
                                  const int nnzU, 
                                  int *uia, 
                                  int *uja, 
                                  double *ua){


  hipsparseCreateCsr(&matL,
                    n,
                    n,
                    nnzL,
                    lia,
                    lja,
                    la,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
  hipsparseCreateCsr(&matU,
                    n,
                    n,
                    nnzU,
                    uia,
                    uja,
                    ua,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);

}

__global__ void cuda_vec_vec_kernel(const int n,
                                    const double *x,
                                    const double *y,
                                    double *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    z[idx] =  x[idx]*y[idx];
  
  idx += blockDim.x * gridDim.x;
}
}

__global__ void cuda_vec_reciprocal_kernel(const int n,
                                     const double *x,
                                     double *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    if  (x[idx] != 0.0 ){z[idx] = 1.0f/x[idx];}
    else z[idx] = 0.0f;
  
  idx += blockDim.x * gridDim.x;
}
}

__global__ void cuda_vec_sqrt_kernel(const int n,
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


__global__ void cuda_vec_zero_kernel(const int n,
                                     double *x){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    x[idx] =  0.0;
  
  idx += blockDim.x * gridDim.x;
}
}

double cuda_dot (const int n, const double *v, const double *w){
  double sum;
  
   cublasDdot (handle_cublas, 
              n, 
              v, 
              1, 
              w, 
              1, 
              &sum);
  return sum;
}

void cuda_scal (const int n, const double alpha, double *v){
  cublasDscal(handle_cublas, 
              n,
              &alpha,
              v, 
              1);

}

void cuda_axpy (const int n, const double alpha, const double *x, double *y){
 cublasDaxpy(handle_cublas, 
              n,
              &alpha,
              x, 
              1,
              y, 
              1);

}

void cuda_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const double*al, const double *bet){
  // y = alpha *A* x + beta *y 

  hipsparseDnVecDescr_t vecX;
  hipsparseDnVecDescr_t vecY;
  hipsparseSpMatDescr_t matCSR;
  hipsparseCreateDnVec(&vecX,
                      n,
                      (void*)x,
                      CUDA_R_64F);
  hipsparseCreateDnVec(&vecY,
                      n,
                      (void *) result,
                      CUDA_R_64F);
  hipsparseStatus_t status_hipsparse;
  status_hipsparse = hipsparseCreateCsr(&matCSR,
                    n,
                    n,
                    nnz,
                    (void *)ia,
                    (void *)ja,
                    (void *)a,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
  status_hipsparse=hipsparseSpMV(handle_hipsparse,
               HIPSPARSE_OPERATION_NON_TRANSPOSE,
               al,
               matCSR,
               vecX,
               bet,
               vecY,
               CUDA_R_64F,
               HIPSPARSE_MV_ALG_DEFAULT,
               mv_buffer);
//printf("matvec status: %d is MV BUFFER NULL? %d  is matA null? %d\n", status_hipsparse, mv_buffer == NULL, matA==NULL);
hipsparseDestroySpMat(matCSR);
  hipsparseDestroyDnVec(vecX);
  hipsparseDestroyDnVec(vecY);
}

void cuda_lower_triangular_solve(const int n, const int nnzL, const int *lia, const int *lja, const double *la,const double *diagonal, const double *x, double *result){
  //compute result = L^{-1}x 
  ////we DO NOT assume anything about L diagonal
  //go thr
  //d_x3 = L^(-1)dx2
double one = 1.0;
  hipsparseDcsrsv2_solve(handle_hipsparse, 
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE, 
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
                                          policy,
                                          L_buffer);
}


void cuda_upper_triangular_solve(const int n, const int nnzU, const int *uia, const int *uja, const double *ua, const double *diagonal, const double *x, double *result){
  //compute result = U^{-1}x 
double one = 1.0;
  hipsparseDcsrsv2_solve(handle_hipsparse, 
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE, 
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
                                          policy,
                                          U_buffer);
}

//not std blas but needed and embarassingly parallel 

//cuda vec-vec computes an element-wise product (needed for scaling)
//
void cuda_vec_vec(const int n, const double * x, const double * y, double *res){

    cuda_vec_vec_kernel<<<n/1024+1, 1024>>>(n, x, y, res);
}

//vector reciprocal computes 1./d 
//

void cuda_vector_reciprocal(const int n, const double *v, double *res){

    cuda_vec_reciprocal_kernel<<<n/1024+1, 1024>>>(n, v, res);
}

//vector sqrt takes an sqrt from each vector entry 


void cuda_vector_sqrt(const int n, const double *v, double *res){

    cuda_vec_sqrt_kernel<<<n, 1024>>>(n, v, res);
}

void cuda_vec_copy(const int n, const double *src, double *dest){

  cudaMemcpy(dest, src, sizeof(double) * n, cudaMemcpyDeviceToDevice);
}


void cuda_vec_zero(const int n, double *vec){

    cuda_vec_zero_kernel<<<n, 1024>>>(n, vec);
}

