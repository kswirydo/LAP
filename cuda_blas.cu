#include "cublas_v2.h"

#include <cusparse.h> 
#include "cuda_blas.h"
static cublasHandle_t handle_cublas;
static cusparseHandle_t handle_cusparse;

static void *mv_buffer;
static void *L_buffer;
static void *U_buffer;
static void *ichol_buffer; // in ichol, we can get away with one buffer

static cusparseSpMatDescr_t matA = NULL;
static cusparseSpMatDescr_t matL;
static cusparseSpMatDescr_t matU;
static cusparseMatDescr_t descrL, descrU, descrLt, descrM; // last two are used only for incomplete CHolesky
static  csrsv2Info_t infoL, infoU, infoLt;
csric02Info_t infoM  = 0; // used only for Incomplete Cholesky

#define policy CUSPARSE_SOLVE_POLICY_USE_LEVEL 


void initialize_handles(){
  //printf("initializing handles! \n");
  cublasCreate(&handle_cublas);
  cusparseCreate(&handle_cusparse);
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
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  size_t mv_buffer_size;
  cusparseStatus_t status_cusparse;

  status_cusparse = cusparseCreateDnVec(&vecX,
                                        n,
                                        (void*) x,
                                        CUDA_R_64F);
  
  // printf("matX creation status %d\n", status_cusparse);  
  status_cusparse = cusparseCreateDnVec(&vecY,
                                        n,
                                        (void *) result,
                                        CUDA_R_64F);
  
  // printf("vecY creation status %d\n", status_cusparse);  
  status_cusparse = cusparseCreateCsr(&matA,
                                      n,
                                      n,
                                      nnz,
                                      ia,
                                      ja,
                                      a,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F);
  
  // printf("matA creation status %d\n", status_cusparse);  
  status_cusparse = cusparseSpMV_bufferSize(handle_cusparse,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            al,
                                            matA,
                                            vecX,
                                            bet,
                                            vecY,
                                            CUDA_R_64F,
                                            CUSPARSE_SPMV_CSR_ALG2,
                                            &mv_buffer_size);

  cudaDeviceSynchronize();
 
  // printf("mv buffer size %d alpha %f beta %f status %d \n", mv_buffer_size, *al, *bet, status_cusparse);
  cudaError t = cudaMalloc( &mv_buffer, mv_buffer_size);
 
   if (t != 0) printf("allocated mv_buffer: is it NULL? %d, error %d \n", mv_buffer == NULL, t);
  
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
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

  cusparseCreateMatDescr(&(descrL));
  cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);

  cusparseCreateMatDescr(&(descrU));
  cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
  cusparseCreateCsrsv2Info(&infoL);
  cusparseCreateCsrsv2Info(&infoU);
  int L_buffer_size;  
  int U_buffer_size;  
 
  cusparseDcsrsv2_bufferSize(handle_cusparse, 
                             CUSPARSE_OPERATION_NON_TRANSPOSE, 
                             n, 
                             nnzL, 
                             descrL,
                             la, 
                             lia, 
                             lja,
                             infoL, 
                             &L_buffer_size);
  //printf("buffer size L %d\n", L_buffer_size);
  cudaMalloc((void**)&(L_buffer), L_buffer_size);

  cusparseDcsrsv2_bufferSize(handle_cusparse, 
                             CUSPARSE_OPERATION_NON_TRANSPOSE, 
                             n, 
                             nnzU, 
                             descrU,
                             ua, 
                             uia, 
                             uja,
                             infoU, 
                             &U_buffer_size);
  //printf("buffer size U %d\n", U_buffer_size);
  cudaMalloc((void**)&(U_buffer), U_buffer_size);
  cusparseStatus_t status_cusparse;
  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             n,
                                             nnzL,
                                             descrL,
                                             la,
                                             lia,
                                             lja,
                                             infoL,
                                             policy, 
                                             L_buffer);

  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
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


  cusparseCreateCsr(&matL,
                    n,
                    n,
                    nnzL,
                    lia,
                    lja,
                    la,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
 
  cusparseCreateCsr(&matU,
                    n,
                    n,
                    nnzU,
                    uia,
                    uja,
                    ua,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);

}

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      double *a)
{

  printf("initializing ICHOL \n");
  cusparseCreateMatDescr(&descrM);
  cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);

  cusparseCreateMatDescr(&descrL);
  cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);

  cusparseCreateCsric02Info(&infoM);
  cusparseCreateCsrsv2Info(&infoL);
  cusparseCreateCsrsv2Info(&infoLt);
  int structural_zero;
  int numerical_zero; 

  cusparseStatus_t status_cusparse;

  /* figure out the buffer size */

  int bufferSize, bufferSizeL, bufferSizeLt, bufferSizeM;
  status_cusparse =  cusparseDcsric02_bufferSize(handle_cusparse, 
                                                 n, 
                                                 nnzA,
                                                 descrM, 
                                                 a,
                                                 ia, 
                                                 ja, 
                                                 infoM, 
                                                 &bufferSizeM);
  
  status_cusparse =  cusparseDcsrsv2_bufferSize(handle_cusparse, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n, 
                                                nnzA,
                                                descrL, 
                                                a, 
                                                ia, 
                                                ja, 
                                                infoL, 
                                                &bufferSizeL);

  status_cusparse =  cusparseDcsrsv2_bufferSize(handle_cusparse, 
                                                CUSPARSE_OPERATION_TRANSPOSE, 
                                                n, 
                                                nnzA,
                                                descrL, 
                                                a, 
                                                ia, 
                                                ja, 
                                                infoLt, 
                                                &bufferSizeLt);


  bufferSize = max(bufferSizeM, max(bufferSizeL, bufferSizeLt));

  cudaMalloc((void**) &ichol_buffer, bufferSize);

  /* and now analyze */

  status_cusparse = cusparseDcsric02_analysis(handle_cusparse,
                                              n, 
                                              nnzA, 
                                              descrM,
                                              a, 
                                              ia, 
                                              ja, 
                                              infoM,
                                              policy, 
                                              ichol_buffer);
  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, infoM, &structural_zero);
  
  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have a problem: A(%d,%d) is missing\n", structural_zero, structural_zero);
  }

  /* analyze the solves as well */

  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                             n, 
                                             nnzA, 
                                             descrL,
                                             a, 
                                             ia, 
                                             ja,
                                             infoL, 
                                             policy, 
                                             ichol_buffer);

  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
                                             CUSPARSE_OPERATION_TRANSPOSE, 
                                             n, 
                                             nnzA, 
                                             descrL,
                                             a, 
                                             ia, 
                                             ja,
                                             infoLt, 
                                             policy, 
                                             ichol_buffer);

  /* decompose */
  status_cusparse = cusparseDcsric02(handle_cusparse, 
                                     n, 
                                     nnzA, 
                                     descrM,
                                     a, 
                                     ia, 
                                     ja, 
                                     infoM, 
                                     policy, 
                                     ichol_buffer);
  
  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, 
                                               infoM, 
                                               &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have another problem: L(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }
}


void cuda_ichol(const int *ia, 
                const int *ja, 
                double *a, 
                const int nnzA,
                pdata *prec_data, 
                double *x, 
                double *y) {
  double one = 1.0;
  cusparseDcsrsv2_solve(handle_cusparse, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        prec_data->n, 
                        nnzA, 
                        &one, 
                        descrL, // replace with cusparseSpSV
                        prec_data->ichol_vals, 
                        ia, 
                        ja, 
                        infoL,
                        x,//input 
                        prec_data->aux_vec1, //output
                        policy, 
                        ichol_buffer);

  /* solve L'*y = aux_vec1 */
  cusparseDcsrsv2_solve(handle_cusparse, 
                        CUSPARSE_OPERATION_TRANSPOSE, 
                        prec_data->n, 
                        nnzA, &one, 
                        descrL, // replace with cusparseSpSV
                        prec_data->ichol_vals, 
                        ia, 
                        ja, 
                        infoLt,
                        prec_data->aux_vec1, 
                        y, 
                        policy, 
                        ichol_buffer);
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
    if  (x[idx] != 0.0 ){
      z[idx] = 1.0/x[idx];
    } else {
      z[idx] = 0.0;
    }

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_vec_sqrt_kernel(const int n,
                                     const double *x,
                                     double *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    if (x[idx]>0) {
      z[idx] =  sqrt(x[idx]);
    } else {
      z[idx] = 0.0;
    }

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

  cublasStatus_t status;
  status = cublasDdot (handle_cublas, 
                       n, 
                       v, 
                       1, 
                       w, 
                       1, 
                       &sum);
  //printf("DOT product status %d\n", status);
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

  cublasStatus_t status;
  status = cublasDaxpy(handle_cublas, 
                       n,
                       &alpha,
                       x, 
                       1,
                       y, 
                       1);
}

void cuda_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const double *a, const double *x, double *result, const double*al, const double *bet){
  /* y = alpha *A* x + beta * y */ 

  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  cusparseSpMatDescr_t matCSR;
  cusparseCreateDnVec(&vecX,
                      n,
                      (void*) x,
                      CUDA_R_64F);

  cusparseCreateDnVec(&vecY,
                      n,
                      (void *) result,
                      CUDA_R_64F);

  cusparseStatus_t status_cusparse;

  status_cusparse = cusparseCreateCsr(&matCSR,
                                      n,
                                      n,
                                      nnz,
                                      (void *)ia,
                                      (void *)ja,
                                      (void *)a,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F);
  // printf("before matvec: input^Tinput %5.16e, output^Toutput %5.16e alpha %f beta %f\n", cuda_dot(n, x,x), cuda_dot(n, result, result), *al, *bet);
  status_cusparse = cusparseSpMV(handle_cusparse,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 al,
                                 matCSR,
                                 vecX,
                                 bet,
                                 vecY,
                                 CUDA_R_64F,
                                 CUSPARSE_SPMV_CSR_ALG2,
                                 mv_buffer);
 //  printf("matvec status: %d is MV BUFFER NULL? %d  is matA null? %d\n", status_cusparse, mv_buffer == NULL, matA==NULL);
 //  printf("after matvec: input^Tinput %5.16e, output^Toutput %5.16e\n", cuda_dot(n, x,x), cuda_dot(n,result, result));
  cusparseDestroySpMat(matCSR);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
}

void cuda_lower_triangular_solve(const int n,
                                 const int nnzL, 
                                 const int *lia, 
                                 const int *lja, 
                                 const double *la,
                                 const double *diagonal, 
                                 const double *x, double *result){
  /* compute result = L^{-1}x */
  /* we DO NOT assume anything about L diagonal */
  /* d_x3 = L^(-1)dx2 */

  double one = 1.0;
  
  cusparseStatus_t status = cusparseDcsrsv2_solve(handle_cusparse, 
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE, 
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
  //printf("status after tri solve is %d \n", status);
}


void cuda_upper_triangular_solve(const int n, 
                                 const int nnzU, 
                                 const int *uia, 
                                 const int *uja, 
                                 const double *ua, 
                                 const double *diagonal, 
                                 const double *x, 
                                 double *result){

  /* compute result = U^{-1}x */
  double one = 1.0;
  cusparseDcsrsv2_solve(handle_cusparse, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
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

/* not std blas but needed and embarassingly parallel */

/* cuda vec-vec computes an element-wise product (needed for scaling) */

void cuda_vec_vec(const int n, const double *x, const double *y, double *res){

  cuda_vec_vec_kernel<<<1024, 1024>>>(n, x, y, res);
}

/* vector reciprocal computes 1./d */ 

void cuda_vector_reciprocal(const int n, const double *v, double *res){

  cuda_vec_reciprocal_kernel<<<1024, 1024>>>(n, v, res);
}

/* vector sqrt takes an sqrt from each vector entry */

void cuda_vector_sqrt(const int n, const double *v, double *res){

  cuda_vec_sqrt_kernel<<<1024, 1024>>>(n, v, res);
}

void cuda_vec_copy(const int n, const double *src, double *dest){

  cudaMemcpy(dest, src, sizeof(double) * n, cudaMemcpyDeviceToDevice);
}

void cuda_vec_zero(const int n, double *vec){

  cuda_vec_zero_kernel<<<1024, 1024>>>(n, vec);
}

