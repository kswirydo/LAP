#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "common.h"
#include "blas.h"
#include "io_utils.h"

#if CUDA
#include "cuda_blas.h"
#include "devMem.h"
#endif
#if OPENMP
#include "openmp_blas.h"
#endif

#if HIP
#include "hip_blas.h"
#include "devMem.h"
#endif

int main(int argc, char *argv[]) { 
  /* read matrix and optional rhs */

  double time_spmv = 0.0;
  struct timeval t1, t2;

  const char *matrixFileName = argv[1];
  int n_trials = atoi(argv[2]);

  mmatrix *A; 
  A = (mmatrix *)calloc(1, sizeof(mmatrix));

  read_mm_file(matrixFileName, A);
  coo_to_csr(A); 

  printf("\n\n");
  printf("Matrix info: \n");
  printf("\n");
  printf("\t Matrix size       : %d x %d \n", A->n, A->n);
  printf("\t Matrix nnz        : %d  \n", A->nnz);
  printf("\t Matrix nnz un     : %d  \n", A->nnz_unpacked);
  printf("\t Number of trials  : %d  \n", n_trials);
#if 1  
  double *x = (double *) calloc (A->n, sizeof(double));
  /* y is RESULT */
  double *y = (double *) calloc (A->n, sizeof(double));
  for (int i = 0; i < A->n; ++i) {
    x[i] = 1.0;   
    y[i] = 0.0;   
  }  
  double one = 1.0;
  double zero = 0.0;
#if (CUDA || HIP)
  initialize_handles();
  double *d_x, *d_y;
  d_x = (double*) mallocForDevice (d_x, A->n, sizeof(double));
  d_y = (double*) mallocForDevice (d_y, A->n, sizeof(double));
  memcpyDevice(d_x, x, A->n, sizeof(double), "H2D");
  memcpyDevice(d_y, y, A->n, sizeof(double), "H2D");

  free(x);
  free(y);
  x = d_x;
  y = d_y;
  int *d_A_ia;
  int *d_A_ja;
  double * d_A_a;

  d_A_ia = (int *)  mallocForDevice ((d_A_ia), (A->n + 1), sizeof(int));
  d_A_ja = (int *)  mallocForDevice ((d_A_ja), (A->nnz_unpacked), sizeof(int));
  d_A_a = (double *)  mallocForDevice ((d_A_a), (A->nnz_unpacked), sizeof(double));
  memcpyDevice(d_A_ia, A->csr_ia, sizeof(int), (A->n + 1), "H2D");
  memcpyDevice(d_A_ja, A->csr_ja , sizeof(int) , (A->nnz_unpacked), "H2D");
  memcpyDevice(d_A_a, A->csr_vals , sizeof(double) , (A->nnz_unpacked), "H2D");

  free(A->csr_ia);
  free(A->csr_ja);
  free(A->csr_vals);
  A->csr_ia = d_A_ia;
  A->csr_ja = d_A_ja;
  A->csr_vals = d_A_a;
#if CUDA 
//  printf("initializin spmv buffer \n"); 
  initialize_spmv_buffer(A->n, 
                         A->nnz_unpacked,
                         A->csr_ia,
                         A->csr_ja,
                         A->csr_vals,
                         x,
                         y, 
                         &one, 
                         &zero);
#else // HIP
  analyze_spmv(A->n, 
               A->nnz_unpacked, 
               A->csr_ia,
               A->csr_ja,
               A->csr_vals,
               x,
               y, 
               "A");
#endif
#endif

  gettimeofday(&t1, 0);
  for (int i = 0; i < n_trials; ++i) {
    csr_matvec(A->n, 
               A->nnz_unpacked,   
               A->csr_ia, 
               A->csr_ja, 
               A->csr_vals, 
               x, 
               y, 
               &one, 
               &zero, 
               "A");
  }
  gettimeofday(&t2, 0);
  time_spmv = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

  double time_spmv_seconds = time_spmv / 1000.0;
  printf("\n\n");
  printf("SpMV test summary results: \n");
  printf("\n");
  printf("\t Time (total)        : %2.16f  \n", time_spmv_seconds);
  printf("\t Time (av., per one) : %2.16f  \n", time_spmv_seconds / (double) n_trials);
  printf("\t Norm of A*x         : %16.16e  \n", sqrt(dot(A->n, y, y)));
  printf("\n\n");
#endif  
  return 0;
}
