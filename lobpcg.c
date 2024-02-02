// based on nvgraph

#include "common.h"
#include "blas.h"
#include "devMem.h"

void lobpcg(int n, 
    double nnz,
		int *ia, //matrix csr data
		int *ja,
		double *a,
		double tol, //DONT MULTIPLY BY NORM OF B
		pdata * prec_data, //preconditioner data: all Ls, Us etc
    int k, // number of eigenvalues wanted
		int maxit,
		int *it, //output: iteration
    double * eig_vecs,
    double * eig_vals		
){

#if (CUDA || HIP)
//allocate data needed for the GPU
  double * AX;
  double * BX;
  double * X;
	AX = (double*) mallocForDevice(AX, n*k, sizeof(double));
	BX = (double*) mallocForDevice(BX, n*k, sizeof(double));
	X = (double*) mallocForDevice(X, n*k, sizeof(double));

#endif
//initialize X

randomInit(X, n*k);

// STEP 1: AX = A*X;
// use SpGemm in cuda
// STEP 2

//
}
