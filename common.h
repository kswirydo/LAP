#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#pragma once

#define NOACC 0
#define CUDA 1
#define OPENMP 0
typedef struct{
 
  int *lia;
  int *lja;
  double *la;
  int lnnz; 

 
  int *uia;
  int *uja;
  double *ua;
  int unnz;

  double *d;
  double *d_r;//d_r = 1./d
  int n;

  double *aux_vec1, *aux_vec2, *aux_vec3;

  char * prec_op;
  int m, k;//m is outer loop, k inner
} pdata;

void prec_function(int *ia, int *ja, double *a, int nnzA,pdata* prec_data, double * x, double *y);


void cg(int n, double nnz,
        int *ia, //matrix csr data
        int *ja,
        double *a,
        double *x, //solution vector, mmust be alocated prior to calling
        double *b, //rhs
        double tol, //DONT MULTIPLY BY NORM OF B
        pdata * prec_data, //preconditioner data: all Ls, Us etc
        int maxit,
        int *it, //output: iteration
        int *flag, //output: flag 0-converged, 1-maxit reached, 2-catastrophic failure
        double * res_norm_history //output: residual norm history
       );


void GS_std(int *ia, int *ja, double *a, int nnzA,  pdata* prec_data, double *vec_in, double *vec_out);
void GS_it(int *ia, int *ja, double *a, int nnzA,  pdata* prec_data, double *vec_in, double *vec_out);
void GS_it2(int *ia, int *ja, double *a, int nnzA,  pdata* prec_data, double *vec_in, double *vec_out);
void it_jacobi(int *ia, int *ja, double *a, int nnzA,  pdata* prec_data, double *vec_in, double *vec_out);
void line_jacobi(int *ia, int *ja, double *a, int nnzA,  pdata* prec_data, double *vec_in, double *vec_out);
