#include "common.h"
#include "blas.h"
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
       ){
  double * r = (double *) calloc (n, sizeof(double));
  double * w = (double *) calloc (n, sizeof(double));
  double * p = (double *) calloc (n, sizeof(double));
  double * q = (double *) calloc (n, sizeof(double));
  double alpha, beta, tolrel, rho_current, rho_previous, pTq;
  int notconv =1, iter =0;
  //compute initial norm of r
  //r = A*x
  csr_matvec(n, nnz, ia, ja, a, x, r);
  //r = -b +r = Ax-b
  axpy(n, -1.0f, b, r);
  // r=(-1.0)*r
  scal(n, -1.0f, r);
  //norm of r
  res_norm_history[0] = dot(n, r,r);
  res_norm_history[0] = sqrt(res_norm_history[0]);
  tolrel = tol*res_norm_history[0];
    printf("CG: it %d, res norm %5.5e \n",0, res_norm_history[0]);
  while (notconv){
    prec_function(ia, ja, a, nnz, prec_data, r, w);
    // rho_current = r'*w;
    rho_current = dot(n, r, w);
    if (iter == 0){
      vec_copy(n, w, p);
    }
    else{
      beta = rho_current/rho_previous;
    //  printf("scaling by beta = %5.5e, rho_current = %5.5e, rho_previous = %5.5e \n", beta, rho_current, rho_previous);
      // p = w+bet*p;
      scal(n, beta, p);
      axpy(n, 1.0, w, p);
    }
    //  q = As*p;
    csr_matvec(n, nnz, ia, ja, a, p, q);
    //  alpha = rho_current/(p'*q);
    pTq = dot(n, p, q);
    alpha = rho_current/pTq; 
    //printf("p^Tq = %5.5e,rho_current = %5.5e, alpha = %5.5e \n", pTq, rho_current, alpha);
    //x = x + alph*p;
    axpy(n, alpha, p, x );
    // r = r - alph*q;
    axpy(n, (-1.0)*alpha, q, r );
    //norm of r
    iter++;
    res_norm_history[iter] = dot(n, r,r);
    res_norm_history[iter] = sqrt(res_norm_history[iter]);
    printf("CG: it %d, res norm %5.5e \n",iter, res_norm_history[iter]);
    //check convergence
    if ((res_norm_history[iter])<tolrel)
    {
      *flag = 0;
      notconv = 0;
      *it = iter; 
    } else {
      if (iter>maxit){
        *flag = 1;
        notconv = 0;
        *it = iter; 
      }
    }
    rho_previous = rho_current;
  }//while
}//cg

