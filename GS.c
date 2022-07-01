//std Gauss Seidel with tri solves
#include "common.h"
#include "blas.h"
void GS_std(int *ia, int *ja, double *a, int nnzA,  pdata* prec_data, double *vec_in, double *vec_out){

  int n = prec_data->n;
  int k = prec_data->k;
  double one = 1.0;
  double zero = 0.0;  
  double minusone = -1.0;
  vec_zero(n, vec_out);
//printf("k = %d\n", k);
  //backward sweep
  for (int i=0; i<k; ++i){
    //x = x + L \ ( b - As*x );
    //printf("before mv: %f \n", dot(n, vec_out, vec_out)); 
    vec_copy(n, vec_in, prec_data->aux_vec2);
    csr_matvec(n, nnzA,ia,  ja,  a, vec_out,  prec_data->aux_vec2, &minusone, &one);
    //printf("after mv: %f \n", dot(n, prec_data->aux_vec1, prec_data->aux_vec1)); 
    //aux_vec2 = aux_vec1*(-1) +vec_in
    //tri solve L^{-1}*aux_vec2

    //printf(" norm r sq: %f \n", dot(n, prec_data->aux_vec2, prec_data->aux_vec2)); 
    //for (int j=0; j<10; ++j) printf("r[%d]=%f\n", j, prec_data->aux_vec2[j]);
    lower_triangular_solve(n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la,prec_data->d, prec_data->aux_vec2, prec_data->aux_vec1);
    //for (int j=0; j<10; ++j) printf("Lr[%d]=%f\n", j, prec_data->aux_vec1[j]);
    //printf(" norm sq after L ts: %f \n", dot(n, prec_data->aux_vec1, prec_data->aux_vec1)); 

    axpy(n, 1.0f, prec_data->aux_vec1, vec_out);
  }


  //forward sweep
  for (int i=0; i<k; ++i){
    //x = x + L \ ( b - As*x );
    //prec_data->aux_vec1 = A*vec_out 

    vec_copy(n, vec_in, prec_data->aux_vec2);
    csr_matvec(n, nnzA,ia,  ja,  a, vec_out,  prec_data->aux_vec2, &minusone, &one);
    //tri solve U^{-1}*aux_vec2

    upper_triangular_solve(n, prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua,prec_data->d, prec_data->aux_vec2, prec_data->aux_vec1);

    axpy(n, 1.0f, prec_data->aux_vec1, vec_out);
  }
}
//iterative GS v1
void GS_it(int *ia, int *ja, double *a,int nnzA, pdata* prec_data, double *vec_in, double *vec_out){

  int n = prec_data->n;

  int k = prec_data->k;

  double one = 1.0;
  double zero = 0.0;  
  double minusone = -1.0;
  int m = prec_data->m;
  //printf("m = %d k = %d \n", m, k);  
  //set vec_out to 0
  vec_zero(n, vec_out); 
  //outer loop
  for (int j=0; j<m; ++j){
    //r = b - A*x
    vec_copy(n, vec_in, prec_data->aux_vec2);
  
  csr_matvec(n, nnzA,ia,  ja,  a, vec_out,  prec_data->aux_vec2, &minusone, &one);
//printf("res nrm %f \n", dot(n,  prec_data->aux_vec2,  prec_data->aux_vec2));  
  //r = aux_vec2 = aux_vec1*(-1) +vec_in
    // y = aux_vec1 = D^{-1}aux_vec2
    vec_vec(n, prec_data->aux_vec2, prec_data->d_r, prec_data->aux_vec1); 
//printf("res nrm after scaling %f \n", dot(n,  prec_data->aux_vec1,  prec_data->aux_vec1));  
    for (int i=0; i<k; ++i){
      //y = v.*(r-L*y);
      //vec3 = L*vec1    
      vec_copy(n, prec_data->aux_vec2, prec_data->aux_vec3);      
      csr_matvec(n, prec_data->lnnz,prec_data->lia,prec_data->lja,  prec_data->la, prec_data->aux_vec1, prec_data->aux_vec3, &minusone, &one);
      //axpy

//printf("\t res nrm inside first loop, after mv  %f \n", dot(n,   prec_data->aux_vec3,  prec_data->aux_vec3));  
      vec_vec(n, prec_data->aux_vec3, prec_data->d_r, prec_data->aux_vec1); 
    }

//printf("res nrm after first loop %f \n", dot(n,  prec_data->aux_vec1,  prec_data->aux_vec1));  
    for (int i=0; i<k; ++i){
      //y = v.*(r-L*y);
      vec_copy(n, prec_data->aux_vec2, prec_data->aux_vec3);      
      csr_matvec(n, prec_data->unnz,prec_data->uia,prec_data->uja,  prec_data->ua, prec_data->aux_vec1, prec_data->aux_vec3, &minusone, &one);
      //axpy
      vec_vec(n, prec_data->aux_vec3, prec_data->d_r, prec_data->aux_vec1); 
    }

//printf("res nrm after second %f \n", dot(n,  prec_data->aux_vec1,  prec_data->aux_vec1));  
    //vec_out = vec_out + vec1  
    axpy(n, 1.0f, prec_data->aux_vec1, vec_out);
//printf("res nrm after update  %f \n", dot(n,  vec_out, vec_out));  
  }
}
//iterative GS v2
void GS_it2(int *ia, int *ja, double *a,int nnzA, pdata* prec_data, double *vec_in, double *vec_out){

  int k = prec_data->k;
  int m = prec_data->m;
  double one = 1.0;
  double zero = 0.0;  
  double minusone = -1.0; 
  //y = Dinv.*b;
  //
  int n = prec_data->n;
  vec_vec(n, vec_in, prec_data->d_r, prec_data->aux_vec1); 
  //outer loop
  for (int j=0; j<m; ++j){

    //inner loop 1
    for (int i=0; i<1; ++i){
      //L*(Dinv*b)
      csr_matvec(n, prec_data->lnnz,prec_data->lia,  prec_data->lja,  prec_data->la, prec_data->aux_vec1,  prec_data->aux_vec2, &one, &zero);
      //U*(Dinv*b)
      csr_matvec(n, prec_data->unnz,prec_data->uia,  prec_data->uja,  prec_data->ua, prec_data->aux_vec1,  prec_data->aux_vec3, &one, &zero);
      //(U+L)Dinv*b
      axpy(n, 1.0f, prec_data->aux_vec3, prec_data->aux_vec2);
      vec_copy(n, vec_in, prec_data->aux_vec3);
      //aux3  = vec_in-1.0f*(U+L)Dinv*b
      axpy(n, -1.0f, prec_data->aux_vec2, prec_data->aux_vec3);
      //scale
      vec_vec(n, prec_data->aux_vec3, prec_data->d_r, prec_data->aux_vec2);
    }//inner loop 1
    //compute residual:  r = b - L*y; 
    // vec3 = b 
    vec_copy(n, vec_in, prec_data->aux_vec3);
    //vec1 = L*y = L*vec2
    csr_matvec(n, prec_data->lnnz,prec_data->lia,  prec_data->lja,  prec_data->la, prec_data->aux_vec2,  prec_data->aux_vec3, &minusone, &one);
    //r = b-L*y : vec3 =  vec3 - vec1
    //inner loop 2
    for (int i=0; i<k; ++i){
      // y = (v).* ( r - U * y );
      //   vec1 = U*vec2 = U*y
      vec_copy(n, prec_data->aux_vec3, prec_data->aux_vec1);
      csr_matvec(n, prec_data->unnz,prec_data->uia,  prec_data->uja,  prec_data->ua, prec_data->aux_vec2,  prec_data->aux_vec1, &minusone, &one);
      //leave r alone dont change

      //vec2 = vec2 -vec1 = r-U*y
      //scale

      vec_vec(n, prec_data->d_r, prec_data->aux_vec1, prec_data->aux_vec2);
    }
    //residual again
    //  r = b - U*y;
    // vec3 = b 
    vec_copy(n, vec_in, prec_data->aux_vec3);
    //vec1 = U*y = U*vec2
    csr_matvec(n, prec_data->unnz,prec_data->uia,  prec_data->uja,  prec_data->ua, prec_data->aux_vec2,  prec_data->aux_vec3, &minusone, &one);
    //r = b-L*y : vec3 =  vec3 - vec1
    //inner loop 3
    for (int i=0; i<k; ++i){
      //    y = (v).* ( r - L * y );
      //   vec1 = L*vec2 = L*y
      vec_copy(n, prec_data->aux_vec3, prec_data->aux_vec1);
      csr_matvec(n, prec_data->lnnz,prec_data->lia,  prec_data->lja,  prec_data->la, prec_data->aux_vec2,  prec_data->aux_vec1, &minusone, &one);
      //leave r alone dont change

      //vec2 -vec1 = r-L*y
      //scale

      vec_vec(n, prec_data->d_r, prec_data->aux_vec1, prec_data->aux_vec2);
    }
    vec_copy(n, prec_data->aux_vec2,prec_data->aux_vec1);

  }//outer loop

  vec_copy(n, prec_data->aux_vec2, vec_out);

}

