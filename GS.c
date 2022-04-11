//std Gauss Seidel with tri solves
void GS_std(int *ia, int *ja, double *a, pdata* prec_data,int k, double *vec_in, double *vec_out){

  int n = prec_data->n;

  vec_zero(n, vec_out);
  //backward sweep
  for (int i=0; i<k; ++i){
    //x = x + L \ ( b - As*x );
    //prec_data->aux_vec1 = A*vec_out 

    csr_matvec(n, nnz,ia,  ja,  a, vec_out,  prec_data->aux_vec1);
    vec_copy(vec_in, aux_vec2);
    //aux_vec2 = aux_vec1*(-1) +vec_in
    axpy(n, -1.0f, prec_data->aux_vec1, prec_data->aux_vec2);
    //tri solve L^{-1}*aux_vec2

    lower_triangular_solve(n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la, prec_data->aux_vec2, prec_data->aux_vec1);

    axpy(n, 1.0f, prec_data->aux_vec1, vec_out);
  }


  //forward sweep
  for (int i=0; i<k; ++i){
    //x = x + L \ ( b - As*x );
    //prec_data->aux_vec1 = A*vec_out 

    csr_matvec(n, nnz,ia,  ja,  a, vec_out,  prec_data->aux_vec1);
    vec_copy(vec_in, aux_vec2);
    //aux_vec2 = aux_vec1*(-1) +vec_in
    axpy(n, -1.0f, prec_data->aux_vec1, prec_data->aux_vec2);
    //tri solve U^{-1}*aux_vec2

    upper_triangular_solve(n, prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua, prec_data->aux_vec2, prec_data->aux_vec1);

    axpy(n, 1.0f, prec_data->aux_vec1, vec_out);
  }
}
//iterative GS v1
void GS_it(int *ia, int *ja, double *a, pdata* prec_data,int k, double *vec_in, double *vec_out){


}
//iterative GS v2
void GS_it2(int *ia, int *ja, double *a, pdata* prec_data,int k, double *vec_in, double *vec_out){


}
