
void it_jacobi(int *ia, int *ja, double *a, pdata* prec_data,int k, double *vec_in, double *vec_out){

  int n = prec_data->n;



  //vec_out = v.*vec_in;

  vec_vec(n, prec_data->d_r, vec_in, vec_out);

  for (int i=0; i<k; ++i){

    //vec_out = (v).*(vec_in - U*vec_out-L*vec_out);
    //aux_vec1 = L*vec_out 
    csr_matvec(n, prec_data->lnnz,  prec_data->lia,  prec_data->lja,  prec_data->la, vec_out,  prec_data->aux_vec1);
    //aux_vec2=U*vec_oout
    csr_matvec(n, prec_data->unnz,  prec_data->uia,  prec_data->uja,  prec_data->ua, vec_out,  prec_data->aux_vec2);
    //aux_vec1 += aux_vec2;
    axpy(n, (1.0f), prec_data->aux_vec1, prec_data->aux_vec2);
    //vec_out = vec_in;
    vec_copy(n, vec_in, vec_out);
    //vec_out = vec_out + (-1)*aux_vec1'
    axpy(n, -1.0f, prec_data->aux_vec1, vec_out);
    //multiply by D^{-1}
    vec_vec(n, vec_out, prec_data->d_r, vec_out);
    
  }

}

