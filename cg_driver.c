
//Written by KS, Mar 2022
//vanilla C version of Laplacian solver.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "common.h"
#include "blas.h"

//Gauss-Seidel, classic version

double time_Symbolic = 0.0;
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
#define MAXIT 50000
// needed for easy sorting
struct indexPlusValue
{
  double value;
  int idx;
};
typedef struct indexPlusValue indexPlusValue;

//neded for qsort

static int indexPlusValue_comp(const void *a, const void *b)
{
  const struct indexPlusValue *da = (indexPlusValue *)a;
  const struct indexPlusValue *db = (indexPlusValue *)b;

  return da->idx < db->idx ? -1 : da->idx > db->idx;
}

typedef struct
{
  int *coo_rows;
  int *coo_cols;
  double *coo_vals;

  int *csr_ia;
  int *csr_ja;
  double *csr_vals;

  int n;
  int m;
  int nnz;
  int nnz_unpacked; //nnz in full matrix;
} mmatrix;
//read the matrix (into messy COO)

void read_mm_file(const char *matrixFileName, mmatrix *A)
{
  // this reads triangular matrix but expands into full as it goes (important)
  int noVals = 0;
  FILE *fpm = fopen(matrixFileName, "r");

  char lineBuffer[256];
  //first line, should start with "%%"
  fgets(lineBuffer, sizeof(lineBuffer), fpm);
  char * s = strstr(lineBuffer, "pattern");
  if (s != NULL) noVals =1; 
  printf("noVals? %d \n", noVals);	
  while (lineBuffer[0] == '%'){ 
    //printf("Still wrong line: %s \n", lineBuffer);
    fgets(lineBuffer, sizeof(lineBuffer), fpm);
  }

  //first line is size and nnz, need this info to allocate memory
  sscanf(lineBuffer, "%ld %ld %ld", &(A->n), &(A->m), &(A->nnz));
  //printf("Matrix size: %d x %d, nnz %d \n",A->n, A->m, A->nnz );
  //allocate

  A->coo_vals = (double *)calloc(A->nnz+A->n, sizeof(double));
  A->coo_rows = (int *)calloc(A->nnz+A->n, sizeof(int));
  A->coo_cols = (int *)calloc(A->nnz+A->n, sizeof(int));
#if 1
  //read
  int r, c;
  double val;
  int i = 0;
  while (fgets(lineBuffer, sizeof(lineBuffer), fpm) != NULL)
  {
    if (noVals == 0){
      sscanf(lineBuffer, "%d %d %lf", &r, &c, &val);
      A->coo_vals[i] = val;
    }else {
      sscanf(lineBuffer, "%d %d", &r, &c);
      A->coo_vals[i] = 1.0;
    }    

    A->coo_rows[i] = r - 1;
    A->coo_cols[i] = c - 1;
    i++;
    if ((c < 1) || (r < 1))
      printf("We have got A PROBLEM! %d %d %16.16f \n", r - 1, c - 1, val);
  }//while
  //main diagonal of A is 0; but L = D-A  so it is not 0 in the Laplacian.
  //this is done to avoid updating CSR pattern
  for (int j=0; j<A->n; ++j){
    A->coo_rows[i] = j;
    A->coo_cols[i] = j;
    A->coo_vals[i] = 1.0f;
    i++;
  } 
  A->nnz+=A->n;
  fclose(fpm);
#endif
}


//COO to CSR
//usual stuff

void coo_to_csr(mmatrix *A)
{
  //this is diffucult
  //first, decide how many nnz we have in each row
  int *nnz_counts;
  nnz_counts = (int *)calloc(A->n, sizeof(int));
  int nnz_unpacked = 0;
  for (int i = 0; i < A->nnz; ++i)
  {
    nnz_counts[A->coo_rows[i]]++;
    nnz_unpacked++;
    if (A->coo_rows[i] != A->coo_cols[i])
    {
      nnz_counts[A->coo_cols[i]]++;
      nnz_unpacked++;
    }
  }
  //allocate full CSR structure
  A->nnz_unpacked = nnz_unpacked;
  A->csr_vals = (double *)calloc(A->nnz_unpacked, sizeof(double));
  A->csr_ja = (int *)calloc(A->nnz_unpacked, sizeof(int));
  A->csr_ia = (int *)calloc((A->n) + 1, sizeof(int));
  indexPlusValue *tmp = (indexPlusValue *)calloc(A->nnz_unpacked, sizeof(indexPlusValue));
  //create IA (row starts)
  A->csr_ia[0] = 0;
  for (int i = 1; i < A->n + 1; ++i)
  {
    A->csr_ia[i] = A->csr_ia[i - 1] + nnz_counts[i - 1];
  }

  int *nnz_shifts = (int *)calloc(A->n, sizeof(int));
  int r, start;

  for (int i = 0; i < A->nnz; ++i)
  {
    //which row
    r = A->coo_rows[i];
    start = A->csr_ia[r];
    if ((start + nnz_shifts[r]) > A->nnz_unpacked)
      printf("index out of bounds\n");
    tmp[start + nnz_shifts[r]].idx = A->coo_cols[i];
    tmp[start + nnz_shifts[r]].value = A->coo_vals[i];

    nnz_shifts[r]++;

    if (A->coo_rows[i] != A->coo_cols[i])
    {

      r = A->coo_cols[i];
      start = A->csr_ia[r];

      if ((start + nnz_shifts[r]) > A->nnz_unpacked)
        printf("index out of boubns 2\n");
      tmp[start + nnz_shifts[r]].idx = A->coo_rows[i];
      tmp[start + nnz_shifts[r]].value = A->coo_vals[i];
      nnz_shifts[r]++;
    }
  }
  //now sort whatever is inside rows

  for (int i = 0; i < A->n; ++i)
  {

    //now sorting (and adding 1)
    int colStart = A->csr_ia[i];
    int colEnd = A->csr_ia[i + 1];
    int length = colEnd - colStart;

    qsort(&tmp[colStart], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }

  //and copy
  for (int i = 0; i < A->nnz_unpacked; ++i)
  {
    A->csr_ja[i] = tmp[i].idx;
    A->csr_vals[i] = tmp[i].value;
  }
#if 0
  for (int i=0; i<10; i++){
    printf("this is row %d \n", i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; ++j){ 
      printf("  (%d, %f)  ", A->csr_ja[j], A->csr_vals[j] );			

    }
    printf("\n");

  }
#endif
  free(nnz_counts);
  free(tmp);
  free(nnz_shifts);
  free(A->coo_cols);
  free(A->coo_rows);
  free(A->coo_vals);

}

void create_L_and_split(mmatrix *A, mmatrix *L, mmatrix *U,mmatrix *D, int weighted){
  // we need access to L, U, and D explicitely, not only to the Laplacian
  // w decides whether weighted (w=1) or not (w=0)
  // we need degree of every row
  // allocate L and U bits and pieces;
  L->csr_ia = (int *) calloc (A->n+1, sizeof(int));
  U->csr_ia = (int *) calloc (A->n+1, sizeof(int));
  D->csr_ia = (int *) calloc (A->n+1, sizeof(int));


  L->csr_ja = (int *) calloc (A->nnz-A->n, sizeof(int));
  U->csr_ja = (int *) calloc (A->nnz-A->n, sizeof(int));
  D->csr_ja = (int *) calloc (A->n, sizeof(int));


  L->csr_vals = (double *) calloc (A->nnz-A->n, sizeof(double));
  U->csr_vals = (double *) calloc (A->nnz-A->n, sizeof(double));
  D->csr_vals = (double *) calloc (A->n, sizeof(double));

  int *DD = (int*) calloc(A->n, sizeof(int));
  int iu =0, il=0;
  int col;
  for (int i=0; i<A->n; ++i){
//    DD[i] = A->csr_ia[i+1]-A->csr_ia[i];
    //dont count yourself
 //   DD[i]--; 
 

    DD[i] = -1.0; //dont count yourself
    for (int j = A->csr_ia[i]; j < A->csr_ia[i+1]; ++j){
      DD[i] += A->csr_vals[j];
    }  

 }
  //    printf("vertex %d has degree %d \n", i, DD[i]);
  double Dsqrt;
  for (int i=0; i<A->n; ++i){
    L->csr_ia[i] = il;
    U->csr_ia[i] = iu;
    if (weighted) Dsqrt = 1.0f/sqrt(DD[i]);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; ++j){
      col = A->csr_ja[j];
      if (col == i) {
        if (!weighted){
          A->csr_vals[j]=(double) DD[i]; 
          D->csr_vals[i] = A->csr_vals[j];
          D->csr_ia[i] = i;
          D->csr_ja[i] = i;
        }
        else {
          //printf("Weighted, putting 1.0 on the diagonal \n");
          A->csr_vals[j]=(double)1.0f; 
          D->csr_vals[i] = A->csr_vals[j];
          D->csr_ia[i] = i;
          D->csr_ja[i] = i;
        }     
      }
      else{
        if (!weighted){
          A->csr_vals[j] = (-1)*A->csr_vals[j];
        }
        else {

          A->csr_vals[j] = (-1.0f)*A->csr_vals[j]*Dsqrt*(1.0f/sqrt(DD[col]));
          //printf("Weighted, putting %f in (%d, %d) \n", A->csr_vals[j], i, j);
        }
      }

      if (i<col){//row< col, upper part
        U->csr_ja[iu] = A->csr_ja[j];
        U->csr_vals[iu] = A->csr_vals[j];
        iu++;
      }
      if (i>col){//row > col, lower part

        L->csr_ja[il] = A->csr_ja[j];
        L->csr_vals[il] = A->csr_vals[j];
        il++;
      }
    }//for with i

  }//for with j
  D->csr_ia[A->n] = A->n;
  L->csr_ia[A->n] = il;
  U->csr_ia[A->n] = iu;
  L->n = A->n;
  L->m = A->m;
  L->nnz = A->nnz-A->n;

  U->n = A->n;
  U->m = A->m;
  U->nnz = A->nnz-A->n;

  D->n = A->n;
  D->m = A->m;
  D->nnz = A->n;

#if 0	
  printf("\n\n ==== A ==== \n");
  for (int i=0; i<10; i++){
    printf("this is row %d \n", i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; ++j){ 
      printf("  (%d,%f)  ", A->csr_ja[j], A->csr_vals[j] );			

    }
    printf("\n");

  }

  printf("\n\n ==== D ==== \n");
  for (int i=0; i<10; i++){
    printf("this is row %d \n", i);
    for (int j=D->csr_ia[i]; j<D->csr_ia[i+1]; ++j){ 
      printf("  (%d,%f)  ", D->csr_ja[j], D->csr_vals[j] );			

    }
    printf("\n");

  }

  printf("\n\n ==== L ==== \n");
  for (int i=0; i<10; i++){
    printf("this is row %d with %d = %d - %d entries \n", i, L->csr_ia[i+1]-L->csr_ia[i], L->csr_ia[i+1], L->csr_ia[i]);
    for (int j=L->csr_ia[i]; j<L->csr_ia[i+1]; ++j){ 
      printf(" (%d,%f)  ", L->csr_ja[j], L->csr_vals[j] );			

    }
    printf("\n");

  }
  printf("\n\n ==== U ==== \n");
  for (int i=0; i<10; i++){
    printf("this is row %d \n", i);
    for (int j=U->csr_ia[i]; j<U->csr_ia[i+1]; ++j){ 

      printf(" (%d,%f)  ", U->csr_ja[j], U->csr_vals[j] );			
    }
    printf("\n");

  }
#endif
}


int main(int argc, char *argv[])
{
  double time_CG = 0.0;
  struct timeval t1, t2;
  srand(12345);
  const char * matrixFileName = argv[1];
  const char * precName = argv[2];
  double cg_tol = atof(argv[3]);
  int cg_maxit = atoi(argv[4]);
  int M = atoi(argv[5]);
  int K = atoi(argv[6]);
  mmatrix *A, *L, *U, *D;
  A = (mmatrix *)calloc(1, sizeof(mmatrix));
  L = (mmatrix *)calloc(1, sizeof(mmatrix));
  U = (mmatrix *)calloc(1, sizeof(mmatrix));
  D = (mmatrix *)calloc(1, sizeof(mmatrix));
  read_mm_file(matrixFileName, A);
  coo_to_csr(A); 

  int weighted = 0;
  create_L_and_split(A,L,U,D, weighted);
  //at this point we know our LAPLACIAN !
  //NOTE: Laplacian is stored in A= L+U+D (matrix splitting).
  //DONT CONFUSE degree matrix with D (D is a diagonal of A) and L with Laplacian (L is lower triangular part of A)

  pdata * prec_data;  
  //for(int i=0; i<A->n; ++i) printf("b[%d] = %f\n", i, b[i]);
  prec_data = (pdata *)calloc(1, sizeof(pdata));
  prec_data->n = A->n;
  prec_data->prec_op = (char *) precName;
  prec_data->k = K;
  prec_data->m = M;

  mmatrix * d_A = (mmatrix *)calloc(1, sizeof(mmatrix));
  mmatrix * d_L = (mmatrix *)calloc(1, sizeof(mmatrix));
  mmatrix * d_U = (mmatrix *)calloc(1, sizeof(mmatrix));

  d_A->n = A->n;
  d_A->nnz = A->nnz;
  d_A->nnz_unpacked = A->nnz_unpacked;
  //now, if the preconditioner is GS_it or GS_it2, the setup is correct but if it is  
  //GS_std, we need to have the diagonal ADDED TO L AND U
  printf("\n\n");
  printf("Solving graph Laplacian linear system for %s\n", matrixFileName);
  if (cg_maxit>MAXIT) {
    printf("\t [ WARNING:]  maxit cannot be larger than %d, re-setting to MAX \n", MAXIT);
    cg_maxit = MAXIT;
  }
  printf("\t Matrix size    : %d x %d \n", A->n, A->n);
  printf("\t Matrix nnz     : %d  \n", A->nnz);
  printf("\t Preconditioner : %s\n", prec_data->prec_op);
  printf("\t CG tolerance   : %2.16g\n", cg_tol);
  printf("\t CG maxit       : %d \n", cg_maxit);
  printf("\t M (outer it)   : %d \n", M);
  printf("\t K (inner it)   : %d \n", K);
  if (weighted) {
    printf("\t Weighted ?     : Yes\n\n\n");
  } else {
    printf("\t Weighted ?     : No\n\n\n");
  }
  //	printf("preconditioner: %s L->nnz = %d L->nnz_unpacked = %d A->nnz %d A->nnz_unpacked %d\n", prec_data->prec_op, L->nnz, L->nnz_unpacked, A->nnz, A->nnz_unpacked);
#if (CUDA || HIP)
  if (strcmp(prec_data->prec_op, "GS_std")  == 0) {

    int *  new_L_ja = (int *) calloc (L->nnz+L->n, sizeof(int));
    int *  new_U_ja = (int *) calloc (U->nnz+U->n, sizeof(int));

    double *  new_L_a = (double *) calloc (L->nnz+L->n, sizeof(double));
    double * new_U_a = (double *) calloc (U->nnz+U->n, sizeof(double));
    int c = 0;
    for (int ii=0; ii<L->n; ++ii){
      for (int jj=L->csr_ia[ii]; jj<L->csr_ia[ii+1]; ++jj){
        new_L_ja[c] = L->csr_ja[jj];
        new_L_a[c] = L->csr_vals[jj];
        c++;     
      }
      //diagonal element
      new_L_ja[c] = ii;
      new_L_a[c] = D->csr_vals[ii]; 
      c++;  
    }
    c=0;
    for (int ii=0; ii<U->n; ++ii){

      //diagonal element
      new_U_ja[c] = ii;
      new_U_a[c] = D->csr_vals[ii]; 
      c++;  
      for (int jj=U->csr_ia[ii]; jj<U->csr_ia[ii+1]; ++jj){
        new_U_ja[c] = U->csr_ja[jj];
        new_U_a[c] = U->csr_vals[jj];
        c++;     
      }
    }
    //now shift row pointers
    for (int ii=1; ii<=A->n; ++ii){
      L->csr_ia[ii]+=ii;
      U->csr_ia[ii]+=ii;
    }
    L->nnz+=A->n;
    U->nnz+=A->n;

    free(L->csr_ja);
    free(L->csr_vals);

    free(U->csr_ja);
    free(U->csr_vals);
    L->csr_ja = new_L_ja;
    L->csr_vals = new_L_a;
    U->csr_ja = new_U_ja;
    U->csr_vals = new_U_a;
#if 0	

    free(L->csr_ja);
    free(L->csr_vals);
    L->csr_ja = new_L_ja;
    L->csr_vals = new_L_a;
    free(U->csr_ja);
    free(U->csr_vals);
    U->csr_ja = new_U_ja;
    U->csr_vals = new_U_a;
#endif
#if 0
    printf("\n\n ==== NEW L ==== \n");
    for (int i=0; i<10; i++){
      printf("this is row %d with %d = %d - %d entries \n", i, L->csr_ia[i+1]-L->csr_ia[i], L->csr_ia[i+1], L->csr_ia[i]);
      for (int j=L->csr_ia[i]; j<L->csr_ia[i+1]; ++j){ 
        printf("(  %d, %f ) ", L->csr_ja[j], L->csr_vals[j] );			

      }
      printf("\n");

    }
    printf("\n\n ====NEW  U ==== \n");
    for (int i=0; i<10; i++){
      printf("this is row %d \n", i);
      for (int j=U->csr_ia[i]; j<U->csr_ia[i+1]; ++j){ 
        printf(" ( %d, %f)  ", U->csr_ja[j], U->csr_vals[j] );			

      }
      printf("\n");
    } //for
#endif 


  }//if
#endif 
  //allocate space for the GPU
  //  double *d_e, *d_etilde, *d_b, *d_d;

  double *e = (double *) calloc (A->n, sizeof(double));
  double *b = (double *) calloc (A->n, sizeof(double));
  double *aux = (double *) calloc (A->n, sizeof(double));
  //vector of vertex degrees
  double *d = (double *) calloc (A->n, sizeof(double));
  for (int i=0; i<A->n; ++i) {
    e[i]= 1.0f;
    b[i] = pow((-1.0),((i%2)));
    //printf("b[%d] = %f\n",i, b[i] );
    //(double) (rand()%200)/(rand()%100);
   // d[i] = A->csr_ia[i+1]-A->csr_ia[i]-1; //dont count yourself
d[i] = D->csr_vals[i];   
//  printf("d[%d] = %f, 1/d = %16.16f \n", i, d[i], 1.0/d[i]);

  }

#if (CUDA || HIP)

  initialize_handles();
  double *d_aux;
  double *d_b;
  double *d_d;
  double *d_e;

  d_aux = (double *) mallocForDevice (d_aux,A->n, sizeof(double));


  d_b = (double*) mallocForDevice (d_b,A->n, sizeof(double));
  d_d = (double*) mallocForDevice (d_d,A->n, sizeof(double));
  d_e = (double*) mallocForDevice (d_e,A->n, sizeof(double));
  memcpyDevice(d_b, b,  A->n,sizeof(double) , "H2D");
  memcpyDevice(d_d, d,  A->n,sizeof(double) , "H2D");
  memcpyDevice(d_e, e,A->n,sizeof(double),  "H2D");
  //printf("norm sq b before projection %16.16f \n", dot(A->n, d_b, d_b));
  free(b);
  free(d);
  free(e);
  free(aux);
  aux=d_aux;
  b=d_b;
  e=d_e;
  d=d_d;
  //printf("CUDA. copying pointers\n");
  //create an rhs.
#endif

#if 1
#if 1
  double norme = (double) sqrt(A->n);  
  double one_over_norme = 1./norme;
  //	printf ("scaling e by %16.16f, norme %16.16e \n", one_over_norme, norme);
  if (weighted == 0){
    //non-weighted version
    double be;
    /* e = (1/norme) e;*/
    scal(A->n, one_over_norme, e);  
    /* be = b'*e*/
    be = dot(A->n, e, b);
    /*b = b-be*e; */
    be = (-1.0f) * be;
    double tt1, tt2;

    tt1 = dot(A->n, e,e);
    tt2 = dot(A->n, b, b);
    printf("norm of e before axpy %16.16f, n = %d \n", tt1, A->n);
    printf("norm of b before axpy %16.16f \n", tt2);

    axpy(A->n, be, e, b); 

    //	printf("norm b after projection %16.16f \n", dot(A->n, b, b));
  } else {
    //weighted version
    //aux = sqrt(d);`
    vector_sqrt(A->n, d, aux);

    //aux = aux.*e
    vec_vec(A->n, aux, e, aux);
    //De_norm = norm(D_De);
    double De_norm;

    De_norm = dot(A->n, aux, aux);  
    De_norm = 1.0/sqrt(De_norm);
    //De = (1/norm(De))*De;

    scal(A->n, De_norm, aux);

    //   bwe = b'*De;
    double bwe;

    bwe = dot(A->n, b, aux);  
    //bProjw = b- bwe*wetilde;
    bwe *= (-1.0f);
    axpy(A->n,bwe, aux, b);
  }
  // at this point the Laplacian and the rhs are created.


  prec_data->lnnz = L->nnz;
  prec_data->unnz = U->nnz;
#if (CUDA || HIP)
  prec_data->lia = (int*) mallocForDevice (prec_data->lia,(A->n+1), sizeof(int));
  prec_data->lja = (int*)  mallocForDevice (prec_data->lja,(L->nnz), sizeof(int));
  prec_data->la = (double*) mallocForDevice (prec_data->la,(L->nnz), sizeof(double));

  prec_data->uia = (int*)  mallocForDevice (prec_data->uia,(A->n+1), sizeof(int));
  prec_data->uja = (int*)  mallocForDevice (prec_data->uja,(U->nnz), sizeof(int));
  prec_data->ua = (double*)  mallocForDevice (prec_data->ua,(U->nnz), sizeof(double));


  memcpyDevice(prec_data->lia,L->csr_ia , (A->n+1),sizeof(int),  "H2D");
  memcpyDevice(prec_data->lja,L->csr_ja , (L->nnz),sizeof(int) , "H2D");
  memcpyDevice(prec_data->la,L->csr_vals , (L->nnz),sizeof(double),  "H2D");

  memcpyDevice(prec_data->uia,U->csr_ia ,(A->n+1), sizeof(int),  "H2D");
  memcpyDevice(prec_data->uja,U->csr_ja , (U->nnz), sizeof(int),  "H2D");
  memcpyDevice(prec_data->ua,U->csr_vals , (U->nnz), sizeof(double),  "H2D");
  prec_data->d_r = (double*) mallocForDevice (prec_data->d_r,(A->n), sizeof(double));
  if (!weighted){

    vector_reciprocal(A->n, d, prec_data->d_r);
  } else {

    double *dd = (double *) calloc (A->n, sizeof(double));
    for (int ii=0; ii<A->n; ++ii){
      dd[ii]=1.0;
    }

    memcpyDevice(prec_data->d_r,dd , (A->n), sizeof(double),  "H2D");
    free(dd);
  }
  //printf("norm of d %f norm of d_r %d \n", dot(A->n, d,d), dot(A->n, prec_data->d_r, prec_data->d_r));
  prec_data->d=d;

  prec_data->aux_vec1 = (double*) mallocForDevice (prec_data->aux_vec1,(A->n), sizeof(double));
  prec_data->aux_vec2 = (double*)  mallocForDevice (prec_data->aux_vec2,(A->n), sizeof(double));
  prec_data->aux_vec3 = (double*)  mallocForDevice (prec_data->aux_vec3,(A->n), sizeof(double));

  double * x;
  x = (double *)  mallocForDevice (x,(A->n), sizeof(double));
  vec_zero(A->n, x);  
  int *d_A_ia;
  int *d_A_ja;
  double * d_A_a;

  d_A_ia = (int *)  mallocForDevice ((d_A_ia),(A->n+1), sizeof(int));
  d_A_ja = (int *)  mallocForDevice ((d_A_ja),(A->nnz_unpacked), sizeof(int));
  d_A_a = (double *)  mallocForDevice ((d_A_a),(A->nnz_unpacked), sizeof(double));
  memcpyDevice(d_A_ia,A->csr_ia , sizeof(int) , (A->n+1), "H2D");
  memcpyDevice(d_A_ja,A->csr_ja , sizeof(int) , (A->nnz_unpacked), "H2D");
  memcpyDevice(d_A_a ,A->csr_vals , sizeof(double) , (A->nnz_unpacked), "H2D");
#if 0
  for (int i=0; i<A->n; i++){
    printf("this is row %d \n", i);
    for (int j=L->csr_ia[i]; j<L->csr_ia[i+1]; ++j){ 
      printf(" (%d, %f) ", L->csr_ja[j], L->csr_vals[j] );			

    }
    printf("\n");

  }
#endif
  //printf("last el of csr_ia: %d\n", A->csr_ia[A->n]);
  free(A->csr_ia);
  free(A->csr_ja);
  free(A->csr_vals);
  A->csr_ia = d_A_ia;
  A->csr_ja = d_A_ja;
  A->csr_vals = d_A_a;
  printf("is ia NULL? %d is ja NULL? %d \n", A->csr_ia == NULL, A->csr_ja == NULL);
  double one =1.0; double minusone=1.0;
#if CUDA 
  //printf("initializin spmv buffer \n");	
  initialize_spmv_buffer(A->n, 
                         A->nnz_unpacked,
                         A->csr_ia,
                         A->csr_ja,
                         A->csr_vals,
                         x,
                         b, 
                         &one, 
                         &minusone);
#else // HIP
  /*printf("ia null? %d ja null? %d a NULL? %d, nnz_unpacked %d \n",
    A->csr_ia == NULL,
    A->csr_ja == NULL,
    A->csr_vals == NULL,
    A->nnz_unpacked
    );*/
  analyze_spmv(A->n, 
               A->nnz_unpacked, 
               A->csr_ia,
               A->csr_ja,
               A->csr_vals,
               x,
               b, 
               "A"
              );
  if ((strcmp(prec_data->prec_op, "GS_it")  == 0) || (strcmp(prec_data->prec_op, "GS_it2")  == 0) ) {
    analyze_spmv(A->n, 
                 prec_data->lnnz,
                 prec_data->lia,
                 prec_data->lja,
                 prec_data->la,
                 x,
                 b, 
                 "L"
                );

    analyze_spmv(A->n, 
                 prec_data->unnz,
                 prec_data->uia,
                 prec_data->uja,
                 prec_data->ua,
                 x,
                 b, 
                 "U"
                );
  }
#endif
  if (strcmp(prec_data->prec_op, "GS_std")  == 0) {
    initialize_and_analyze_L_and_U_solve(A->n, 
                                         prec_data->lnnz,
                                         prec_data->lia,
                                         prec_data->lja,
                                         prec_data->la,
                                         prec_data->unnz,
                                         prec_data->uia,
                                         prec_data->uja,
                                         prec_data->ua);

  }

#if CUDA
  initialize_L_and_U_descriptors(A->n, 
                                 prec_data->lnnz,
                                 prec_data->lia,
                                 prec_data->lja,
                                 prec_data->la,
                                 prec_data->unnz,
                                 prec_data->uia,
                                 prec_data->uja,
                                 prec_data->ua);

#endif
#else //NOT cuda nor hip

  prec_data->lia = L->csr_ia;
  prec_data->lja = L->csr_ja;
  prec_data->la = L->csr_vals;

  prec_data->uia = U->csr_ia;
  prec_data->uja = U->csr_ja;
  prec_data->ua = U->csr_vals;



  double *dd = (double *) calloc (A->n, sizeof(double));
  vector_reciprocal(A->n, d, dd);

  double *aux_vec1 = (double *) calloc (A->n, sizeof(double));
  double *aux_vec2 = (double *) calloc (A->n, sizeof(double));
  double *aux_vec3 = (double *) calloc (A->n, sizeof(double));
  double *x = (double *) calloc (A->n, sizeof(double));
  prec_data->d = d;
  prec_data->d_r = dd;

  prec_data->aux_vec1 = aux_vec1;
  prec_data->aux_vec2 = aux_vec2;
  prec_data->aux_vec3 = aux_vec3;
#endif

  double *res_hist = (double *) calloc (26000, sizeof(double));
  int it, flag;
#if 1
  //printf("A->nnz = %d \n", A->nnz_unpacked);
  gettimeofday(&t1, 0);
  cg(A->n,
     A->nnz_unpacked,
     A->csr_ia, //matrix csr data
     A->csr_ja,
     A->csr_vals,
     x, //solution vector, mmust be alocated prior to calling
     b, //rhs
     cg_tol, //DONT MULTIPLY BY NORM OF B
     prec_data, //preconditioner data: all Ls, Us etc
     cg_maxit,
     &it, //output: iteration
     &flag, //output: flag 0-converged, 1-maxit reached, 2-catastrophic failure
     res_hist //output: residual norm history
    );
  gettimeofday(&t2, 0);
  time_CG = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
#endif
  printf("\n\n");
  printf("CG summary results \n");
  printf("\t Iters              : %d  \n", it);
  printf("\t Time               : %2.4f  \n", time_CG/1000.0);
  printf("\t Res. norm          : %2.16g  \n", res_hist[it]);
  printf("\t Preconditioner     : %s\n", prec_data->prec_op);
  if (flag == 0){
    printf("\t Reason for exiting : CG converged  \n");
  } else {
    if (flag == 1){

      printf("\t Reason for exiting : CG reached maxit \n");
    } else {

      printf("\t Reason for exiting : CG failed\n");
    }
  }
  // printf("cg done, it: %d it took %f s FLAG %d\n", it, time_CG/1000.0, flag);
#endif
#endif
  return 0;
}
