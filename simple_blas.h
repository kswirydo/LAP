//prototypes
//

double simple_dot (const int n, const double *v, const double *w);

void simple_scal (const int n, const double alpha, double *v);

void simple_axpy (const int n, const double alpha, const double *x, double *y);

void simple_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const int *a, const double *x, double *result);

void simple_lower_triangular_solve(const int n, const int nnz, const int *lia, const int *lja, const int *la, const double *x, double *result);

void simple_uppper_triangular_solve(const int n, const int nnz, const int *uia, const int *uja, const int *ua, const double *x, double *result);

void simple_vec_vec(const int n, const double * x, const double * y, double *res);


void simple_vector_sqrt(const int n, const double *v, double *res){

void simple_vector_reciprocal(const int n, const double *v, double *res);

void simple_vec_copy(const int n, const double *src, double *dest);
