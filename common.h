struct pdata{
 
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
}
