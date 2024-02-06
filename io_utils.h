#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#pragma once

/* needed for easy sorting */
struct indexPlusValue
{
  double value;
  int idx;
};
typedef struct indexPlusValue indexPlusValue;

/* neded for qsort */
static int indexPlusValue_comp(const void *a, const void *b)
{
  const struct indexPlusValue *da = (indexPlusValue *)a;
  const struct indexPlusValue *db = (indexPlusValue *)b;

  return da->idx < db->idx ? -1 : da->idx > db->idx;
}

/* matrix data structure */
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
  
  bool symmetric;
} mmatrix;

void read_mm_file(const char *matrixFileName, mmatrix *A);
void read_adjacency_file(const char *matrixFileName, mmatrix *A); 
void coo_to_csr(mmatrix *A);
void read_rhs(const char *rhsFileName, double *rhs);
void split(mmatrix *A, mmatrix *L, mmatrix *U, mmatrix *D);
void create_L_and_split(mmatrix *A, mmatrix *L, mmatrix *U, mmatrix *D, int weighted);

