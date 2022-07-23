#include <string>
#include "cudaMem.h" 
void * mallocForDevice(void *x, int n, int size){
//printf ("size is %d \n", size*n);  

cudaError  t = cudaMalloc ((void **)&x,n* size);

//printf("is x NULL? %d error is %d\n", x == NULL, t);
return x;
}
void memcpyDevice (void * dest, void * src, int n, int size, char * type){
  if (strcmp("H2D", type) == 0){
  //  printf("H2D \n"); 
    cudaMemcpy(dest, src, size * n, cudaMemcpyHostToDevice);
  }
  if (strcmp("D2H", type) == 0){
   // printf("D2H \n"); 
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
  }
  if (strcmp("D2D", type) == 0){
   // printf("D2D \n"); 
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
  }
}
