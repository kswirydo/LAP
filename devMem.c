#include <string>
#include "devMem.h" 
void * mallocForDevice(void *x, int n, int size){
  //printf ("size is %d \n", size*n);  
#if CUDA
  cudaError  t = cudaMalloc ((void **)&x,n* size);
#elif  ROCM
  hipError  t = hipMalloc ((void **)&x,n* size);
#endif
  //printf("is x NULL? %d error is %d\n", x == NULL, t);
  return x;
}

void memcpyDevice (void * dest, void * src, int n, int size, char * type){
  if (strcmp("H2D", type) == 0){
    #if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyHostToDevice);
    #elif ROCM
    hipMemcpy(dest, src, size * n, cudaMemcpyHostToDevice);
    #endif

#
  }
  if (strcmp("D2H", type) == 0){
    #if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
    #elif ROCM
    hipMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
    #endif
  }
  if (strcmp("D2D", type) == 0){
    #if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
    #elif ROCM
    hipMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
    #endif
  }
}
