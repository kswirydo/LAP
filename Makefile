CC := g++
NVCC := nvcc
AR := ar
INCLUDES  :=
CUDA_LIBS := -lcusparse -lcublas
FLAGS := -O3 -g
LIBS = -lm
OBJS = cuda_blas.o simple_blas.o blas.o cg.o GS.o it_jacobi.o line_jacobi.o prec.o cg_driver.o

# Rules                                                                         
%.o: %.c
	${CC} ${FLAGS} ${INCLUDES} -o $@ -c $<

%.o: %.cu
	${NVCC} ${FLAGS} ${CUDA_LIBS} ${INCLUDES} -o $@ -c $<

lap:${OBJS}
	${NVCC} ${CFLAGS} ${INCLUDES} -o $@ ${OBJS} ${LIBS} ${CUDA_LIBS}

