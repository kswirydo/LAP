CC := gcc
NVCC := nvcc
AR := ar
INCLUDES  :=
CUDA_LIBS := -lcusparse -lcublas
FLAGS := -O3 -g
LIBS = -lm
OBJS = simple_blas.o cuda_blas.o blas.o cg.o GS.o it_jacobi.o line_jacobi.o prec.o cg_driver.o 
# Rules
.c.o:
	${CC} ${FLAGS} ${INCLUDES} -c $<
.cu.o:
	${NVCC} ${FLAGS} ${INCLUDES} -c $<

lap:${OBJS}
	${CC} ${CFLAGS} ${INCLUDES} -o $@ ${OBJS} ${LIBS} ${CUDA_LIBS}
