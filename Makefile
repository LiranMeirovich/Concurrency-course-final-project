#note: 'make build' will compile all files even when its not necessary
build:
	mpicxx -fopenmp -c ParallelProject.c -o ParallelProject.o
	# note: cFunctions.c is a "C" file (not C++) but it can be compiled as
	# a C++ file with mpicxx (which uses a C++ compiler). 
	# gcc -c cFunctions.c -o cFunctions.o 
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	#linking:
	mpicxx  -fopenmp -o proj  ParallelProject.o cFunctions.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	
clean:
	rm -f *.o ./proj

run:
	mpiexec -n 2 ./proj




