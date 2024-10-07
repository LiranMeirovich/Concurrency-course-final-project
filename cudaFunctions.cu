#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cDefenitions.h"

//We will need a max of 2000 threads in total, so we think it would be better to just divide it into
//2 blocks of 1000, in the end each thread will only be incharge of 1 letter and the max size
//of a string is 2000 (this refers to secondary since the calculations will depend on that)
//we ended up using 1024 since reduction shown to us only works on powers of 2
#define BLOCK_DIM 1024

enum op{PLUS, MAX};

extern void dealWithMpi();


//can't use strlen from string.h on device
__device__ void cuda_strlen(char* str, int* size)
{
    *size = 0;
    while(*str)
    {
        (*size)++;
        str++;
    }
}

//taken from lecture and changed a bit
__device__ void reverse(int *a, int n)
{
    //only 1 block so we can use threadidx alone
    int index = threadIdx.x;
    int tmp;

    if (index < n/2) {
        tmp = a[index];
        a[index] = a[n-1-index];
        a[n-1-index] = tmp;
    }
}

//need to reset the string before continuing
__device__ char cuda_MS_reset(char ch)
{
        if(ch != 'Z')
            return ch + 1;
        else
            return 'A';
}


//in here each thread mutates up to 2 chars at a time and so it's a bit different than our normal MS
__device__ char cuda_MS(char ch)
{
        if(ch != 'Z')
            return ch + 1;
        else
            return 'A';
}


__device__ int op_case(int n1, int n2, op rec)
{
    switch(rec)
    {
        case PLUS:
            return n1 + n2;
            break;
        case MAX:
            return n1 > n2 ? n1:n2;
            break;
        default:
            printf("error op_case\n");
    }
}

__device__ void reduceMax(int *array, int size,int *location)
{
    int tid = threadIdx.x;
    //taken from lecture
    for(unsigned int stride = (BLOCK_DIM)/2; stride > 0; stride >>= 1)
    {
        if (tid < stride&&tid+stride<size+1){
		if(array[tid+stride] > array[tid]){
	       
			//printf("replacing %d and %d because of %d and %d Leeran's dick size is %d\n",location[tid],location[tid+stride],array[tid],array[tid+stride],size);
			array[tid]=array[tid+stride];
			location[tid] = location[tid+stride];
		}
		else if(array[tid+stride] == array[tid] && location[tid] < location[tid+stride])
			location[tid] = location[tid+stride];
        }
        __syncthreads();
    }
    //if(tid==0){
   //printf("[");
   //for(int i = 0; i<BLOCK_DIM*2;i++)
    //printf("%d, ",location[i]);
   //printf("]\n");
    //}
}

__device__ void reduce(int *array, int size, op choice)
{
    int tid = threadIdx.x;
    //taken from lecture
    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            array[tid] = op_case(array[tid+stride], array[tid], choice);
               
        __syncthreads();
    }
}


//taken from Homework 3 and adjusted to work on an arr of size 2000
__device__ void scan(int *array, int size, op choice)
{
   for (unsigned int stride=1; stride <= (size+1)/2+1; stride *= 2) {
        int v;
        int t;
        if (threadIdx.x >= stride)
            v = array[threadIdx.x - stride];
        //in here the thread will be incharged of its own index and index + 1000, in the end should work similarly to our original scan
        if (threadIdx.x + BLOCK_DIM >= stride)
            t = array[threadIdx.x + BLOCK_DIM - stride];
       
        __syncthreads(); /* wait until all threads finish reading
                   an element */

        if (threadIdx.x >= stride && threadIdx.x < size)
            array[threadIdx.x] += v;
        if (threadIdx.x + BLOCK_DIM >= stride && threadIdx.x < size)
            array[threadIdx.x + BLOCK_DIM] += t;
           

        __syncthreads(); /* wait until all threads finish updating an
                   element */
     }
     
}


__global__  void cudaLaunch(char* mainStr, int mainSize, char** strArrs, int arrSize, int* evals,
        int* value, int* offsets, int* mutations) {
        //printf("cudaTest?\n");
    int tid = threadIdx.x;//only using 1 block
    int tid2= tid + BLOCK_DIM; //this is the second index that the thread is incharge of

    if(tid==0) // for debugging purposes
    {
         //printf("mainStr:%s\n", mainStr);
         //printf("mainSize:%d\n", mainSize);
         //printf("strArrs[1]:%s\n", strArrs[1]);
         //printf("arrSize:%d\n", arrSize);
         //printf("evals[1][1]:%d\n", *(evals+1*MAXEVAL+1));
    }

    // Prep values for calculating depending on evals
    // we prepared 2000 in advance just in case, the alternative is to malloc specifically depending on max arr size
    // so we sacrifice memory for better performance
    __shared__ int values[BLOCK_DIM*2];
    __shared__ int differences[BLOCK_DIM*2];
    __shared__ int indexes[BLOCK_DIM*2];
    int maxOffset, localVal;
    int secSize;
    int mIndex, sIndex;
    char* secStr;
    __shared__ char mutatedStr[BLOCK_DIM*2+1];

    //if (tid == 0)
    //    printf("cudaTest\n");  

    //iterate over each string
    for (int i = 0; i < (arrSize+1)/2; i++)
    {
    	value[i] = -9999;
    	__syncthreads();
        secStr = strArrs[i]; //make reference for easy access to the str
        cuda_strlen(strArrs[i], &secSize);
        maxOffset = mainSize - secSize + 1;

        //this allows us to not copy the original str but rather just get the mutated already
        if(tid == secSize)
            mutatedStr[tid] = '\0';
        if(tid2 == secSize)
            mutatedStr[tid2] = '\0';

        if(tid < secSize)
            mutatedStr[tid] = cuda_MS(secStr[tid]);
        if(tid2 < secSize)
            mutatedStr[tid2] = cuda_MS(secStr[tid2]);
        //we wil lbe using a mutatedStr since it will always be the same
        //so we don't need to recalculate it each iteration and then reset it
        // if(tid==0)
        //     printf("str:%s, size:%d, mutate:%s\n",secStr, secSize, mutatedStr);


        //iterate over offset
        for (int j = 0; j < maxOffset; j++)
        {
            //calc all values and differences, it's convinient to do both at once
            //and doesn't really affect the flow of the program
            if(tid < secSize)
            {
                //calculate specific index for char inside eval matrix
                mIndex = (int)(mainStr[tid + j] - 'A');//j is offset
                sIndex = (int)(secStr[tid] - 'A');
                values[tid] = *(evals + mIndex*MAXEVAL + sIndex);

                //recalculate index for mutated string, in the end we only need the diff between the mutated and original
                //scan and reductions will make sure we get the best mutation between them
                sIndex = (int)(mutatedStr[tid] - 'A');
                differences[tid] = *(evals + mIndex*MAXEVAL + sIndex) - values[tid]; //calculates each individual difference
                if(tid2 < secSize)
                {
                    //we let each thread act as "2 threads" this way we can do scan with 1 block without too many changes
                    //to the one we learned
                    mIndex = (int)(mainStr[tid2 + j] - 'A');//j is offset
                    sIndex = (int)(secStr[tid2] - 'A');
                    values[tid2] = *(evals + mIndex*MAXEVAL + sIndex);

                    sIndex = (int)(mutatedStr[tid2] - 'A');
                    differences[tid2] = *(evals + mIndex*MAXEVAL + sIndex) - values[tid2];
                }
                else
                {
                    values[tid2] = 0;
                    differences[tid2] = 0;
                }
            }
            else
            {
                values[tid] = 0;
                values[tid2] = 0;
                differences[tid] = 0;
                differences[tid2] = 0;
            }

            //printf("tid:%d, values[tid]:%d i:%d\n", tid, values[tid],i);
            //printf("%s\n", secStr);
            // if(i == 0 &&tid < secSize)
            //     printf("mutationStr:%s value[%d]:%d\n", mutatedStr,tid,values[tid]);

            __syncthreads();
           
            reduce(values, secSize, PLUS);

            __syncthreads();
            // if(i == 0 &&tid < secSize)
            //     printf("value[%d]:%d, j:%d\n", tid,values[tid],j);
            // if (tid == 0)
            //     printf("values in Cuda:%d\n", values[0]);
            //no dependancy on this value to continue so we don't sync again
            reverse(differences, secSize);
            __syncthreads();

            scan(differences, secSize, PLUS);
            // if(i == 0&& j == 14)
            //     printf("diff[%d]:%d, j:%d\n", tid,differences[tid],j);
            // if(i == 0 &&tid < secSize)
            //     printf("diff[%d]:%d, j:%d\n", tid,differences[tid],j);
            //printf("diff[tid]:%d\n", differences[tid]);
            //scan has to finish before we reduce
            if(tid==0)
		    for(int k = 0; k<BLOCK_DIM*2; k++){
		    	indexes[k]=k;
		    }
            __syncthreads();

            reverse(differences, secSize);
            __syncthreads();
            

            reduceMax(differences, secSize,indexes);

            // if(i == 0 &&tid < secSize)
            //     printf("diff[%d]:%d, j:%d\n", tid,differences[tid],j);
            __syncthreads();
            //value[i] will hold the value that gave the difference
            if(tid == 0)
            {
                // if(i==0)
                //     printf("values[0]:%d, differences[0]:%d j:%d\n ", values[0], differences[0], j);
                localVal = values[0] + differences[0];
                if(localVal > value[i])
                {
                    value[i] = localVal;
                    offsets[i] = j;
                    //printf("%d %d \n",secSize,indexes[0]);
                    mutations[i]=indexes[0];
                   
                }
            }


        }
       
    }
     

    //__syncthreads();
   
}

// returns 0 if successful, otherwise returns 1
extern int computeOnGPU(char* mainStr, char** strArrs, int start, int arrSize, int evals[MAXEVAL][MAXEVAL])
{
	//printf("test?");
    char    *dev_mainStr;
    char    **dev_strArrs;
    int     *dev_offsets, *dev_mutations;
    int     newArrSize = arrSize - start;
    int     currSize, *dev_values;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int mainSize = strlen(mainStr);

    //allocate for each dev variable
    //dev_mutations and dev_offsets will be used to hold the results from each calculation
    cudaMalloc((void**)&dev_mainStr, (mainSize+1)*sizeof(char));//+1 for null terminated
    cudaMalloc((void**)&dev_strArrs, newArrSize*sizeof(char*));
    cudaMalloc((void**)&dev_offsets, newArrSize*sizeof(int));
    cudaMalloc((void**)&dev_mutations, newArrSize*sizeof(int));
    cudaMalloc((void**)&dev_values, newArrSize*sizeof(int));



    //copy the main str and the arr
    cudaMemcpy(dev_mainStr, mainStr, mainSize*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_strArrs, strArrs+start, newArrSize*sizeof(char*), cudaMemcpyHostToDevice);


    //loop trough the arr to copy each string in it individually
    //and in general set up the strings
    for (int i = 0; i < newArrSize; i++)
    {
        //i+start is the corresponding location of the string in the original arr since we didn't want to
        //copy it twice we just sent the entire address and copy accordingly here
        currSize = strlen(strArrs[i+start])+1;//+1 for null
        char* dev_currStr;
        cudaMalloc((void**)&dev_currStr, currSize*sizeof(char));
        cudaMemcpy(dev_currStr, strArrs[i+start], currSize*sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_strArrs + i, &dev_currStr, sizeof(char*), cudaMemcpyHostToDevice);
    }

    //set up evals, code taken from lecture
    int *dev_evals;
    cudaMalloc((void**)&dev_evals, MAXEVAL* MAXEVAL * sizeof(int));
    cudaMemcpy(dev_evals, evals, MAXEVAL* MAXEVAL * sizeof(int), cudaMemcpyHostToDevice);
   
               
           
   

    // // Allocate memory on GPU to copy the data from the host
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
    //     return 1;
    // }

    // // Copy data from host to the GPU memory
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
    //     return(1);
    // }


    // Launch the Kernel
    // Doing scan with more than one block is difficult and needs a lot of work
    // and using mechanisms not studied during the course
    // and so we decided to use 1 block to perform this task
    cudaLaunch<<<1, BLOCK_DIM>>>(dev_mainStr, mainSize, dev_strArrs, newArrSize, dev_evals, dev_values, dev_offsets, dev_mutations);
    cudaDeviceSynchronize();
    //dealWithMpi(); //kernel was launched and now we need to deal with the MPI part while cuda is running
    //CAN DO WORK INSTEAD OF dealWithMpi()


    //cudaDeviceSynchronize(); // for debugging
	cudaError_t cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
		printf("error %s\n", cudaGetErrorString(cudaStatus));
    int* values  = (int*)malloc(sizeof(int)*newArrSize);
    int* mutations  = (int*)malloc(sizeof(int)*newArrSize);
    int* offsets = (int*)malloc(sizeof(int)*newArrSize);


    cudaMemcpy(offsets, dev_offsets, newArrSize*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, dev_values, newArrSize*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mutations, dev_mutations,newArrSize*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i = 0; i < (newArrSize+1)/2; i++)
    {
        printf("Maximum alignment score between %s(main) and %s is %d, offset:%d, K:%d\n", mainStr,
                 strArrs[i], values[i],offsets[i],mutations[i]);
    }
    free(values);
    free(mutations);
    free(offsets);
    cudaFree(mutations);
    cudaFree(values);
    cudaFree(offsets);
    cudaFree(dev_evals);
    cudaFree(evals);
    cudaFree(dev_mainStr);
	for (int i = newArrSize - 1; i >= 0; i--)
	{
	    char* dev_currStr;
	    cudaMemcpy(&dev_currStr, dev_strArrs + i, sizeof(char*), cudaMemcpyDeviceToHost);
	    cudaFree(dev_currStr);
	}

	cudaFree(dev_strArrs); 
    cudaFree(dev_offsets);
    cudaFree(dev_mutations);
    cudaFree(dev_values);



    //copy the main str and the arr
    //cudaMemcpy(dev_mainStr, mainStr, mainSize*sizeof(char), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_strArrs, strArrs+start, newArrSize*sizeof(char*), cudaMemcpyHostToDevice);
   
/* note: next lines may be executed before the kernel is done */
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to launch incrementByOne kernel -  %s\n", cudaGetErrorString(err));
    //     return(1);
    // }

    // // Copy the  result from GPU to the host memory.
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
    //     return(1);
    // }

    // ***Free allocated memory on GPU***


    return 0;
}
