#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "mpi.h"
#include "cDefenitions.h"
#include <time.h>

//nvcc -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o || For compiling
//
//gcc -o mpiCudaOpenMP  main.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart~ || for compiling after making o files

struct maxUnit{
    int value;
    int offset;
    int K;
};

//code taken from our exersize 2 and changed to fit
#pragma omp declare reduction (maxVal: struct maxUnit :\
	                    omp_out = (omp_out.value > omp_in.value ? omp_out : omp_in)) /* combiner */ \
						initializer(omp_priv = {-1000000, -1, -1})//doesn't let use MIN_VALUE so I entered it as a const


void    init(int argc, char** argv);
void    printEvals();
void    initDefaultEvals();
void    work();
char    *mainStr;
char    **strArrs;
int     strArrSize;
int     Evaluations[MAXEVAL][MAXEVAL];
void    readWeights(char* fileName);
void    readStrings();
void    skip_white_space();


int     evaluateStrings(char* mainStr, char* secStr, int* offset, int* K);
void    MSreset(char* secStr);
void    MS(char* secStr, int K, int size); 
int     getValue(char* mainStr, char* secStr, int offset);
void    toUpperString(char* str);
int     getDifference(char* mainStr, char* secStr, int secSize, int offset, int K);
void    create_maxunit_struct(MPI_Datatype *mpi_maxunit);
void    maxUnit_max(void* in, void* inout, int* len, MPI_Datatype* datatype); 

//externing for cuda
void dealWithMpi();


//cuda
extern int computeOnGPU(char* mainStr, char** strArrs, int start, int arrSize, int evals[MAXEVAL][MAXEVAL]);


int nprocs, rank;
int main(int argc, char **argv)
{	
	clock_t start, end;
	double cpu_time_used;
	//mpi initializing
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	init(argc, argv);
	start = clock();
	if(rank==0){
		//printf("test1\n");
		computeOnGPU(mainStr, strArrs, 0, strArrSize, Evaluations);
		//printf("test2\n");
		}
	else{
		work();
		for(int i = 0; i<strArrSize;i++)
			free(strArrs[i]);
		free(mainStr);
		end = clock();
		
		cpu_time_used = ((double) (end - start))/CLOCKS_PER_SEC;
		//printf("time taken: %f",cpu_time_used);
	
	}
	if(rank==0){
		for(int i = 0; i<strArrSize;i++)
			free(strArrs[i]);
		free(mainStr);
		//MPI_Finalize();
		end = clock();
		
		cpu_time_used = ((double) (end - start))/CLOCKS_PER_SEC;
		//printf("time taken: %f",cpu_time_used);
	}
	MPI_Finalize();
    return 1;
}

void work()
{

    int offset, mutation, foundEval;
    char tempStr[S_STR_LEN];
    
    //divide the work by strings between differenct processes. wanted by offset, but that requires         reduce which makes it slower.
    toUpperString(mainStr);
	int range_size = strArrSize-(strArrSize+1)/2;
	int process_load = range_size/(nprocs-1);
	int leftovers = range_size%(nprocs-1);
	int start = (strArrSize+1)/2+(rank-1)*process_load;
	int end = start+process_load;
	if(rank-1<leftovers){
		start+=rank-1;
		end+=rank;
	} 
	else{
		start +=leftovers;
		end+=leftovers;
	}
	if(end > strArrSize)
		end = strArrSize;
    for (int i = start; i < end; i++)
    {
    //printf("aaa\n");
        //convert the str to upper for convinience 
        //printf("%s\n", strArrs[i]);
        toUpperString(strArrs[i]);
        strcpy(tempStr, strArrs[i]);
        foundEval = evaluateStrings(mainStr, tempStr, &offset, &mutation);
        //printf("value:%d, offset:%d, K:%d, i:%d\n", bufferUnit.value, bufferUnit.offset, bufferUnit.K, i);
        printf("Maximum alignment score between %s(main) and %s is %d, offset:%d, K:%d\n", mainStr,
        	strArrs[i], foundEval,offset, mutation);
	//printf("bbb\n");
    }
    //printf("cccc\n");
    
}

void init(int argc, char** argv)
{   
    //init, either with grading scheme or without
    if (argc == 1) // default
        initDefaultEvals();
    else
        readWeights(argv[1]); 
    //printEvals();

    readStrings();
}


void initDefaultEvals()
{
	//default value is an I matrix
    for (int i = 0; i < MAXEVAL; i++)
        for (int j = 0; j < MAXEVAL; j++)
            Evaluations[i][j] = i == j ? 1 : 0;
}
//for debug
void printEvals()
{
    int letter;
    printf("   ");
    for (letter = 'A'; letter <= 'Z'; letter++)
        printf("%c  ", (char)letter);
    printf("\n");

    letter = 'A';
    for (int i = 0; i < MAXEVAL; i++)
    {
        printf("%c  ", (char)letter++);
        for (int j = 0; j < MAXEVAL; j++)
            printf("%d ", Evaluations[i][j]);
        printf("\n");
    }
}



int evaluateStrings(char* mainStr, char* secStr, int* offset, int* K)
{

//evaluates the string, by checking difference in mutation on all offsets. done with an OMP for.
    int offsetItr;
    struct maxUnit maxStruct = {MIN_VALUE, -1, -1};
    int currentValue = 0;
    int secStrLen = strlen(secStr);//preferable to not calculate it couple of times
    char* localStr;
    int difference;
	int citr = 0;
    //if main str len is 3 and secondary is 3 then we need 1 iteration
    //so for each offset we will need +1 iterations than size-size
    offsetItr = strlen(mainStr) - secStrLen + 1;

#pragma omp parallel for default(none) reduction(maxVal: maxStruct) shared (secStrLen, mainStr, offsetItr) firstprivate(secStr,citr) private (currentValue,difference,localStr)
    for (int i = 0; i < offsetItr; i++)
    {
    	if(citr==0){
	    	localStr = (char*) malloc(secStrLen*sizeof(char));
	    	strcpy(localStr,secStr);
	    	citr++;
    }
        //evaluate for no mutation and we will calculate the new mutation depending on the changed char
	currentValue = getValue(mainStr, localStr, i); 
        //printf("currentValue:%d\n", currentValue);
        if(currentValue > maxStruct.value)
            {
                //struct maxUnit tempUnit = {currentValue, i, secStrLen};
                //maxStruct = tempUnit;
                 maxStruct.value = currentValue;
                 maxStruct.offset = i;
                 maxStruct.K = secStrLen;
            }
	//printf("current value is %d\n",currentValue);
        //mutation 0 already calculated "default"
        //<= since there is 1 more iteration for "default"
        for (int j = 0; j < secStrLen; j++)
        {   
            //this also mutates the string
            difference = getDifference(mainStr, localStr, secStrLen, i+1, j+1);
            //printf("diff:%d\n", difference);
            //for j = 0 diff will be 0 and so we get the original current value
            //it's better than adding an if statement since you only do this once per i iteration

            currentValue += difference;
            if(currentValue > maxStruct.value)
            {
                struct maxUnit tempUnit = {currentValue, i, secStrLen - (j+1)};
                //maxStruct = tempUnit;
                 maxStruct.value = currentValue;
                 maxStruct.offset = i;
                 maxStruct.K = secStrLen-(j+1);
            }
                
        }
        MSreset(localStr);
        //printf("AAAA %s\n\n\n\n", secStr);
    }


    //finalizes the findings
    *offset = maxStruct.offset;
    *K = maxStruct.K;
    return maxStruct.value;
}
//resets mutations.
void MSreset(char* secStr)
{   
    int count = 0;
    while(secStr[count])
    {
        if(secStr[count] != 'A')
            secStr[count++] -= 1;
        else
            secStr[count++] = 'Z';
    }
        
    
}

//We don't do MS as defined because it is better to look at it as indexes that you change
//in the end as long as we calculate all mutations, the order doesn't matter
void MS(char* secStr, int K, int size)
{

    if(K != 0)
        if(secStr[size - K] != 'Z')
            secStr[size - K] += 1;
        else
            secStr[size - K ] = 'A';
}
//get value between 2 chars.
int getValue(char* mainStr, char* secStr, int offset)
{
    int count = 0;
    int res = 0;
    int mIndex, sIndex;
    while(secStr[count])
    {
        //printf("%s, %s\n", mainStr, secStr);
        mIndex = (int)(mainStr[count + offset] - 'A');
        sIndex = (int)(secStr[count] - 'A');
        //printf("Eval:%d, mIndex:%c, sIndex:%c\n", offset, mainStr[count + offset], secStr[count]);
        res += Evaluations[mIndex][sIndex];
        count ++;
    }

    return res;
}
void toUpperString(char* mainStr)
{
    while(*mainStr)
    {
        toupper(*mainStr);
        mainStr++;
    }
        
}
//gets difference between mutated and unmutated chars.
int getDifference(char* mainStr, char* secStr, int secSize ,int offset, int K)
{
    int mIndex, sIndex;
    int mutatedSIndex;

    mIndex = (int)mainStr[secSize - K + offset-1] - 'A';
    sIndex = (int)secStr[secSize - K] - 'A';
    //printf("%c, %d, %c,%d\n",mainStr[secSize - K + offset-1],mIndex,secStr[secSize - K],sIndex);
    //printf("before MS %s\n",secStr);
    MS(secStr, K, secSize);
    //printf("before MS %s\n",secStr);
    //printf("secStr:%s\n", secStr);
    mutatedSIndex = (int)secStr[secSize - K] - 'A';

    return Evaluations[mIndex][mutatedSIndex] - Evaluations[mIndex][sIndex];
}

//code for read Weights is taken from exer 2 and is changed to fit our task
//changed heavily because we use a fixed array[][] of chars and not a pointer of pointers
void readWeights(char* fileName) 
{
	//printf("CCCC%d\n",rank);
    FILE *fp = fopen(fileName, "r"); 
    if(!fp)
    {
        printf("failed to open eval file\n");
        exit(0);
    }

    int c, flag = 0;
    unsigned int w;
    int count_w = 0; // number of entries read in so far

    /* First number in the input is the number of vertices. Use it to initialize 'NV' */
    int i, j;
    if(rank==0){
	    for (int i = 0; i < MAXEVAL; i++)
	    {
		for (int j = 0; j < MAXEVAL; j++)
		{
		    if (!feof(fp))
		        fscanf(fp, "%d", &Evaluations[i][j]);
		    else
		        Evaluations[i][j] = 0; 
		}
	    }
	    fclose(fp);
    }
    MPI_Bcast(Evaluations, MAXEVAL*MAXEVAL, MPI_INT, 0, MPI_COMM_WORLD);
    
}
//reads strings from input and sends them over to other processes.
void readStrings()
{
	int mainStringLength;
	int *strLens;
	if(rank==0){
	    char tempStr[M_STR_LEN];
	    scanf("%s\n", tempStr);
	    mainStr = strdup(tempStr);
	    mainStringLength = strlen(mainStr)+1;
	    //printf("%s\n", mainStr);
	    scanf("%d\n", &strArrSize);
	    strLens = (int*)malloc(strArrSize*sizeof(int));
	    strArrs = (char**)malloc(sizeof(char*)*strArrSize);
	    for (int i = 0; i < strArrSize; i++)
	    {
		char currentStr[S_STR_LEN];
		scanf("%s\n", currentStr);   
		strArrs[i] = strdup(currentStr);
		strLens[i] = strlen(strArrs[i])+1;
		//printf("%s\n", strArrs[i]);
	    }
    }
    MPI_Bcast(&mainStringLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //printf("%d %d\n",mainStringLength,rank);
    if(rank !=0)
    	mainStr = (char*)malloc(mainStringLength*sizeof(char));
    MPI_Bcast(mainStr, mainStringLength, MPI_CHAR, 0, MPI_COMM_WORLD);
    //printf("%s %d\n",mainStr,rank);
    MPI_Bcast(&strArrSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank!=0){
    	strLens = (int*)malloc(strArrSize*sizeof(int));
    	strArrs = (char**)malloc(sizeof(char*)*strArrSize);
    }
    MPI_Bcast(strLens, strArrSize, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = 0; i<strArrSize;i++){
    		
	    if(rank!=0){
	    	//printf("aa %d %d\n",i, strLens[i]);
	    	strArrs[i] = (char*)malloc(sizeof(char)*(strLens[i]));
	    	//printf("bb %d %d\n",i, strLens[i]);
	    }
	    //printf("cc %d %d\n",i, strLens[i]);
	    //printf("test %s \n",strArrs[i]);
	    MPI_Bcast(strArrs[i], strLens[i], MPI_CHAR, 0, MPI_COMM_WORLD);
	    //printf("dd %d %d\n",i, strLens[i]);
	    //printf("%d %d %d\n",strLens[i],i,rank);
	    //printf("%s %d\n",strArrs[i],rank);
    }
    free(strLens);
    
}

int lineno = 1; // current input line number

void skip_white_space() {
   int c;
   while(1) {
       if ((c = getchar()) == '\n')
           lineno++;
       else if (isspace(c))
           continue;
       else if (c == EOF)
           break;
       else {
         ungetc(c, stdin); // push non space character back onto input stream
         break;
       }
   }
}

//taken from exer 2, ended up not used
void create_maxunit_struct(MPI_Datatype *mpi_maxunit)
{
    int block_lengths[3] = {1, 1, 1};
    MPI_Aint displacements[4];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};

    // this also works and is probably more portable:
    struct maxUnit maxDummy;
    MPI_Aint base_address;
    MPI_Get_address(&maxDummy, &base_address);
    MPI_Get_address(&maxDummy.value, &displacements[0]);
    MPI_Get_address(&maxDummy.offset, &displacements[1]);
    MPI_Get_address(&maxDummy.K, &displacements[2]);


    for (int i = 0; i < 3; i++)
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);
    
    MPI_Type_create_struct(3, block_lengths, displacements, types, mpi_maxunit);
    MPI_Type_commit(mpi_maxunit);
}

void maxUnit_max(void* in, void* inout, int* len, MPI_Datatype* datatype) {
    struct maxUnit* in_data = (struct maxUnit*)in;
    struct maxUnit* inout_data = (struct maxUnit*)inout;
    //iterates over each value from each process
    for (int i = 0; i < *len; i++) {
        if (in_data[i].value > inout_data[i].value) {
            inout_data[i] = in_data[i];
        }
    }
}
//not used anymore, left for context (and legacy!)
void dealWithMpi()
{
    //creating new type
    MPI_Datatype mpi_maxunit;
    create_maxunit_struct(&mpi_maxunit);
    struct maxUnit bufferUnit = {MIN_VALUE, -1, -1};
    struct maxUnit resUnit;

    //creating new OP
    MPI_Op maxUnit_max_op;
    MPI_Op_create((MPI_User_function *)maxUnit_max, 0, &maxUnit_max_op);

    //lets other processes continue working
    for (int i = (strArrSize+1)/2; i < strArrSize; i++)
    {
        MPI_Reduce(&bufferUnit, &resUnit, 1, mpi_maxunit, maxUnit_max_op, 1 ,MPI_COMM_WORLD);
    }
}
