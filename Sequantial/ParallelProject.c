#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "cDefenitions.h"

//nvcc -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o || For compiling
//
//gcc -o mpiCudaOpenMP  main.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart~ || for compiling after making o files

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

//cuda
extern int computeOnGPU(char* mainStr, char** strArrs, int start, int arrSize, int evals[MAXEVAL][MAXEVAL]);


int nprocs, rank;
int main(int argc, char **argv)
{
    init(argc, argv);
    printEvals();
    work();
    return 1;
}

void work()
{
    printf("currently working");
    int offset, mutation, foundEval;
    char tempStr[S_STR_LEN];
    //initmaybe?
    //printEvals(Evaluations);
    //convert the str to upper for convinience 
    toUpperString(mainStr);

    for (int i = 0; i < strArrSize; i++)
    {
        //printf("i:%d\n", i);
        //convert the str to upper for convinience 
        //printf("%s\n", strArrs[i]);
        toUpperString(strArrs[i]);
        strcpy(tempStr, strArrs[i]);
        foundEval = evaluateStrings(mainStr, tempStr, &offset, &mutation);
        printf("Maximum alignment score between %s(main) and %s is %d, offset:%d, K:%d\n", mainStr,
        strArrs[i], foundEval,offset, mutation);
    }
    
}

void init(int argc, char** argv)
{   
    printf("%d\n", argc);
    if (argc == 1) // default
        initDefaultEvals();
    else
        readWeights(argv[1]); 
    //printEvals();

    readStrings();
}


void initDefaultEvals()
{
	//printf("BBBB%d\n",rank);
    for (int i = 0; i < MAXEVAL; i++)
        for (int j = 0; j < MAXEVAL; j++)
            Evaluations[i][j] = i == j ? 1 : 0;
}

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
    int offsetItr;
    int maxValue = MIN_VALUE;
    int currentValue = 0;
    int secStrLen = strlen(secStr);//preferable to not calculate it couple of times
    int difference;

    //if main str len is 3 and secondary is 3 then we need 1 iteration
    //so for each offset we will need +1 iterations than size-size
    offsetItr = strlen(mainStr) - secStrLen + 1;
    for (int i = 0; i < offsetItr; i++)
    {
    	//printf("%d, %da\n",i,rank);
        if(i>0)
            MSreset(secStr);
        //printf("%s\n", secStr);
        //evaluate for no mutation and we will calculate the new mutation depending on the changed char
        currentValue = getValue(mainStr, secStr, i); 
        if(currentValue > maxValue)
            {
                maxValue = currentValue;
                *offset = i;
                *K = secStrLen;
            }
	    //printf("current value is %d\n",currentValue);
        //mutation 0 already calculated "default"
        //<= since there is 1 more iteration for "default"
        for (int j = 0; j < secStrLen; j++)
        {   
            //this also mutates the string
            difference = getDifference(mainStr, secStr, secStrLen, i+1, j+1);
            //printf("diff:%d\n", difference);
            //for j = 0 diff will be 0 and so we get the original current value
            //it's better than adding an if statement since you only do this once per i iteration

            
            currentValue += difference;

            if(currentValue > maxValue)
            {
                maxValue = currentValue;
                *offset = i;
                *K = secStrLen-(j+1);
            }
                
        }
    }
    
    return maxValue;
}

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

void readStrings()
{
	int mainStringLength;
	int *strLens;
    char tempStr[M_STR_LEN];
    scanf("%s\n", tempStr);
    mainStr = strdup(tempStr);
    mainStringLength = strlen(mainStr);
    //printf("%s\n", mainStr);
    scanf("%d\n", &strArrSize);
    strLens = (int*)malloc(strArrSize*sizeof(int));
    strArrs = (char**)malloc(sizeof(char*)*strArrSize);
    for (int i = 0; i < strArrSize; i++)
    {
    char currentStr[S_STR_LEN];
    scanf("%s\n", currentStr);   
    strArrs[i] = strdup(currentStr);
    strLens[i] = strlen(strArrs[i]);
    //printf("%s\n", strArrs[i]);
    }
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
