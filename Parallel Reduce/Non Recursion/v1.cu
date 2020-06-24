#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <time.h>
using namespace std;

#define BLOCK_SIZE 32

void cpuReduction(unsigned int *arr, int size, float gpuTime);
void printArray(unsigned int *arr, int sizeOfArray);
__global__ void reduceKernel(unsigned int * d_in,unsigned int* d_out,const unsigned int N);
void mainReductionMethod(unsigned int N, bool debug);

int main(void){
    for(int x=1;x<33;x++){
        mainReductionMethod(1<<x,false);
    }
    
    return 0;
}


void mainReductionMethod(unsigned int N, bool debug){


    // Developed appropraite timing functionality
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalExicutionTime=0.0f;

    //Initilaise the value of the array
    unsigned int *input;
    cudaMallocManaged(&input, N * sizeof(unsigned int));
    for(unsigned int x=0;x<N;x++){
        input[x]=1;
    }
    
    //Initialise the output array values
    unsigned int *output;
    cudaMallocManaged(&output, N * sizeof(unsigned int));
    cudaMemset(output, 0, sizeof(unsigned int) * N); // Initialise the values in the array to 0
    
    //Calling the reduce kernel once
    unsigned int grid_size =  (N+BLOCK_SIZE -1)/BLOCK_SIZE;
    cudaEventRecord(start);
    reduceKernel<<<grid_size,BLOCK_SIZE>>>(input,output,N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&totalExicutionTime, start, stop);
    
    //Insert here method for CPU final reduce 
    cpuReduction(output,grid_size,totalExicutionTime);


    //Method for calling GPU reduction, the set of values reduces every loop, till the values can fit in 
    //a singular block, of size BLOCK_SIZE
    float temp=0.0f;
    while(grid_size>1){
        cudaEventRecord(start);
        reduceKernel<<<grid_size,BLOCK_SIZE>>>(output,output,grid_size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventElapsedTime(&temp, start, stop);
        totalExicutionTime+=temp;
        grid_size =  (grid_size+BLOCK_SIZE -1)/BLOCK_SIZE;

    }


    //Appropraite debugging method
    if(debug){
        printArray(output,N);
    }


    // printf("\nGPU Total: %d Time:%f \n\n",output[0],totalExicutionTime);
    // printf("%f\n",totalExicutionTime);
    cudaFree(input);
    cudaFree(output);
    
}



void cpuReduction(unsigned int *arr, int size, float gpuTime)
{
    unsigned int final_reduction = 0;
    clock_t start = clock();
    clock_t tStart = clock();
    /* Do your stuff here */

    for (int i = 0; i < size; i++)
    {
        final_reduction += arr[i];
    }
    // Recordingend time.
    //Calculating the actual total difference
    clock_t stop = clock();
    double time_taken = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    // printf("CPU: Summation %d ", final_reduction);
    // printf("Time %f ms", time_taken + double(gpuTime));
    printf("%f\n",time_taken + double(gpuTime));

    // printf("CPU: Summation %f , Time: %f ms\n", final_reduction,time_taken + double(gpuTime));
}

void printArray(unsigned int *arr, int sizeOfArray)
{
        printf("Printing %d values \n", sizeOfArray);
        for (int i = 0; i < sizeOfArray; i++)
        {
            printf("%d|", arr[i]);
            if((i+1)%BLOCK_SIZE==0){
                printf("\t");
            }
        }
        printf("\n Fin printing");
    
    printf("\n");
}



__global__ void reduceKernel(unsigned int * d_in,unsigned int* d_out,const unsigned int N) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x; // ID relative to whole array
    int tid = threadIdx.x; // Local ID within the current block
    __shared__ unsigned int temp[BLOCK_SIZE];
    temp[tid] = d_in[myId];
    __syncthreads();
    // do reduction in shared memory
    for (unsigned int s = blockDim.x/2; s >= 1; s >>= 1)
    {
        if (tid < s && myId<N)
        {
        temp[tid] += temp[tid + s];
        }
    __syncthreads(); // make sure all adds at one stage are done!
    }
    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
    {
        d_out[blockIdx.x] = temp[tid];
    }
}