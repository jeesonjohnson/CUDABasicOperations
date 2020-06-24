#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 512
float gpuRecursiveReduce(float * d_in, int N);
__global__ void reduceKernel(float *d_out, float *d_in);
__global__ void summationKernel(float *d_out, float *d_in,const int N);
void cpuReduce(float * h_in,int N);
void gpuReduce(float * h_in,int N);
void cpuReduction(float *arr, int size, float gpuTime);
void printArray(float *arr, int sizeOfArray);


int main(){
    for (int k = 1; k < 35; ++k)
	{
        //Initialising the values
        int N = (1 << k);

        float * h_in = new float[N];
        for (int x=0; x<N;x++){
            h_in[x]=1.0f;
        }
        //Method for doing entier function on GPU
        gpuReduce(h_in,N);
        
        //Method for doing entier function of CPU
        cpuReduce(h_in,N);
    }
    return 0;
}

void cpuReduce(float * h_in,int N){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;

    bool debug =false;
    size_t size = N * sizeof(float);
    //How many blocks should be created, provided that each block has 1024 threads
    int GRID_SIZE = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // The above calculation makes it such that the correct amount of blocks are
    //Generated when we assume each block has 1024 threads in them.
    size_t size_o = GRID_SIZE * sizeof(float); ///.PROBALY CFHANE TO BLOCK_SIZE
    // Define the array for host device
    float h_out[GRID_SIZE];
    // Define the arrays for the cuda device(GPU)
    float *d_in;
    float *d_out;


    cudaMalloc((void **)&d_in, size);
    //The values are copied from the host array, to the device array.
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    //We allocate glbal memory to out array on the cuda device
    cudaMalloc((void **)&d_out, size_o);

    // thread size is a calculation of how many threads there will be in a given block
    if(debug){
        printf("INITITAL: Block dimensions(Threads in a block): %d\n", BLOCK_SIZE);
        printf("INITIAL: Grid dimensions(Blocks in a grid): %d\n", GRID_SIZE);
    }

    //Defining the size of the block
    dim3 blockDim(BLOCK_SIZE);
    //Defiing the size of the grid
    dim3 gridDim(GRID_SIZE);

    //Starting timing of cuda kernel
    cudaEventRecord(start);
    //Starting the kernel
    summationKernel<<<gridDim, blockDim>>>(d_out, d_in,N);
    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize(); //Ensure all the blocks are finished executing
    //Stop the event timing recording as well as present appropraite errors
    cudaEventRecord(stop);
    if(debug)
        printf("Run kernel: %s\n", cudaGetErrorString(err)); // FOr debugging

    //Moving the results form the GPU to cPU
    err = cudaMemcpy(h_out, d_out, size_o, cudaMemcpyDeviceToHost);
    if(debug)
        printf("Copy h_C off device: %s\n",cudaGetErrorString(err)); //For debugging

    //Print the elapsed time of the kernel
    cudaEventSynchronize(stop);
    float totalExecutionTime = 0;
    cudaEventElapsedTime(&totalExecutionTime, start, stop);
    // printf("\n\nElapsed GPU time was: %f milliseconds\n", milliseconds);


    if(debug)
        printArray(h_out,GRID_SIZE);

    cpuReduction(h_out, GRID_SIZE, totalExecutionTime);

}






void gpuReduce(float * h_in,int N){
    std::clock_t start;
	double duration;

    //Allocating appropaite space on GPU
    float * d_in;
    cudaMalloc(&d_in,sizeof(float)*N);
    cudaMemcpy(d_in,h_in,sizeof(float)*N,cudaMemcpyHostToDevice);
    start = std::clock();
    float total = gpuRecursiveReduce(d_in,N);
    duration = (std::clock() - start)*1000 / (double)CLOCKS_PER_SEC;
    printf("\n\nGPU: Total For %d is %f in a duration of %f ms\n",N,total,duration);
    // std::cout << "CPU time: " << duration << " ms" << std::endl;

}

float gpuRecursiveReduce(float * d_in, int N){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalSum=0.0f;

    int grid_size=((N + BLOCK_SIZE - 1) / BLOCK_SIZE); 


    float * d_blocks;
    cudaMalloc(&d_blocks,sizeof(int)*grid_size);
    cudaMemset(d_blocks, 0, sizeof(float) * grid_size);
    
    float temp=0;
    cudaEventRecord(start);
    reduceKernel<<<grid_size,BLOCK_SIZE>>>(d_blocks,d_in);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&temp, start, stop);
    printf("T:%f",temp);

    if(grid_size<=BLOCK_SIZE){
        float* d_total;
        cudaMalloc(&d_total,sizeof(float));
        cudaMemset(&d_total,0,sizeof(float));
        cudaEventRecord(start);
        reduceKernel<<<1,BLOCK_SIZE>>>(d_total,d_blocks);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&temp, start, stop);
        printf("T:%f",temp);
        cudaMemcpy(&totalSum, d_total, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_total);
    }	
    else{
		float * d_in_block_sums;
		cudaMalloc(&d_in_block_sums, sizeof(float) * grid_size);
		cudaMemcpy(d_in_block_sums, d_blocks, sizeof(float) * grid_size, cudaMemcpyDeviceToDevice);
		totalSum = gpuRecursiveReduce(d_in_block_sums, grid_size);
		cudaFree(d_in_block_sums);
	}

	cudaFree(d_blocks);
	return totalSum;


}


__global__ void reduceKernel(float *d_out, float *d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x; // ID relative to whole array 
    int tid = threadIdx.x;             // Local ID within the current block
    __shared__ float temp[BLOCK_SIZE];
    temp[tid] = d_in[myId];
    __syncthreads();
    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s >= 1; s >>= 1)
    {
        if (tid < s)
        {
            temp[tid] += temp[tid + s];
        }
        __syncthreads(); // make sure all adds at one stage are done !
    }
    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
    {
        d_out[blockIdx.x] = temp[tid];
    }
}



__global__ void summationKernel(float *d_out, float *d_in,const int N)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x; // ID relative to whole array 
    int tid = threadIdx.x;             // Local ID within the current block
    __shared__ float temp[BLOCK_SIZE];
    
    // if(myId<=N){
        temp[tid] = d_in[myId];
    // }
    //else{
    //     temp[tid] =0;
    // }
        __syncthreads();
        // do reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s >= 1; s >>= 1)
        {
            if (tid < s && myId<=N)
            {
                temp[tid] += temp[tid + s];
            }
            __syncthreads(); // make sure all adds at one stage are done !
        }

    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
    {
        d_out[blockIdx.x] = temp[tid];
    }
}





void cpuReduction(float *arr, int size, float gpuTime)
{

    float final_reduction = 0.0f;
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

    printf("CPU: sumation %fin a time %f milliseconds\n\n", final_reduction,time_taken + double(gpuTime));
}

void printArray(float *arr, int sizeOfArray)
{
        printf("Printing %d values \n", sizeOfArray);
        for (int i = 0; i < sizeOfArray; i++)
        {
            printf("%f|", arr[i]);
            if((i+1)%BLOCK_SIZE==0){
                printf("\t");
            }
        }
        printf("\n Fin printing");
    
    printf("\n");
}

