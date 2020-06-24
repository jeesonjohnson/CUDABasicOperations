#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#define BLOCK_SIZE 64
// This sets the number of bins in the histogram
#define BIN_COUNT 8

__global__ void simple_histogram(unsigned int *d_bins, const unsigned int *d_in, const unsigned int bin_count,unsigned int N);
__global__ void shared_memory_histogram(unsigned int *d_bins, const unsigned int *d_in, const unsigned int bin_count,unsigned int N);
void sharedMemorykernelCallMethod(unsigned int N,bool debug);
void nonSharedkernelCallMethod(unsigned int N,bool debug);

int main(void)
{    
    //Method for passing in values of N, that are multiples of 2^x
    for (int k = 1; k < 32; ++k)
	{
        //Initialising the values
        unsigned int N = (1 << k);
        nonSharedkernelCallMethod(N,false);
        printf("\n");
        sharedMemorykernelCallMethod(N,false);
    }
 
    return 0;
}


void nonSharedkernelCallMethod(unsigned int N,bool debug){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int *d_bins; // This is the array that will contain the histogram
    unsigned int *d_in;   // This is the array that will contain the input data
    // We will use the CUDA unified memory model to ensure data is transferred between host and device
    cudaMallocManaged(&d_bins, BIN_COUNT * sizeof(unsigned int));
    cudaMallocManaged(&d_in, N * sizeof(unsigned int));

    //We firstly generate input data that can be used
    for (unsigned int i = 0; i < N; i++)
    {
        d_in[i] = i;
    }
    // We initialise the size of each bin to equal 0 initially
    for (unsigned int i = 0; i < BIN_COUNT; i++)
    {
        d_bins[i] = 0;
    }

    //THIS ALLOWS US TO FUNCTION WITH VALUES OF N THAT ARE NOT MULTIPLES OF BLOCK_SIZE
    unsigned int grid_size = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEventRecord(start);
    simple_histogram<<<grid_size, BLOCK_SIZE>>>(d_bins, d_in, BIN_COUNT,N);
    // wait for Device to finish before accessing data on the host
    cudaDeviceSynchronize();
    // Recording the execution time below
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalExecutionTime = 0;
    cudaEventElapsedTime(&totalExecutionTime, start, stop);

    // Now we can print out the resulting histogram
    unsigned int total = 0; //For debugging
    for (unsigned int i = 0; i < BIN_COUNT; i++)
    {
        if(debug)
            printf("Bin no. %d: Count = %d\n", i, d_bins[i]);
        total += d_bins[i];
    }
    //Do appropriate error handling if required.
    if(total!=N){
        printf("Error for value %d\n",N);
    }
    //printf("NonShared: Elements N: %d, histogram total: %d, excution time:%f", N, total, totalExecutionTime);
    printf("%f|",totalExecutionTime);
    if (debug)
        printf("\n Blocks: %d, Threads %d , Histogram total %d\n", grid_size, BLOCK_SIZE, total);

}

void sharedMemorykernelCallMethod(unsigned int N,bool debug){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int *d_bins; // This is the array that will contain the histogram
    unsigned int *d_in;   // This is the array that will contain the input data
    // We will use the CUDA unified memory model to ensure data is transferred between host and device
    cudaMallocManaged(&d_bins, BIN_COUNT * sizeof(unsigned int));
    cudaMallocManaged(&d_in, N * sizeof(unsigned int));

    //We firstly generate input data that can be used
    for (unsigned int i = 0; i < N; i++)
    {
        d_in[i] = i;
    }
    // We initialise the size of each bin to equal 0 initially
    for (unsigned int i = 0; i < BIN_COUNT; i++)
    {
        d_bins[i] = 0;
    }

    //THIS ALLOWS US TO FUNCTION WITH VALUES OF N THAT ARE NOT MULTIPLES OF BLOCK_SIZE
    unsigned int grid_size = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEventRecord(start);
    shared_memory_histogram<<<grid_size, BLOCK_SIZE>>>(d_bins, d_in, BIN_COUNT,N);
    // wait for Device to finish before accessing data on the host
    cudaDeviceSynchronize();
    // Recording the execution time below
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float totalExecutionTime = 0;
    cudaEventElapsedTime(&totalExecutionTime, start, stop);

    // Now we can print out the resulting histogram
    unsigned int total = 0; //For debugging
    for (unsigned int i = 0; i < BIN_COUNT; i++)
    {
        if(debug)
            printf("Bin no. %d: Count = %d\n", i, d_bins[i]);
        total += d_bins[i];
    }
    //Do appropriate error handling if required.
    if(total!=N){
        printf("Error for value %d\n",N);
    }
    // printf("SharedMemory: Elements N: %d, histogram total: %d, excution time:%f\n\n", N, total, totalExecutionTime);
    printf("%f|",totalExecutionTime);
    if (debug)
        printf("\n Blocks: %d, Threads %d , Histogram total %d\n", grid_size, BLOCK_SIZE, total);

}






__global__ void simple_histogram(unsigned int *d_bins, const unsigned int *d_in, const unsigned int bin_count,unsigned int N)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    //The below method is just one of MANY ways to allocate elements to bins
    // Below is purley just an illustation
    int myBin = myItem % bin_count;
    //The below condition with additonal blocks sizes above allow the method
    // to function with any number of elements and NOT just multiples of BLOCK_SIZE
    if (myId < N)
        atomicAdd(&(d_bins[myBin]), 1);
}


__global__ void shared_memory_histogram(unsigned int *d_bins, const unsigned int *d_in, const unsigned int bin_count,unsigned int N)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    //Assinges approrpatie memory methods
    __shared__ unsigned int bins[BIN_COUNT];
    unsigned int myItem = d_in[myId];
    //Each shared memory location is passed in a value that can be used.
    unsigned int itemBinId = myItem % bin_count;
    bins[itemBinId] = 0;
    __syncthreads();
    //Method for adding up the sum in bin values
    if (myId < N){
        atomicAdd(&(bins[itemBinId]), 1);
    }
    __syncthreads();

    //If the thread id of a given block is 0, then it adds all the item values to the output array
    if(tid==0){
        for(unsigned int x=0;x<BIN_COUNT;x++){
            atomicAdd(&(d_bins[x]), bins[itemBinId]);
        }
    }

}