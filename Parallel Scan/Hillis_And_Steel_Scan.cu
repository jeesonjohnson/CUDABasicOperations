#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#define BLOCK_SIZE 32

void gpuScan(int N);
void printArray(float *arr, int sizeOfArray,bool printStament);
__global__ void scanKernel(float *idata,int n,int startIndex=0);
__global__ void findMaxValueArray(float* scanData,float* outputData);
__global__ void finalScanKernel(float* initialData,float* maxValues);



int main(void)
{

    printf("Starting the scan kernel");
    for (int i = 1; i < 29; i++)
    {
        // printf("%d \n",1<<i);
        gpuScan(1<<i);
        
    }
    return 0;
}


void gpuScan(int N){
    //Methods for debugging
    bool debug = false;
    bool debugStep1=false;
    bool debugStep2=false;
    bool debugStep3=false;
    bool debugStep3Indepth=false;
    bool debugStep4=false;

    float *idata;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;
    // Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&idata, N * sizeof(float));
    // Initialise the input data on the host
    // Making it easy to test the result
    for (int i = 0; i < N; i++)
    {
        idata[i] = 1.0f;
    }


    // STEP 1 : Generate an intial scan 
    int grid_size = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEventRecord(start);
    scanKernel<<<grid_size, BLOCK_SIZE>>>(idata,BLOCK_SIZE);
    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    cudaEventRecord(stop);
    printf("Kernel Errors: %s\n", cudaGetErrorString(err));
    float totalExecutionTime = 0;
    cudaEventElapsedTime(&totalExecutionTime, start, stop);

    // Now output the resulting array:
    if(debug)
        printf("\nSTEP 1: Generating initial scan values, grid dim: %d, block dim: %d \n",grid_size,BLOCK_SIZE);
    printArray(idata,N,debugStep1);


    // Step 2: Generate scan of the max values of the output, i.e the last element in each block.
    // Then apple a scan within the kernel call
    float *scanMax;
    cudaMallocManaged(&scanMax, grid_size * sizeof(float));
    cudaEventRecord(start);
    findMaxValueArray<<<grid_size, BLOCK_SIZE>>>(idata,scanMax);
    err = cudaDeviceSynchronize();
    cudaEventRecord(stop);
    float temp=0;
    cudaEventElapsedTime(&temp, start, stop);
    totalExecutionTime+=temp;
    //Print output values for debugbing
    if(debug)
        printf("\n STEP 2: Finding max scan values, grid dim: %d, block dim: %d \n",grid_size,BLOCK_SIZE);
    if(debugStep2){

        for (int i = 0; i < grid_size; i++)
        {
            printf("%f|", scanMax[i]);
            if((i+1)%BLOCK_SIZE==0){
                printf("\t");
            }
        }
    }

    // Step3 : Apply scan again to max values from previous scan, this needs to be done by calling the scan kernel
    //multiple times depending on howmany elements fit into a block, since you need cross block communication to ensure
    //overall scan method can function.
    //The method works by striding over a fixed block of values, and adding the last element of a previus block to the current blocl. Eventually gettting a reusult.
    int tempBlockSize=512;
    int scanGridSize=((grid_size + tempBlockSize - 1) / tempBlockSize);
    int startIndex=0;
    int endIndex=0;
    while(scanGridSize>0){
        //Initialising the scan start values.
        scanGridSize=scanGridSize-1;
        endIndex=grid_size-(tempBlockSize*(scanGridSize));

        if(debugStep3Indepth)
            printf("\nscanGridsize %d,startIndex:%d,endIndex:%d\n",scanGridSize,grid_size,endIndex);
        
        //Calling kernel, as well as intialising the timing for the method.
        cudaEventRecord(start);
        scanKernel<<<1, tempBlockSize>>>(scanMax,grid_size,startIndex);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        //Adding timing to overall timing
        cudaEventElapsedTime(&temp, start, stop);
        totalExecutionTime+=temp;
        
        //Setting the kernel start point for next itteration.
        startIndex=endIndex;

        // Only up untill the last block should the previous blocks last value should be added to next blocks 
        // starting value!
        if(scanGridSize>0){
            scanMax[endIndex]+=scanMax[endIndex-1];
            if(debugStep3Indepth)
                printf("\nscanMax[endIndex+1]:%d ,scanMax[endIndex] %d\n",scanMax[endIndex+1],scanMax[endIndex]);
            
        }
    }
    if(debug)
        printf("\n\n STEP 3: Apply scan again to max values from previous scan, grid dim: %d, block dim: %d , meaning max %d elements \n",scanGridSize,BLOCK_SIZE, scanGridSize*BLOCK_SIZE);
    printArray(scanMax,grid_size,debugStep3);


    // Step 4: The step now is to ensure that you can apply the summation to the remaining values
    // Such that values in scanMax get added to the original array
    cudaEventRecord(start);
    finalScanKernel<<<grid_size, BLOCK_SIZE>>>(idata,scanMax);
    err = cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventElapsedTime(&temp, start, stop);
    totalExecutionTime+=temp;

    if(debug)
        printf("\n\n STEP 4: Adding scan max values to original array of values , grid dim: %d, block dim: %d \n",grid_size,BLOCK_SIZE);
    printArray(idata,N,debugStep4);
    
    printf("\n ALL GPU: N: %d ,exuction time: %f , Last Elem: %f\n",N,totalExecutionTime,idata[N-1]);
    
    //Free used memory
    cudaFree(idata);
    cudaFree(scanMax);

    cudaDeviceReset();

    


}


__global__ void finalScanKernel(float* initialData,float* maxValues){
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;

    
    if(blockIdx.x!=0){
        initialData[thIdx]+=maxValues[blockIdx.x-1];
    }
}



__global__ void scanKernel(float *idata,int n,int startIndex)
{
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x+startIndex;
    int tid = threadIdx.x;
    //Add appropriat data elements to shared memeory of the blocks
    __shared__ float temp[1024];
    temp[tid] = idata[thIdx];
    __syncthreads();
    //Make sure the value at the current point is equal that of the previous value
    // The fllowiing loop creats an initial scan
    for (int offset = 1; offset < n; offset *= 2)
    {
        if (tid >= offset)
            temp[tid] += temp[tid - offset];
        __syncthreads();
    }
    idata[thIdx] = temp[tid];
}

__global__ void findMaxValueArray(float* input,float* output){
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    //This makes the assumption that each max value in a block is the last element in the block.
    if(tid==BLOCK_SIZE-1){
        output[blockIdx.x]=input[thIdx];
    }
}




void printArray(float *arr, int sizeOfArray, bool printStament)
{
    if(printStament){
        printf("Printing %d values \n", sizeOfArray);
        for (int i = 0; i < sizeOfArray; i++)
        {
            printf("%f|", arr[i]);
            if((i+1)%BLOCK_SIZE==0){
                printf("\t");
            }
        }
        printf("\n Fin printing");
    
    }
    // printf("\n");
}


