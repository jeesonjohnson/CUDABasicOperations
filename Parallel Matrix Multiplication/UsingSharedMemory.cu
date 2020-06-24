#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define BLOCK_SIZE 16
// Matrices are stored in row-major order
typedef struct
{
    int width;
    int height;
    float *elements;
    int stride;
} Matrix;
__global__ void MultKernShared(const Matrix A, const Matrix B, Matrix C);

// For simplicity you could just put all the content into this file instead of having a separate .h file
// Now, we are going to have to define some utilities to get and set
// elements in matrices and to get sub-matrices. Please refer to Lecture 13
// to make sure you understand why we need to use the "stride" property to
// walk down the column of a sub-matrix
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
// Note, we are not doing any copying here - just finding the address of the
// start of the sub-matrix we are interested in
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}
// Now we have all the device functions we need to define the matrix multiplication
// Kernel. This is going to be called from the host by MatMul()
__global__ void MultSharedKernel(Matrix A, Matrix B, Matrix C)
{
    // Identify the Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Now find the row and column of the element within Csub
    // that this thread is going to calculate
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
    {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from global memory into shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronise to make sure the sub-matrices are completely loaded
        // before starting the computation for each phase
        __syncthreads();
        // Now multiply Asub and Bsub together to complete phase m of the
        // calculation of this threads element of Csub
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronise again to make sure that the preceding calculation
        // has been completed by all threads in the block before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Once all the phases are complete we can write Csub to device (global) memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatrixMult(const Matrix h_A, const Matrix h_B, Matrix h_C) {
    //Functionaltiy for CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = h_A.width;
    d_A.height = h_A.height;
    size_t size = h_A.width * h_A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    //printf("CUDA malloc h_A: %s\n",cudaGetErrorString(err));
    cudaMemcpy(d_A.elements, h_A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = h_B.width;
    d_B.height = h_B.height;
    size = h_B.width * h_B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);
    //printf("CUDA malloc h_B: %s\n",cudaGetErrorString(err));
    cudaMemcpy(d_B.elements, h_B.elements, size, cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = h_C.width;
    d_C.height = h_C.height;
    size = h_C.width * h_C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);

    //printf("CUDA malloc h_C: %s\n",cudaGetErrorString(err));
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(h_B.width / dimBlock.x, h_A.height / dimBlock.y);

    //Cuda timing functionaltiy
    cudaEventRecord(start);
    MultSharedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    //Cuda Timing functionaltiy
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    //printf("Run kernel: %s\n", cudaGetErrorString(err));
    // Read C from device memory
    err = cudaMemcpy(h_C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    //printf("Copy h_C off device: %s\n",cudaGetErrorString(err));
    // Free device memory
    // cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Elapsed time was: %f\n milliseconds", milliseconds);
    printf("%f\n",milliseconds);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
   }



   void printSelectAmount(Matrix A,Matrix B,Matrix C,int N){
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%f ", A.elements[i * A.width + j]);
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%f ", B.elements[i * B.width + j]);
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%f ", C.elements[i * C.width + j]);
        printf("\n");
    }
    printf("\n");
}

void mainMatrixFunction(int N){
    Matrix A, B, C;
    // Read Dimensions of A and B
    A.height = N;
    A.width = N;
    B.height = A.width;
    B.width = N;
    A.elements = (float *)malloc(A.width * A.height * sizeof(float));
    B.elements = (float *)malloc(B.width * B.height * sizeof(float));
    C.height = A.height;
    C.width = B.width;
    C.elements = (float *)malloc(C.width * C.height * sizeof(float));
    for (int i = 0; i < A.height; i++)
        for (int j = 0; j < A.width; j++)
            A.elements[i * A.width + j] = (float)(rand() % 3);
    for (int i = 0; i < B.height; i++)
        for (int j = 0; j < B.width; j++)
            B.elements[i * B.width + j] = (float)(rand() % 2);
    MatrixMult(A, B, C);
    printSelectAmount(A,B,C,N);
}

int main(int argc, char *argv[])
{
    for(int x=1;x<1024;x*=2){
        //printf("%d\n",16*x);
        mainMatrixFunction(16);
    }

    return 0;
}