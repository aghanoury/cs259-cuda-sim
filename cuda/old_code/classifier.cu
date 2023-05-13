#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define T 32

// CPU function (TEST)
/*
void FC_layer_cpu(int* in, int* weights, int* out, int Ni, int Nn)
{
int sum = 0;
for (int i = 0; i < Nn; i++) { // output
sum = 0;
for (int j = 0; j < Ni; j++) { // input
sum += in[j] * weights[j * Ni + i];
}
out[i] = sum;
}
}
*/

// GPU functions
__global__ void FC_layer_gpu(int *in, int *weights, int *out, int Ni, int Nn)
{
    // in: 1 * 25088     w: 25088 * 4096    out: 1 * 4096

    // Putting in, w in the Shared Memory
    // extern __shared__ int ins[T];
    // extern __shared__ int ws[T][T];

    int i = bx * T + tx;
    int j = by * T + ty;

    int sum = 0;
    for (int m = 0; m < (Ni / T); m++)
    {

        // Shared Memory
        extern __shared__ int ins[T];
        extern __shared__ int ws[T][T];

        /*
        I, by, m
        Isub = &I[T * by + T * m]

        Isub, ty, tx
        Isub [ty * ? + tx]

        W, m, bx
        Wsub = &W[Nn*T*m + T*bx]
        Wsub [ty * Nn * ? + tx]
        */

        ins[tx] = in[(m + by) * T + tx]; // in[m * T + tx];
        ws[ty][tx] = weights[(m * T + ty) * Nn + bx * T + tx];
        __syncthreads();

        for (int k = 0; k < T; k++)
        {
            sum += ins[k] * ws[k][tx];
        }
    }
    __syncthreads();

    /*
    O, by, bx
    Osub = &O[T * by + T * bx]

    Osub, ty, tx
    Osub [ty * ? + tx]
    */
    out[(bx + by) * T + tx] = sum; // out[j*m + i] = sum;
}

__device__ void AF_layer_gpu(int *in, float *out)
{ // Activation Function
    int idx = threadIdx.x;
    out[idx] = 1.0 / (1.0 + exp(-in[idx]));
}

int main(int argc, char *argv[])
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    int Ni, Nn;

    Ni = 4096; // 25088;
    Nn = 4096; // 4096;

    // CPU memory
    int *in, *out, *weights, *host_out;

    in = (int *)malloc(Ni * sizeof(int));
    out = (int *)malloc(Nn * sizeof(int));
    weights = (int *)malloc(Ni * Nn * sizeof(int));
    host_out = (int *)malloc(Nn * sizeof(int));

    for (int i = 0; i < Ni; i++)
    {
        in[i] = rand(); // % 10;
        for (int j = 0; j < Nn; j++)
            weights[i * Ni + j] = rand(); // % 10;
    }

    // CPU (TEST)
    // FC_layer_cpu(in, weights, host_out, Ni, Nn);

    // GPU memory
    int *device_in, *device_out, *device_weights;

    cudaMalloc(&device_in, Ni * sizeof(int));
    cudaMalloc(&device_out, Nn * sizeof(int));
    cudaMalloc(&device_weights, Ni * Nn * sizeof(int));

    cudaMemcpy(device_in, in, Ni * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights, weights, Ni * Nn * sizeof(int), cudaMemcpyHostToDevice);

    //////////////////////////////////////////////////

    dim3 dimBlock(T, T);
    dim3 dimGrid(Nn / dimBlock.x, Ni / dimBlock.y); // # 1

    // Timer
    cudaEventRecord(start);

    FC_layer_gpu<<<dimGrid, dimBlock>>>(device_in, device_weights, device_out, Ni, Nn);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, device_out, Nn * sizeof(int), cudaMemcpyDeviceToHost);

    // Timing Result
    printf("\n");
    printf("Fully connected layer\n");
    printf("Time (GPU): %16.10f ms\n", milliseconds);
    printf("\n");

    // Free memory
    free(in);
    free(out);
    free(weights);

    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_weights);
}
