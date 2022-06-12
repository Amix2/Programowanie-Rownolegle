#include <stdio.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>
#include "gputimer.h"
#include "1_reduction.h"

namespace
{
    void run_benchmark(void (*reduce)(float*, float*, int, int),
        float* d_outPtr, float* d_inPtr, int size);
    void init_input(float* data, int size);
    float get_cpu_result(float* data, int size);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int mainReductionGlobal(int argc, char *argv[])
{
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;
	

    unsigned int size = 1 << 8;

    float result_host, result_gpu;

    srand(2019);

    // Allocate memory
    h_inPtr = (float *)malloc(size * sizeof(float));

    // Data initialization with random values
    init_input(h_inPtr, size);

    // Prepare GPU resource
    cudaMalloc((void **)&d_inPtr, size * sizeof(float));
    cudaMalloc((void **)&d_outPtr, size * sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);

    // Get reduction result from GPU
    run_benchmark(reduction, d_outPtr, d_inPtr, size);
    run_benchmark(global_reduction, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    // Get reduction result from GPU

    // Get all sum from CPU
    result_host = get_cpu_result(h_inPtr, size);
    printf("host: %f, device %f\n", result_host, result_gpu);

    // Terminates memory
    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}

namespace
{

    void run_benchmark(void (*reduce)(float*, float*, int, int),
        float* d_outPtr, float* d_inPtr, int size)
    {
        int num_threads = 256;
        int test_iter = 1;

        // warm-up
        reduce(d_outPtr, d_inPtr, num_threads, size);

        // initialize timer
        GpuTimer timer;
        timer.Start();
        ////////
        // Operation body
        ////////
        for (int i = 0; i < test_iter; i++)
        {
            cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
            reduce(d_outPtr, d_outPtr, num_threads, size);
            cudaDeviceSynchronize();
        }

        // getting elapsed time
        timer.Stop();

        cudaDeviceSynchronize();


        // Compute and print the performance
        float elapsed_time_msed = timer.Elapsed() / (float)test_iter;
        float bandwidth = size * sizeof(float) / elapsed_time_msed / 1e6;
        printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);


    }

    void init_input(float* data, int size)
    {
        for (int i = 0; i < size; i++)
        {
            // Keep the numbers small so we don't get truncation error in the sum
            data[i] = (rand() & 0xFF) / (float)RAND_MAX;
        }
    }

    float get_cpu_result(float* data, int size)
    {
        float result = 0.f;
        for (int i = 0; i < size; i++)
            result += data[i];

        return (float)result;
    }

}