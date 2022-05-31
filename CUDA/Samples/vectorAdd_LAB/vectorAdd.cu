#include <stdio.h>
#include <stdlib.h>
#include <cmath>

  // For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "gputimer.h"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		C[i] = A[i] + B[i] + 0.0f;
	}
}

void TestTime(int vecSize, int threadsPerBlockMin, int threadsPerBlockMax, int threadsPerBlockStep, int repeats)
{
	

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = vecSize;
	size_t size = numElements * sizeof(float);
	//printf("[Vector addition of %d elements on %d threads per block]\n", numElements, threadsPerBlock);

	// Allocate the host input vector A
	float* h_A = (float*)malloc(size);

	float* h_B = (float*)malloc(size);

	float* h_C = (float*)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL) { fprintf(stderr, "Failed to allocate host vectors!\n"); exit(EXIT_FAILURE); }

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = (float)i / numElements;// rand() / (float)RAND_MAX;
		h_B[i] = 1.f - (float)i / numElements;// rand() / (float)RAND_MAX;
	}

	// Allocate the device input vector A
	float* d_A = NULL;
	err = cudaMalloc((void**)&d_A, size);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	// Allocate the device input vector B
	float* d_B = NULL;
	err = cudaMalloc((void**)&d_B, size);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	// Allocate the device output vector C
	float* d_C = NULL;
	err = cudaMalloc((void**)&d_C, size);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	// Copy the host input vectors A and B in host memory to the device input
	// vectors in
	// device memory
	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	// Launch the Vector Add CUDA Kernel
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	for (int threads = threadsPerBlockMin; threads <= threadsPerBlockMax; threads += threadsPerBlockStep)
	{
		if (threads == 32)
			continue;
		int blocksPerGrid = (numElements + threads - 1) / threads;

		GpuTimer timer;
		timer.Start();
		int reps = repeats;
		for(; reps; reps--)
			vectorAdd << <blocksPerGrid, threads >> > (d_A, d_B, d_C, numElements);
		timer.Stop();
		printf("%d\t%d\t%f\n", numElements, threads, timer.Elapsed() / repeats);
		err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		if (err != cudaSuccess) { fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

		// Verify that the result vector is correct
		for (int i = 0; i < numElements; ++i) {
			if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
				printf("Result verification failed at element %d!\n", i);
				exit(EXIT_FAILURE);
			}
		}
	}




	err = cudaGetLastError();

	if (err != cudaSuccess) { fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	//printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	//printf("Test PASSED\n");

	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	err = cudaFree(d_B);

	if (err != cudaSuccess) { fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

	err = cudaFree(d_C);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	//printf("Done\n");
}
/**
 * Host main routine
 */
int main(int argc, char* argv[])
{
	//int threadsPerBlock = 256;
	//if (argc >= 2)
	//	vecSize = atoi(argv[1]);
	//if (argc >= 3)
	//	threadsPerBlock = atoi(argv[2]);
	//if (argc >= 4)
	//	repeats = atoi(argv[4]);

	int vecSizeMax  = 100000000;
	int vecSizeMin  = 100000;
	int vecSizeStep = 1000000;
	for (int vecSize = vecSizeMin; vecSize <= vecSizeMax; vecSize += vecSizeStep)
	{
		int repeats = 20000000000 / vecSize;
		TestTime(vecSize, 2, 64, 4, repeats);
	}

	return 0;
}