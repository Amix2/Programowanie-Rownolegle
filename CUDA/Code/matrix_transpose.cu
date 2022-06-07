#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include "gputimer.h"

#define N_d 32
#define BLOCK_SIZE 32

namespace
{
	__global__ void matrix_transpose_naive_dis_store(int* input, int* output, int N)
	{
		int indexX = threadIdx.x + blockIdx.x * blockDim.x;
		int indexY = threadIdx.y + blockIdx.y * blockDim.y;
		int index = indexY * N + indexX;
		int transposedIndex = indexX * N + indexY;

		// this has discoalesced global memory store
		output[transposedIndex] = input[index];

		// this has discoalesced global memore load
		// output[index] = input[transposedIndex];
	}

	__global__ void matrix_transpose_naive_dis_load(int* input, int* output, int N)
	{
		int indexX = threadIdx.x + blockIdx.x * blockDim.x;
		int indexY = threadIdx.y + blockIdx.y * blockDim.y;
		int index = indexY * N + indexX;
		int transposedIndex = indexX * N + indexY;

		// this has discoalesced global memory store
		//output[transposedIndex] = input[index];

		// this has discoalesced global memore load
		output[index] = input[transposedIndex];
	}

	template<int BS>
	__global__ void matrix_transpose_shared_slow(int* input, int* output, int N)
	{
		__shared__ int sharedMemory[BS][BS];
		// global index
		int indexX = threadIdx.x + blockIdx.x * blockDim.x;
		int indexY = threadIdx.y + blockIdx.y * blockDim.y;

		// transposed global memory index
		int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
		int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

		// local index
		int localIndexX = threadIdx.x;
		int localIndexY = threadIdx.y;

		int index = indexY * N + indexX;
		int transposedIndex = tindexY * N + tindexX;

		// reading from global memory in coalesed manner and performing tanspose in shared memory
		sharedMemory[localIndexX][localIndexY] = input[index];

		__syncthreads();

		// writing into global memory in coalesed fashion via transposed data in shared memory
		output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
	}

	template<int BS>
	__global__ void matrix_transpose_shared_fast(int* input, int* output, int N)
	{
		__shared__ int sharedMemory[BS][BS + 1];

		// global index
		int indexX = threadIdx.x + blockIdx.x * blockDim.x;
		int indexY = threadIdx.y + blockIdx.y * blockDim.y;

		// transposed global memory index
		int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
		int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

		// local index
		int localIndexX = threadIdx.x;
		int localIndexY = threadIdx.y;

		int index = indexY * N + indexX;
		int transposedIndex = tindexY * N + tindexX;

		// reading from global memory in coalesed manner and performing tanspose in shared memory
		sharedMemory[localIndexX][localIndexY] = input[index];

		__syncthreads();

		// writing into global memory in coalesed fashion via transposed data in shared memory
		output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
	}

	//basically just fills the array with index.
	void fill_array(int* data) {
		for (int idx = 0; idx < (N_d * N_d); idx++)
			data[idx] = idx;
	}

	void print_output(int* a, int* b) {
		printf("\n Original Matrix::\n");
		for (int idx = 0; idx < (N_d * N_d); idx++) {
			if (idx % N_d == 0)
				printf("\n");
			printf(" %d ", a[idx]);
		}
		printf("\n Transposed Matrix::\n");
		for (int idx = 0; idx < (N_d * N_d); idx++) {
			if (idx % N_d == 0)
				printf("\n");
			printf(" %d ", b[idx]);
		}
	}

	template<int BS>
	float TestTime(int N, int repeats, int option)	// 1-naive slow, 2-naive fast, 3-shared slow, 4-shared fast
	{
		int* a, * b;
		int* d_a, * d_b; // device copies of a, b, c

		int size = N * N * sizeof(int);

		// Alloc space for host copies of a, b, c and setup input values
		a = (int*)malloc(size); fill_array(a);
		b = (int*)malloc(size);

		// Alloc space for device copies of a, b, c
		cudaMalloc((void**)&d_a, size);
		cudaMalloc((void**)&d_b, size);

		// Copy inputs to device
		cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

		dim3 blockSize(BS, BS, 1);
		dim3 gridSize(N / BS, N / BS, 1);

		GpuTimer timer;
		timer.Start();
		for (int reps = repeats; reps; reps--)
		{
			switch (option)
			{
			case 1:
				matrix_transpose_naive_dis_store <<<gridSize, blockSize>>> (d_a, d_b, N);
				break;
			case 2:
				matrix_transpose_naive_dis_load <<<gridSize, blockSize>>> (d_a, d_b, N);
				break;
			case 3:
				matrix_transpose_shared_slow<BS> <<<gridSize, blockSize>>> (d_a, d_b, N);
				break;
			case 4:
				matrix_transpose_shared_fast<BS> <<<gridSize, blockSize>>> (d_a, d_b, N);
				break;
			}
		}
		timer.Stop();
		float time = timer.Elapsed() / repeats;

		// Copy result back to host
		cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

		// terminate memories
		free(a);
		free(b);
		cudaFree(d_a);
		cudaFree(d_b);

		return time;
	}

	template<int BS>
	void TestAll(int N, int repeats)
	{
		printf("%d\t", BS);
		for (int option = 1; option <= 4; option++)
		{
			float time = TestTime<BS>(N, repeats, option);
			printf("%f\t", time);
		}
		printf("\n");
	}
}
int mainMatTranspose_1(int argc, char* argv[])
{
	int N = 1024;
	int repeats = 1;

	if (argc >= 2)
		N = atoi(argv[1]);
	if (argc >= 3)
		repeats = atoi(argv[2]);

	printf("N=%d, reps=%d\n", N, repeats);
	TestAll<4>(N, repeats);
	TestAll<5>(N, repeats);
	TestAll<6>(N, repeats);
	TestAll<7>(N, repeats);
	TestAll<8>(N, repeats);
	TestAll<9>(N, repeats);
	TestAll<10>(N, repeats);
	TestAll<11>(N, repeats);
	TestAll<12>(N, repeats);
	TestAll<13>(N, repeats);
	TestAll<14>(N, repeats);
	TestAll<15>(N, repeats);
	TestAll<16>(N, repeats);
	TestAll<17>(N, repeats);
	TestAll<18>(N, repeats);
	TestAll<19>(N, repeats);
	TestAll<20>(N, repeats);
	TestAll<21>(N, repeats);
	TestAll<22>(N, repeats);
	TestAll<23>(N, repeats);
	TestAll<24>(N, repeats);
	TestAll<25>(N, repeats);
	TestAll<26>(N, repeats);
	TestAll<27>(N, repeats);
	TestAll<28>(N, repeats);
	TestAll<29>(N, repeats);
	TestAll<30>(N, repeats);
	TestAll<31>(N, repeats);
	TestAll<32>(N, repeats);

	return 0;
}