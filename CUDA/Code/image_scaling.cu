#include <cuda_runtime.h>
#include<stdio.h>
#include"scrImagePgmPpmPackage.h"
#include <string>
#include "gputimer.h"

namespace
{
	//Kernel which calculate the resized image
	__global__ void createResizedImage(unsigned char* imageScaledData, int scaled_width, float scale_factor, cudaTextureObject_t texObj)
	{
		const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned index = tidY * scaled_width + tidX;
		//printf("%d\t%d\n", )
		imageScaledData[index] = tex2D<unsigned char>(texObj, (float)(tidX * scale_factor), (float)(tidY * scale_factor));
	}

	float GetTime(int pic, int block_size, float scaling_ratio, int repeats)
	{
		int height = 0, width = 0, scaled_height = 0, scaled_width = 0;
		//Define the scaling ratio	
		unsigned char* data;
		unsigned char* scaled_data, * d_scaled_data;
		std::string inputString, outputString;
		char inputStr[1024];
		char outputStr[1024];
		if (pic == 1)
		{
			inputString = "aerosmith-double.pgm" ;
			outputString = "aerosmith-double-scaled.pgm";
		}
		else if (pic == 2)
		{
			inputString = "voyager2.pgm";
			outputString = "voyager2-scaled.pgm";
		}
		strcpy(inputStr, inputString.c_str());
		strcpy(outputStr, outputString.c_str());
		cudaError_t returnValue;

		//Create a channel Description to be used while linking to the tecture
		cudaArray* cu_array;
		cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);

		get_PgmPpmParams(inputStr, &height, &width);	//getting height and width of the current image
		data = (unsigned char*)malloc(height * width * sizeof(unsigned char));
		//printf("\n Reading image width height and width [%d][%d]", height, width);
		scr_read_pgm(inputStr, data, height, width);//loading an image to "inputimage"

		scaled_height = (int)(height * scaling_ratio);
		scaled_width = (int)(width * scaling_ratio);
		scaled_data = (unsigned char*)malloc(scaled_height * scaled_width * sizeof(unsigned char));
		//printf("\n scaled image width height and width [%d][%d]", scaled_height, scaled_width);

		//Allocate CUDA Array
		returnValue = cudaMallocArray(&cu_array, &channelDesc, width, height);
		//printf("\n%s", cudaGetErrorString(returnValue));
		// cudaMemcpyToArray()
		returnValue = (cudaError_t)(returnValue | cudaMemcpyToArray(cu_array, 0, 0, data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));
		// printf("\n%s", cudaGetErrorString(returnValue));

		if (returnValue != cudaSuccess)
		{
			printf("\n Got error while running CUDA API Array Copy");
		}

		// Step 1. Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cu_array;

		// Step 2. Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		// Step 3: Create texture object
		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		if (returnValue != cudaSuccess)
		{
			printf("\n Got error while running CUDA API Bind Texture");
		}

		cudaMalloc(&d_scaled_data, scaled_height * scaled_width * sizeof(unsigned char));

		dim3 dimBlock(block_size, block_size, 1);
		dim3 dimGrid(scaled_width / dimBlock.x + 1, scaled_height / dimBlock.y + 1, 1);
		//printf("\n Launching grid with blocks [%d][%d] ", dimGrid.x, dimGrid.y);

		GpuTimer timer;
		timer.Start();

		for (int reps = repeats; reps; reps--)
			createResizedImage <<<dimGrid, dimBlock>>> (d_scaled_data, scaled_width, 1.0 / scaling_ratio, texObj);
		
		timer.Stop();
		float time = timer.Elapsed() / repeats;

		returnValue = (cudaError_t)(returnValue | cudaDeviceSynchronize());

		returnValue = (cudaError_t)(returnValue | cudaMemcpy(scaled_data, d_scaled_data, scaled_height * scaled_width * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		if (returnValue != cudaSuccess)
		{
			printf("\n Got error while running CUDA API kernel\n");
		}

		// Step 5: Destroy texture object
		cudaDestroyTextureObject(texObj);

		scr_write_pgm(outputStr, scaled_data, scaled_height, scaled_width, "####"); //storing the image with the detections

		if (data != NULL)
			free(data);
		if (cu_array != NULL)
			cudaFreeArray(cu_array);
		if (scaled_data != NULL)
			free(scaled_data);
		if (d_scaled_data != NULL)
			cudaFree(d_scaled_data);

		return time;
	}
}

int mainImageScailing(int argc, char* argv[])
{
	int reps = 1000;
	printf("BS\tp1s05\tp1s1\tp1s2\tp1s4\tp2s05\tp2s1\tp2s2\tp2s4\n");
	for (int BS = 2; BS <= 32; BS++)
	{
		printf("%d\t", BS);

		for (int p = 1; p <= 2; p++)
		{
			for (float s = 0.5f; s <= 2.1f; s *= 2)
			{
				float time = GetTime(p, BS, s, reps);
				printf("%f\t", time);

			}
		}
		printf("\n");

	}
	return 0;
}
