#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "gputimer.h"
#include "cudaHeader.cuh"

int main(int argc, char* argv[])
{
	int mode = 4;
	if (argc >= 2)
		mode = atoi(argv[1]);
	switch (mode)
	{
	case 1:
		mainMatTranspose_1(argc-1, argv+1);
		break;
	case 2:
		mainImageScailing(argc-1, argv+1);
		break;
	case 3:
		mainReductionGlobal(argc - 1, argv + 1);
		break;
	case 4:
		mainWarmDivergence(argc - 1, argv + 1);
		break;
	}
	//mainVectorAdd(argc, argv);

	return 0;
}