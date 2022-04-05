#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[]) {
	int scheduleMode = 1;
	int testSampleCount = 1;
	int threadNum = 1;

    int size = 100000, i, j, idx;
    if(argc>=2)
    {
        size = atoi(argv[1]);
    }
    if(argc>=3)
    {
        threadNum = atoi(argv[2]);
    }

	omp_set_num_threads(threadNum);
	unsigned short int x16v[3];
	double* Array = new double[size];
	
	if (scheduleMode == 0) omp_set_schedule(omp_sched_static, 1);
	if (scheduleMode == 1) omp_set_schedule(omp_sched_static, 10000);
	if (scheduleMode == 2) omp_set_schedule(omp_sched_dynamic, 1);
	if (scheduleMode == 3) omp_set_schedule(omp_sched_dynamic, 10000);
	if (scheduleMode == 4) omp_set_schedule(omp_sched_guided, 1);
	if (scheduleMode == 5) omp_set_schedule(omp_sched_guided, 10000);

	double timeStartInit, timeEndInit;
	double timeStartSort, timeEndSort;

	while (testSampleCount) {
		std::vector<std::vector<double>> buckets(size);
		std::vector<double> sorted_list;
		sorted_list.reserve(size);
		std::vector<omp_lock_t> locks(size);
		std::vector<int> sizes(size);
		for (i = 0;i < size; i++) {
			buckets[i].reserve(8);
			//omp_init_lock_with_hint(&locks[i], omp_sync_hint_uncontended);
			omp_init_lock(&locks[i]);
		}

		testSampleCount--;


		timeStartInit = omp_get_wtime();
		
		#pragma omp parallel for private(i, x16v) schedule(runtime)
		for(i=0; i<size; i++)
		{
			Array[i] = erand48(x16v);
		}

		#pragma omp parallel for private(i, x16v, idx) schedule(runtime)
		for(i=0; i<size; i++)
		{
			idx = int(Array[i] * size);
			omp_set_lock(&locks[idx]);
			buckets[idx].push_back(Array[i]);
			omp_unset_lock(&locks[idx]);
		}

		#pragma omp parallel for private(i, x16v, idx) schedule(runtime)
		for(i=0; i<size; i++)
		{
			std::sort(buckets[i].begin(), buckets[i].end());
			sizes[i] = buckets[i].size();
		}
		
		for (i = 1;i<size;i++)sizes[i] += sizes[i-1];

		//#pragma omp parallel for private(i, j, x16v, idx) schedule(runtime)
		for(i=0; i<size; i++)
		{
			for (j = 0; j < buckets[i].size();j++){
				sorted_list.push_back(buckets[i][j]);
			}
		}

		timeEndInit = omp_get_wtime();

		// check correctness
		bool corr = true;
		std::sort(Array, Array + size);
		for (i = 0;i<size;i++){
			if(Array[i] != sorted_list[i]) corr = false;
		}
		if (corr) printf("good\n");
		else printf("bad\n");

		printf("time: %lf", timeEndInit - timeStartInit);
	}

    return 0; 
}