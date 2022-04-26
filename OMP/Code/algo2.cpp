#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>

double get_random_number(){
    static thread_local std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    return distribution(generator);
}

void bucket_sort(size_t size, size_t num_buckets, size_t num_repeats){
    size_t i, j, idx;
    double* Array = new double[size];
    int bucket_size = size / num_buckets;

    double time_start, time_end, t1, t2, t3;
    while (num_repeats) {
        std::vector<std::vector<double>> buckets(num_buckets);
        std::vector<double> sorted_list(size);
        std::vector<omp_lock_t> locks(num_buckets);
        std::vector<int> sizes(num_buckets);
        time_start = omp_get_wtime();
        for (i = 0;i < num_buckets; i++) {
            buckets[i].reserve(bucket_size);
            //omp_init_lock_with_hint(&locks[i], omp_sync_hint_uncontended); 
            omp_init_lock(&locks[i]);
        }

        num_repeats--;
        #pragma omp parallel private(i)
        {
            #pragma omp parallel for schedule(runtime)
            for(i=0; i<size; i++)
            {
                
				Array[i] = get_random_number();
            }
        }
        t1 = omp_get_wtime();

        #pragma omp parallel for private(i, idx) schedule(runtime)
        for(i=0; i<size; i++)
        {
            idx = int(Array[i] * num_buckets);
            omp_set_lock(&locks[idx]);
            buckets[idx].push_back(Array[i]);
            omp_unset_lock(&locks[idx]);
        }
        t2 = omp_get_wtime();

        #pragma omp parallel for private(i, idx) schedule(runtime)
        for(i=0; i<num_buckets; i++)
        {
            std::sort(buckets[i].begin(), buckets[i].end());
            sizes[i] = buckets[i].size();
        }
        
        for (i = 1;i<num_buckets;i++)sizes[i] += sizes[i-1];

        t3 = omp_get_wtime();

        #pragma omp parallel for private(i, j, idx) schedule(runtime)
        for(i=0; i<num_buckets; i++)
        {
            for (j = 0; j < buckets[i].size();j++){
                if (i == 0)
                    sorted_list[j] =  buckets[i][j];
                else
                    sorted_list[sizes[i-1] + j] = buckets[i][j];
            }
        }

        time_end = omp_get_wtime();

        #ifdef CHECK 
        {
            // check correctness
            bool corr = true;
            std::sort(Array, Array + size);
            for (i = 0;i<size;i++){
                if(Array[i] != sorted_list[i]) corr = false;
            }
            if (corr) printf("good\n");
            else printf("bad\n");
        }
        #endif
        printf("%d\t", omp_get_max_threads());
        printf("%ld\t%ld\t", size, num_buckets);
        printf("%lf\t", t1 - time_start); // init
        printf("%lf\t", t2 - t1); // into buckets
        printf("%lf\t", t3 - t2); // sort buckets
        printf("%lf\t", time_end - t3); // finalize
        printf("%lf\n", time_end - time_start); // total
        for (i = 0;i < num_buckets; i++) {
            //omp_init_lock_with_hint(&locks[i], omp_sync_hint_uncontended); 
            omp_destroy_lock(&locks[i]);
        }
    }

}


int main(int argc, char* argv[]) {
    int scheduleMode = 1;
    int testSampleCount = 10;
    int threadNum = 1;
    int option = 2;
    int size = 100000;
    if(argc>=2)
    {
        size = atoi(argv[1]);
    }
    if(argc>=3)
    {
        threadNum = atoi(argv[2]);
    }
    if(argc>=4){
        option = atoi(argv[3]);
    }
    if(argc>=5){
        testSampleCount = atoi(argv[4]);
    }

    omp_set_num_threads(threadNum);
    
    if (scheduleMode == 0) omp_set_schedule(omp_sched_static, 1);
    if (scheduleMode == 1) omp_set_schedule(omp_sched_static, 10000);
    if (scheduleMode == 2) omp_set_schedule(omp_sched_dynamic, 1);
    if (scheduleMode == 3) omp_set_schedule(omp_sched_dynamic, 10000);
    if (scheduleMode == 4) omp_set_schedule(omp_sched_guided, 1);
    if (scheduleMode == 5) omp_set_schedule(omp_sched_guided, 10000);

    std::vector<omp_lock_t> locks(1000000);
    for (int i = 0;i < 1000000; i++) {
        omp_init_lock(&locks[i]);
    }

    if(option == 2){
        int step = size / 200;
        if(step < 1)
            step = 1;
        for(int bucketCount = 1; bucketCount <= 0.5f * size; bucketCount += std::min(bucketCount, step))
        {
            omp_set_num_threads(1);
            bucket_sort(size, bucketCount, testSampleCount);
        }
    }
    if(option == 3){
        int maxThreadCount = 4;
        int bucketCount = size * 0.03;

        for (int thCount = 1; thCount <= maxThreadCount; thCount++)
        {
            omp_set_num_threads(thCount);
            bucket_sort(size, bucketCount, testSampleCount);
        }
    }
    if(option == 4) {
        int step = size / 200;
        if(step < 1)
            step = 1;
        for(int curent_size = 1; curent_size < size; curent_size += std::min(curent_size, step))
        {
            omp_set_num_threads(1);
            bucket_sort(curent_size, curent_size * 0.03, testSampleCount);
        }
    }

    return 0; 
}