#define _CRT_RAND_S

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>   
#include <iostream>
#include <vector> 
#include <algorithm>
#include <cstdlib>
#include <random>

/*
////////////////////////////////////////////////////////////////////////////////

STRUKTURA PLIKU
1. Funkcje do poszczególnych etapów sortowania
2. Funkcja sortująca BucketSort 
3. Funkcje main dla każdego testu oraz główna funkcja main
*/

double DoubleRand() {
    static thread_local std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    return distribution(generator);
}

void CreateRandomData(double* Array, int size)
{
 #pragma omp parallel for schedule(runtime) firstprivate(Array)
    for (int s = 0; s < size; s++)
    {
        Array[s] = DoubleRand();
    }
}

void FillBuckets(int size, std::vector<double>** Buckets, double* Array, int bucketCount)
{
#pragma omp parallel firstprivate(Buckets, Array, bucketCount)
    { 
        const int myThreadNum = omp_get_thread_num();
        std::vector<double>* myBucket = Buckets[myThreadNum];
        #pragma omp for schedule(runtime)
        for (int s = 0; s < size; s++)
        {
           const double val = Array[s];
           const int valBucket = val * bucketCount;
           myBucket[valBucket].push_back(val);
        }
    }
}

void MergeBuckets(int bucketCount, std::vector<double>* SingleBuckets, int threadNum, std::vector<double>** Buckets)
{ 
#pragma omp parallel for firstprivate(threadNum, SingleBuckets, Buckets) schedule(runtime)
    for (int b = 0; b < bucketCount; b++)
    {
        std::vector<double>* mySingleBucket = &SingleBuckets[b];
        for (int t = 0; t < threadNum; t++)
        {
            std::vector<double>* mergeBucket = &Buckets[t][b];
            mySingleBucket->insert(mySingleBucket->end(), mergeBucket->begin(), mergeBucket->end());
        }
    }
}

void SortSingleBucket(int bucketCount, std::vector<double>* SingleBuckets, std::vector<int>* bucketSizes)
{
#pragma omp parallel for schedule(runtime) firstprivate(SingleBuckets, bucketSizes)
     for (int b = 0; b < bucketCount; b++)
     {
         std::vector<double>* mySingleBucket = &SingleBuckets[b];
         std::sort(mySingleBucket->begin(), mySingleBucket->end());
         bucketSizes->at(b) = (int)mySingleBucket->size();
     }
}

void ExportToSortedArray(int bucketCount, std::vector<int>* bucketSizes, std::vector<double>* SingleBuckets, double* Array)
{
    for (int b = 1; b < bucketCount; b++)
        bucketSizes->at(b) += bucketSizes->at(b - 1);

    #pragma omp parallel for schedule(runtime) firstprivate(SingleBuckets, bucketSizes, Array)
    for (int b = 0; b < bucketCount; b++)
    {
        int insertId = b == 0 ? 0 : bucketSizes->at(b - 1);
        for (int i = 0; i < SingleBuckets[b].size(); i++)
        {
            Array[insertId + i] = SingleBuckets[b][i];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

double BucketSort(int threadNum, int size, int bucketCount, int scheduleMode, int testSampleCount)
{
    bool DebugPrint = false;
    omp_set_schedule(omp_sched_guided, 10);
    int timeMul = 1000;
    omp_set_num_threads(threadNum);
    double* Array = new double[size];
    std::vector<double>** Buckets = new std::vector<double>*[threadNum];
    int bucketSize = size / bucketCount;
    bucketSize += bucketSize * 1.0f;
    for (int t = 0; t < threadNum; t++)
    {
        Buckets[t] = new std::vector<double>[bucketCount];
        for (int b = 0; b < bucketCount; b++)
            Buckets[t][b].reserve(bucketSize);
    }
    std::vector<double>* SingleBuckets = new std::vector<double>[bucketCount];
    for (int b = 0; b < bucketCount; b++)
    {
        SingleBuckets[b].reserve(bucketSize);
    }
    std::vector<int> bucketSizes(bucketCount);

    double timeStart = omp_get_wtime();

    CreateRandomData(Array, size);

    double t1 = omp_get_wtime();

    FillBuckets(size, Buckets, Array, bucketCount);

    double t2 = omp_get_wtime();

    MergeBuckets(bucketCount, SingleBuckets, threadNum, Buckets);

    double t3 = omp_get_wtime();

    SortSingleBucket(bucketCount, SingleBuckets, &bucketSizes);

    double t4 = omp_get_wtime();

    ExportToSortedArray(bucketCount, &bucketSizes, SingleBuckets, Array);

    double timeEnd = omp_get_wtime();

    bool PrintPartTimes = false;
    bool PrintPartTimesSolo = true;

    if(PrintPartTimes)
    {
        printf("rand: %lf\t", (t1 - timeStart)* timeMul);
        printf("into buckets : %lf\t ", (t2 - t1) * timeMul);
        printf("merge buckets : %lf\t", (t3 - t2)* timeMul);
        printf("sort buckets : %lf\t", (t4 - t3)* timeMul);
        printf("finalize : %lf\t", (timeEnd - t4)* timeMul);

    }
    if(PrintPartTimesSolo)
    {
        printf("%lf\t", (t1 - timeStart) * timeMul);
        printf("%lf\t", (t2 - t1) * timeMul);
        printf("%lf\t", (t3 - t2) * timeMul);
        printf("%lf\t", (t4 - t3) * timeMul);
        printf("%lf\t", (timeEnd - t4) * timeMul);
    }

    delete[] Array;
    delete[] SingleBuckets;
    for (int i = 0; i < threadNum; i++)
    {
        delete[] Buckets[i];
    }
    delete[] Buckets;

    return (timeEnd - timeStart);
}

/////////////////////////////////////////////////////////////////////////////////////////

double TestTimeGoodRand(int threadNum, int size, int scheduleMode, int testSampleCount)
{
    omp_set_num_threads(threadNum);
    double* Array = (double*)malloc(sizeof(double) * size);

    int i = 0;
    unsigned short int x16v[3];
    /*
      typedef enum omp_sched_t {
        omp_sched_static = 1,
        omp_sched_dynamic = 2,
        omp_sched_guided = 3,
        omp_sched_auto = 4
      } omp_sched_t;
     */
    double timeStart = omp_get_wtime();
        if (scheduleMode == 0)
            omp_set_schedule(omp_sched_static, 1);
        if (scheduleMode == 1)
            omp_set_schedule(omp_sched_static, 10000);
        if (scheduleMode == 2)
            omp_set_schedule(omp_sched_dynamic, 1);
        if (scheduleMode == 3)
            omp_set_schedule(omp_sched_dynamic, 10000);
        if (scheduleMode == 4)
            omp_set_schedule(omp_sched_guided, 1);
        if (scheduleMode == 5)
            omp_set_schedule(omp_sched_guided, 10000);

        while (testSampleCount)
        {
            testSampleCount--;
#pragma omp parallel for private(i, x16v) schedule(runtime)
            for (i = 0; i < size; i++)
            {
                Array[i] = erand48(x16v);
            }
        }
    double timeEnd = omp_get_wtime();
    free(Array);
    return (timeEnd - timeStart);
}

int main1(int argc, char* argv[])
{
    int setThreadNum = 4;
    int size = 100000;
    if (argc >= 2)
    {
        size = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        setThreadNum = atoi(argv[2]);
    }
    const int maxThreadCount = setThreadNum;//omp_get_max_threads();
    double testTime = TestTimeGoodRand(maxThreadCount, size, 1, 8) / 8;
    int testSampleCount = (int)(1.0 / testTime) + 1;
    testSampleCount = 1;
    //printf("%lf - %d ---\n", testTime, testSampleCount);
    printf("time [ms], size=%d\tmaxThreadCount=%d\ttestSampleCount=%d\n", size, maxThreadCount, testSampleCount);
    int thCount = 0;
    for (thCount = 1; thCount <= maxThreadCount; thCount++)
    {
        printf("%d\t", thCount);
        int schMode = 0;
        for (schMode = 0; schMode <= 6; schMode++)
        {
            double time = TestTimeGoodRand(thCount, size, schMode, testSampleCount) * 1000;
            printf("%lf\t", time / testSampleCount);
        }
        printf("\n");
    }
    return 0;
}


int main2(int argc, char* argv[])
{
    int setThreadNum = 4;
    int size = 1000000;
    if (argc >= 2)
    {
        size = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        setThreadNum = atoi(argv[2]);
    }
    int testSampleCount = 1;
    const int maxThreadCount = setThreadNum;//omp_get_max_threads();
    //printf("%lf - %d ---\n", testTime, testSampleCount);
    printf("time [ms], size=%d\tmaxThreadCount=%d\ttestSampleCount=%d\n", size, maxThreadCount, testSampleCount);
    int thCount = 0;
    for (thCount = 1; thCount <= maxThreadCount; thCount++)
    {
        printf("%d\t", thCount);
        int schMode = 0;
        for (schMode = 0; schMode <= 0; schMode++)
        {
            double time = BucketSort(thCount, size, size, schMode, testSampleCount);
            //printf("%lf\t",time/testSampleCount);
        }
        printf("\n");
    }
    return 0;
}

int main_TestRandBucket(int argc, char* argv[])
{
    bool DebugPrint = true;
    int size = 1000000;
    if (argc >= 3)
    {
        size = atoi(argv[2]);
    }
    float fBucketProp = 0.10f;
    int bucketCount = size * fBucketProp;
    int threadNum = 1;
    omp_set_num_threads(threadNum);

    double* Array = new double[size];
    std::vector<double>** Buckets = new std::vector<double>*[threadNum];
    for (int t = 0; t < threadNum; t++)
    {
        Buckets[t] = new std::vector<double>[bucketCount];
        for (int b = 0; b < bucketCount; b++)
            Buckets[t][b].reserve(8);
    }
    std::vector<double>* SingleBuckets = new std::vector<double>[bucketCount];
    for (int b = 0; b < bucketCount; b++)
    {
        SingleBuckets[b].reserve(8);
    }
    std::vector<int> bucketSizes(bucketCount);

     // Create data
    CreateRandomData(Array, size);
    // DEBUG_printArray(true, size, Array);
    FillBuckets(size, Buckets, Array, bucketCount);

    MergeBuckets(bucketCount, SingleBuckets, threadNum, Buckets);

    std::vector<double>* Buc = SingleBuckets;
    std::vector<int> counter;
    counter.resize(30);
    for(int i=0; i<bucketCount; i++)
    {
        int c = Buc[i].size();
        if(c >= counter.size())
        {
            printf("ERRRRRRRRRRRRRRRR\n");

        }
        else
            counter[c]++;
    }
    for(int i=0; i<counter.size(); i++)
    {
        printf("%d\t%d\n", i, counter[i]);
    }
    printf("=============\n");
    for(int i=0; i<bucketCount; i++)
    {
       // printf("%d\t%d\n", i, Buc[i].size());
    }

    return 0;
}

int main2_Seq(int argc, char* argv[])
{
    int setThreadNum = 1;
    omp_set_num_threads(setThreadNum);

    int size = 1000000;
    if (argc >= 3)
    {
        size = atoi(argv[2]);
    }
    int step = size / 200;
    if(step < 1)
        step = 1;
    printf("main2_Seq, time [ms], size=%d\tstep=%d\n", size, step);
    for(int bucketCount = 1; bucketCount < 1.0f * size; bucketCount += std::min(step, step))
    {
        printf("%d\t", bucketCount);
        int schMode = 0;
        double time = BucketSort(setThreadNum, size, bucketCount, schMode, 1) * 1000;
        printf("%lf\t",time);
        printf("\n");
    }
    return 0;
}

int main2_Seq2(int argc, char* argv[])
{
    int setThreadNum = 1;
    omp_set_num_threads(setThreadNum);

    int size = 1000000;
    if (argc >= 3)
    {
        size = atoi(argv[2]);
    }
    int step = size / 200;
    if(step < 1)
        step = 1;

    float fBucketProp = 0.15f;
    int bucketCount = size * fBucketProp;

    omp_set_schedule(omp_sched_static, -10);
    printf("main2_Seq, time [ms], size=%d\tstep=%d\n", size, step);
    for(int s = 10; s < size; s += std::min(size, step))
    {
        int bucketCount = s * fBucketProp;
        printf("%d\t", s);
        int schMode = 0;
        double time = BucketSort(setThreadNum, s, bucketCount, schMode, 1) * 1000;
        printf("%lf\t",time);
        printf("\n");
    }
    return 0;
}

int main3_Par(int argc, char* argv[])
{
    int maxThreadCount = 4;
    int size = 1000000;
    if (argc >= 3)
    {
        size = atoi(argv[2]);
    }
    float fBucketProp = 0.15f;
    int bucketCount = size * fBucketProp;

    omp_set_schedule(omp_sched_guided, size/100);
    printf("main3_Par, time [ms], size=%d\n", size);

    for (int thCount = 1; thCount <= maxThreadCount; thCount++)
    {
        printf("%d\t", thCount);
        double time = BucketSort(thCount, size, bucketCount, 0, 1)* 1000;
        printf("%lf\t",time);
        printf("\n");
    }
    return 0;
}

int main(int argc, char* argv[])
{
    int option = 0;
    //return main2(argc, argv);
    if (argc >= 2)
    {
        option = atoi(argv[1]);
    }
    if(option == 1)
        main_TestRandBucket(argc, argv);
    if(option == 2)
        return main2_Seq(argc, argv);
    if(option == 3)
        return main3_Par(argc, argv);
    return 0;
}
