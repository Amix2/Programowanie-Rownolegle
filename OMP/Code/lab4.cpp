#include <stdio.h>
#include <omp.h>
#include <stdlib.h>   
#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <vector> 
#include <algorithm>

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
    if (scheduleMode != 6)
    {
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
    }
    else {
        while (testSampleCount)
        {
            testSampleCount--;
#pragma omp parallel for private(i, x16v) schedule(runtime)
            for (i = 0; i < size; i++)
            {
                Array[i] = erand48(x16v);
            }
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

void CreateRandomData(double* Array, int size)
{
    unsigned short int x16v[3];
#pragma omp parallel for private(x16v) schedule(runtime)
    for (int s = 0; s < size; s++)
    {
        int myThreadNum = omp_get_thread_num();
        x16v[0] = myThreadNum;
        x16v[1] = s;
        x16v[2] = myThreadNum + size;
        Array[s] = double(erand48(x16v));
    }
}

void FillBuckets(int size, std::vector<double>** Buckets, double* Array, int bucketCount)
{
#pragma omp parallel for schedule(runtime)
    for (int s = 0; s < size; s++)
    {
        int myThreadNum = omp_get_thread_num();
        std::vector<double>* myBucket = Buckets[myThreadNum];

        double val = Array[s];
        int valBucket = val * bucketCount;
        myBucket[valBucket].push_back(val);
    }
}

void MergeBuckets(int bucketCount, std::vector<double>* SingleBuckets, int threadNum, std::vector<double>** Buckets)
{
    // Merge buckets   
#pragma omp parallel for firstprivate(threadNum) schedule(runtime)
    for (int b = 0; b < bucketCount; b++)
    {
        std::vector<double>* mySingleBucket = &SingleBuckets[b];
        for (int t = 0; t < threadNum; t++)
        {
            std::vector<double>* mergeBucket = &Buckets[t][b];
            for (int x = 0; x < mergeBucket->size(); x++)
            {
                mySingleBucket->push_back(mergeBucket->at(x));
            }
        }
    }
}

void SortSingleBucket(int bucketCount, std::vector<double>* SingleBuckets, std::vector<int>& bucketSizes)
{
#pragma omp parallel for schedule(runtime)
    for (int b = 0; b < bucketCount; b++)
    {
        std::vector<double>* mySingleBucket = &SingleBuckets[b];
        std::sort(mySingleBucket->begin(), mySingleBucket->end());
        bucketSizes[b] = mySingleBucket->size();
    }
}

void ExportToSortedArray(int bucketCount, std::vector<int>& bucketSizes, std::vector<double>* SingleBuckets, double* Array)
{
    for (int b = 1; b < bucketCount; b++)
        bucketSizes[b] += bucketSizes[b - 1];

#pragma omp parallel for schedule(runtime)
    for (int b = 0; b < bucketCount; b++)
    {
        int insertId = b == 0 ? 0 : bucketSizes[b - 1];
        for (int i = 0; i < SingleBuckets[b].size(); i++)
        {
            Array[insertId + i] = SingleBuckets[b][i];
        }
    }
}

void DEBUG_printArray(bool DebugPrint, int size, double* Array)
{
    // DEBUG print array
    if (DebugPrint)
    {
        printf("DEBUG print array\n");
        for (int j = 0; j < size; j++)
        {
            printf("%lf, ", Array[j]);
        }
        printf("\n");
    }
}

void DEBUG_printBuckets(bool DebugPrint, int threadNum, int bucketCount, std::vector<double>** Buckets)
{
    // DEBUG print buckets
    if (DebugPrint)
    {
        printf("DEBUG print buckets\n");
        for (int i = 0; i < threadNum; i++)
        {
            printf("\n");
            for (int j = 0; j < bucketCount; j++)
            {
                for (int x = 0; x < Buckets[i][j].size(); x++)
                    printf("%lf,", Buckets[i][j][x]);
                printf(" | ");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void DEBUG_printSingleBucket(bool DebugPrint, int bucketCount, std::vector<double>* SingleBuckets)
{
    // DEBUG print not sorted bucket
    if (DebugPrint)
    {
        printf("DEBUG single bucket\n");
        for (int j = 0; j < bucketCount; j++)
        {
            for (int x = 0; x < SingleBuckets[j].size(); x++)
            {
                printf("%lf,", SingleBuckets[j][x]);
            }
            printf(" | ");
        }
        printf("\n");
    }
}

double BucketSort(int threadNum, int size, int bucketCount, int scheduleMode, int testSampleCount)
{
    bool DebugPrint = false;
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
    std::vector<int> bucketSizes(bucketCount);

    double timeStart = omp_get_wtime();

    // Create data
    CreateRandomData(Array, size);

    double t1 = omp_get_wtime();
    printf("rand: %lf\t", (t1 - timeStart));
    DEBUG_printArray(DebugPrint, size, Array);

    // Put into buckets
    FillBuckets(size, Buckets, Array, bucketCount);

    double t2 = omp_get_wtime();
    printf("into buckets : %lf\t ", (t2 - t1));

    DEBUG_printBuckets(DebugPrint, threadNum, bucketCount, Buckets);

    MergeBuckets(bucketCount, SingleBuckets, threadNum, Buckets);

    double t3 = omp_get_wtime();
    printf("merge buckets : %lf\t", (t3 - t2));

    DEBUG_printSingleBucket(DebugPrint, bucketCount, SingleBuckets);

    SortSingleBucket(bucketCount, SingleBuckets, bucketSizes);

    double t4 = omp_get_wtime();
    printf("sort buckets : %lf\t", (t4 - t3));

    DEBUG_printSingleBucket(DebugPrint, bucketCount, SingleBuckets);

    ExportToSortedArray(bucketCount, bucketSizes, SingleBuckets, Array);

    double timeEnd = omp_get_wtime();
    printf("finalize : %lf\t", (timeEnd - t4));

    // DEBUG print sorted array
    if (DebugPrint)
    {
        printf("DEBUG print sorted array\n");
        for (int j = 0; j < size; j++)
        {
            printf("%lf, ", Array[j]);
        }
        printf("\n");
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
    const int maxThreadCount = setThreadNum;//omp_get_max_threads();
    double testTime = TestTimeGoodRand(maxThreadCount, size, 1, 8) / 8;
    int testSampleCount = (int)(1.0 / testTime) + 1;
    //printf("%lf - %d ---\n", testTime, testSampleCount);
    printf("time [ms], size=%d\tmaxThreadCount=%d\ttestSampleCount=%d\n", size, maxThreadCount, testSampleCount);
    int thCount = 0;
    for (thCount = 1; thCount <= maxThreadCount; thCount++)
    {
        printf("%d\t", thCount);
        int schMode = 0;
        for (schMode = 0; schMode <= 0; schMode++)
        {
            double time = BucketSort(thCount, size, size, schMode, testSampleCount) * 1000;
            //printf("%lf\t",time/testSampleCount);
        }
        printf("\n");
    }
    return 0;
}

int main(int argc, char* argv[])
{
    return main2(argc, argv);
}
