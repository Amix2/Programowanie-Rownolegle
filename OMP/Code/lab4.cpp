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
    #pragma omp parallel
    { 
        unsigned short int x16v[3];
        int myThreadNum = omp_get_thread_num();
        x16v[0] = myThreadNum;
        x16v[1] = myThreadNum >> 8;
        x16v[2] = myThreadNum >> 16;
        #pragma omp for schedule(runtime)
        for (int s = 0; s < size; s++)
        {
            Array[s] = double(erand48(x16v));
        }
    }
}

void FillBuckets(int size, std::vector<double>** Buckets, double* Array, int bucketCount)
{
#pragma omp parallel for schedule(runtime)
    for (int s = 0; s < size; s++)
    {
        std::vector<double>* myBucket = Buckets[omp_get_thread_num()];
        const double val = Array[s];
        const int valBucket = val * bucketCount;
        myBucket[valBucket].push_back(val);
       // if(myBucket[valBucket].size() > 8)
           // printf("err %d\n", myBucket[valBucket].size());
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
            mySingleBucket->insert(mySingleBucket->end(), mergeBucket->begin(), mergeBucket->end());
        }
    }
}

void SortSingleBucket(int bucketCount, std::vector<double>* SingleBuckets, std::vector<int>& bucketSizes)
{
// #pragma omp parallel for schedule(runtime)
//     for (int b = 0; b < bucketCount; b++)
//     {
//         std::vector<double>* mySingleBucket = &SingleBuckets[b];
//         std::sort(mySingleBucket->begin(), mySingleBucket->end());
//         bucketSizes[b] = mySingleBucket->size();
//     }
#pragma omp parallel for schedule(runtime)
for (int b = 0; b < bucketCount; b++)
{
    std::vector<double>* mySingleBucket = &SingleBuckets[b];
    const int size = mySingleBucket->size();
    if(size == 2)
    {
        if(mySingleBucket->at(0) > mySingleBucket->at(1))
            std::swap(mySingleBucket->at(0), mySingleBucket->at(1));
    }
    else if(size > 2)
    {
        std::sort(mySingleBucket->begin(), mySingleBucket->end());
    }
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

    int timeMul = 1000;
    omp_set_num_threads(threadNum);
    double* Array = new double[size];
    std::vector<double>** Buckets = new std::vector<double>*[threadNum];
    int bucketSizeMul = size / bucketCount;
    bucketSizeMul++;
    if(bucketSizeMul < 1)
        bucketSizeMul = 1;
    for (int t = 0; t < threadNum; t++)
    {
        Buckets[t] = new std::vector<double>[bucketCount];
        for (int b = 0; b < bucketCount; b++)
            Buckets[t][b].reserve(8 * bucketSizeMul);
    }
    std::vector<double>* SingleBuckets = new std::vector<double>[bucketCount];
    for (int b = 0; b < bucketCount; b++)
    {
        SingleBuckets[b].reserve(8 * bucketSizeMul);
    }
    std::vector<int> bucketSizes(bucketCount);

    double timeStart = omp_get_wtime();

    CreateRandomData(Array, size);

    double t1 = omp_get_wtime();

    //DEBUG_printArray(DebugPrint, size, Array);

    FillBuckets(size, Buckets, Array, bucketCount);

    double t2 = omp_get_wtime();

    //DEBUG_printBuckets(DebugPrint, threadNum, bucketCount, Buckets);

    MergeBuckets(bucketCount, SingleBuckets, threadNum, Buckets);

    double t3 = omp_get_wtime();

    //DEBUG_printSingleBucket(DebugPrint, bucketCount, SingleBuckets);

    SortSingleBucket(bucketCount, SingleBuckets, bucketSizes);

    double t4 = omp_get_wtime();

   // DEBUG_printSingleBucket(DebugPrint, bucketCount, SingleBuckets);

    ExportToSortedArray(bucketCount, bucketSizes, SingleBuckets, Array);

    double timeEnd = omp_get_wtime();

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
    if (argc >= 2)
    {
        size = atoi(argv[1]);
    }
    int bucketCount = size;
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

    FillBuckets(size, Buckets, Array, bucketCount);

    MergeBuckets(bucketCount, SingleBuckets, threadNum, Buckets);

    std::vector<double>* Buc = SingleBuckets;
    std::vector<int> counter;
    counter.resize(12);
    for(int i=0; i<bucketCount; i++)
    {
        int c = Buc[i].size();
        if(c >= counter.size())
            printf("ERRRRRRRRRRRRRRRR\n");
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
    for(int bucketCount = 1; bucketCount < 1.5f * size; bucketCount += std::min(bucketCount, step))
    {
        printf("%d\t", bucketCount);
        int schMode = 0;
        double time = BucketSort(setThreadNum, size, bucketCount, schMode, 1) * 1000;
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
    int bucketCount = size;

    printf("main3_Par, time [ms], size=%d\n", size);

    for (int thCount = 1; thCount <= maxThreadCount; thCount++)
    {
        printf("%d\t", thCount);
        double time = BucketSort(thCount, size, bucketCount, 0, 1);
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
    if(option == 2)
        return main2_Seq(argc, argv);
    if(option == 3)
        return main3_Par(argc, argv);
    return 0;
}
