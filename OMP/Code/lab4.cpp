#include <stdio.h>
#include <omp.h>
#include <stdlib.h>   
#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <vector> 
#include <algorithm>

double TestTime(int threadNum, int size)
{
    
    omp_set_num_threads(threadNum);
    double* Array = (double*)malloc(sizeof(double) * size);

    double timeStart = omp_get_wtime();
    #pragma omp parallel
    {
        int myThreadNum = omp_get_thread_num();
        int threadCount = omp_get_num_threads();
        int i=0;
        #pragma omp for
        for(i=myThreadNum; i<size; i+=threadCount)
        {
            Array[i] = i / 1234.0;
        }
        
    }
    double timeEnd = omp_get_wtime();
    free(Array);
    return (timeEnd - timeStart);
}

double TestTime2(int threadNum, int size)
{
    
    omp_set_num_threads(threadNum);
    double* Array = (double*)malloc(sizeof(double) * size);

    double timeStart = omp_get_wtime();
    int i=0;
    #pragma omp parallel for private(i)
    for(i=0; i<size; i++)
    {
        Array[i] = i / 1234.0;
    }
    double timeEnd = omp_get_wtime();
    free(Array);
    return (timeEnd - timeStart);
}

double TestTimeBadRand(int threadNum, int size)
{
    
    omp_set_num_threads(threadNum);
    double* Array = (double*)malloc(sizeof(double) * size);

    double timeStart = omp_get_wtime();
    int i=0;
    #pragma omp parallel for private(i)
    for(i=0; i<size; i++)
    {
        Array[i] = (double) random () / (double) ( RAND_MAX );
    }
    double timeEnd = omp_get_wtime();
    free(Array);
    return (timeEnd - timeStart);
}

double TestTimeGoodRand(int threadNum, int size, int scheduleMode, int testSampleCount)
{
    
    omp_set_num_threads(threadNum);
    double* Array = (double*)malloc(sizeof(double) * size);

    int i=0;
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
    if(scheduleMode != 6)
    {
        if(scheduleMode == 0)
            omp_set_schedule(omp_sched_static,1);
        if(scheduleMode == 1)
            omp_set_schedule(omp_sched_static,10000);
        if(scheduleMode == 2)
            omp_set_schedule(omp_sched_dynamic,1);
        if(scheduleMode == 3)
            omp_set_schedule(omp_sched_dynamic,10000);
        if(scheduleMode == 4)
            omp_set_schedule(omp_sched_guided,1);
        if(scheduleMode == 5)
            omp_set_schedule(omp_sched_guided,10000);

        while(testSampleCount)
        {
            testSampleCount--;
            #pragma omp parallel for private(i, x16v) schedule(runtime)
            for(i=0; i<size; i++)
            {
                Array[i] = erand48(x16v);
            }
        }
    }
    else{
        while(testSampleCount)
        {
            testSampleCount--;
            #pragma omp parallel for private(i, x16v) schedule(static)
            for(i=0; i<size; i++)
            {
                Array[i] = erand48(x16v);
            }
        }
    }
    double timeEnd = omp_get_wtime();
    free(Array);
    return (timeEnd - timeStart);
}

int main1 (int argc, char * argv[])
{
    int setThreadNum = 4;
    int size = 100000;
    if(argc>=2)
    {
        size = atoi(argv[1]);
    }
    if(argc>=3)
    {
        setThreadNum = atoi(argv[2]);
    }
    const int maxThreadCount = setThreadNum;//omp_get_max_threads();
    double testTime = TestTimeGoodRand(maxThreadCount, size, 1, 8)/8;
    int testSampleCount = (int)(1.0 / testTime ) + 1;
    testSampleCount = 1;
    //printf("%lf - %d ---\n", testTime, testSampleCount);
    printf("time [ms], size=%d\tmaxThreadCount=%d\ttestSampleCount=%d\n", size, maxThreadCount, testSampleCount);
    int thCount = 0;
    for(thCount=1; thCount<=maxThreadCount;  thCount++)
    {
        printf("%d\t",thCount);
        int schMode=0;
        for(schMode=0; schMode<=6; schMode++)
        {
            double time = TestTimeGoodRand(thCount, size, schMode, testSampleCount)*1000;
            printf("%lf\t",time/testSampleCount);
        }
        printf("\n");
    }
    return 0;
}

double BucketSort(int threadNum, int size, int scheduleMode, int testSampleCount)
{
    bool DebugPrint = false;
    omp_set_num_threads(threadNum);
    double* Array = (double*)malloc(sizeof(double) * size);
    std::vector<double>** Buckets = new std::vector<double>*[threadNum];
    std::vector<double>* SingleBuckets = new std::vector<double>[size];
    for(int i=0; i<threadNum; i++)
    {
        Buckets[i] = new std::vector<double>[size];
        for(int j=0; j<size; j++)
            Buckets[i][j].reserve(8);
    }

    double timeStart = omp_get_wtime();

    // Create data
    unsigned short int x16v[3];
    #pragma omp parallel for private(x16v, i) schedule(static)
    for(int i=0; i<size; i++)
    {
        Array[i] = erand48(x16v);
    }

    double t1 = omp_get_wtime();
    printf("rand: %lf\t", (t1 - timeStart));

    // Put into buckets
    #pragma omp parallel for private(threadNum) schedule(static)
    for(i=0; i<size; i++)
    {
        int myThreadNum = omp_get_thread_num();
       // int myMinId = int(float(myThreadNum) / threadNum * size);
        //int myMaxId = int(float(myThreadNum+1) / threadNum * size);
        std::vector<double>* myBucket = Buckets[myThreadNum];

        double val = Array[i];
        int valBucket = val * size;
        myBucket[valBucket].push_back(val);
    }

    double t2 = omp_get_wtime();
    printf("into buckets : %lf\t ", (t2 - t1));

    // DEBUG print buckets
    if(DebugPrint)
    {
        printf("DEBUG print buckets\n");
        for(int i=0; i<threadNum; i++)
        {
            printf("\n");
            for(int j=0; j<size; j++)
            {
                for(int x=0; x<Buckets[i][j].size(); x++)
                    printf("%lf,", Buckets[i][j][x]);
                printf(" | ");
            }
            printf("\n");
        }
        printf("\n");
    }
    // Merge buckets    TODO rowlonegle 
    //#pragma omp parallel for private(threadNum) schedule(static)
    for(i=0; i<size; i++)
    {
        std::vector<double>* mySingleBucket = &SingleBuckets[i];
        for(int j=0; j<threadNum; j++)
        {
            std::vector<double>* mergeBucket = &Buckets[j][i];
            for(int x=0; x<mergeBucket->size(); x++)
            {
                mySingleBucket->push_back(mergeBucket->at(x));
            }
        }
    }

    double t3 = omp_get_wtime();
    printf("merge buckets : %lf\t", (t3 - t2));

    // DEBUG print not sorted bucket
    if(DebugPrint)
    {
        printf("DEBUG single bucket\n");
        for(int j=0; j<size; j++)
        {
            for(int x=0; x<SingleBuckets[j].size(); x++)
            {
                printf("%lf,", SingleBuckets[j][x]);
            }
            printf(" | ");
        }
        printf("\n");
    }

    #pragma omp parallel for private(threadNum) schedule(static)
    for(i=0; i<size; i++)
    {
        std::vector<double>* mySingleBucket = &SingleBuckets[i];
        std::sort(mySingleBucket->begin(), mySingleBucket->end());
    }

    double t4 = omp_get_wtime();
    printf("sort buckets : %lf\t", (t4 - t3));

    // DEBUG print sorted bucket
    if(DebugPrint)
    {
        printf("DEBUG single bucket\n");
        for(int j=0; j<size; j++)
        {
            for(int x=0; x<SingleBuckets[j].size(); x++)
            {
                printf("%lf,", SingleBuckets[j][x]);
            }
            printf(" | ");
        }
        printf("\n");
    }
    double timeEnd = omp_get_wtime();
 
    //free(Array);
    //delete Buckets; // TODO better delete 
   // delete SingleBuckets;
    return (timeEnd - timeStart);
}

int main2 (int argc, char * argv[])
{
    int setThreadNum = 4;
    int size = 100000;
    if(argc>=2)
    {
        size = atoi(argv[1]);
    }
    if(argc>=3)
    {
        setThreadNum = atoi(argv[2]);
    }
    const int maxThreadCount = setThreadNum;//omp_get_max_threads();
    double testTime = TestTimeGoodRand(maxThreadCount, size, 1, 8)/8;
    int testSampleCount = (int)(1.0 / testTime ) + 1;
    //printf("%lf - %d ---\n", testTime, testSampleCount);
    printf("time [ms], size=%d\tmaxThreadCount=%d\ttestSampleCount=%d\n", size, maxThreadCount, testSampleCount);
    int thCount = 0;
    for(thCount=1; thCount<=maxThreadCount;  thCount++)
    {
        printf("%d\t",thCount);
        int schMode=0;
        for(schMode=0; schMode<=0; schMode++)
        {
            double time = BucketSort(thCount, size, schMode, testSampleCount)*1000;
            //printf("%lf\t",time/testSampleCount);
        }
        printf("\n");
    }
    return 0;
}

int main (int argc, char * argv[])
{

    return main2(argc, argv);
}
