#include <stdio.h>
#include <omp.h>
#include <stdlib.h>   

double TestTime(int threadNum, int size)
{
    
    omp_set_num_threads(threadNum);
    double* Array = malloc(sizeof(double) * size);

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
    double* Array = malloc(sizeof(double) * size);

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
    double* Array = malloc(sizeof(double) * size);

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
    double* Array = malloc(sizeof(double) * size);

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
    // if(scheduleMode == 0)
    //     omp_set_schedule(omp_sched_static,1);
    // if(scheduleMode == 1)
    //     omp_set_schedule(omp_sched_static,10000);
    // if(scheduleMode == 2)
    //     omp_set_schedule(omp_sched_dynamic,1);
    // if(scheduleMode == 3)
    //     omp_set_schedule(omp_sched_dynamic,10000);
    // if(scheduleMode == 4)
    //     omp_set_schedule(omp_sched_guided,1);
    // if(scheduleMode == 5)
    //     omp_set_schedule(omp_sched_guided,10000);

    double timeStart = omp_get_wtime();
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
    int testSampleCount = (int)(0.5 / testTime ) + 1;
    //printf("%lf - %d ---\n", testTime, testSampleCount);
    printf("time [ms], size=%d\tmaxThreadCount=%d\ttestSampleCount=%d\n", size, maxThreadCount, testSampleCount);
    int thCount = 0;
    for(thCount=1; thCount<=maxThreadCount;  thCount++)
    {
        printf("%d\t",thCount);
        int schMode=0;
        for(schMode=0; schMode<=5; schMode++)
        {
            double time = TestTimeGoodRand(thCount, size, schMode, testSampleCount)*1000;
            printf("%lf\t",time/testSampleCount);
        }
        printf("\n");
    }
    return 0;
}

int main (int argc, char * argv[])
{
    return main1(argc, argv);
}
