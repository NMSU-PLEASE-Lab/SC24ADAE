/**
* Test program to see how OpenMP and PThreads behave
* in regards to thread IDs. On Ubuntu Linux, it appears
* that PThread IDs are unique for OpenMP threads, but
* OpenMP is unaware if raw PThreads are used. However,
* there may be platforms where OpenMP does not use 
* PThreads underneath, and so we may not be able to
* rely on PThread IDs always.
**/
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>

// PThread thread "main" function
void* threadFunction(void *data)
{
   printf("Pthread self   ID: %lu\n",pthread_self()%47);
   printf("OpenMP thread num: %d\n",omp_get_thread_num());
   return 0;
}

int main(int argc, char **argv)
{
   int i;
   pthread_t tid;
   // test Pthreads
   printf("Testing Pthreads...\n");
   for (i=0; i < 4; i++) {
      pthread_create(&tid,0,threadFunction,0);
   }
   sleep(1); // leave time for threads to end
   // test OpenMP
   printf("Testing OpenMP...\n");
   omp_set_num_threads(4);
   #pragma omp parallel for
   for (i=0; i < 4; i++) {
      printf("Pthread self   ID: %lu\n",pthread_self()%47);
      printf("OpenMP thread num: %d\n",omp_get_thread_num());
   }
}

