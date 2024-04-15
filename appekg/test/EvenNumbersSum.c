//
// Heartbeat generator that does computations over an array
//
// The makefile compiles it with OpenMP, if you just run "./evensum" the
// OpenMP runtime will decide how many threads to use.
//
// To test as a single thread, do:
//   shell> OMP_NUM_THREADS=1 ./evensum
// 
// You can use other numbers as well, to get a known # of threads
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "appekg.h"

void generateArray();
int findEvenNumbers();
int calculateSumOfEvenNumbers(int length);

#define ARRAY_SIZE 1000
int valArray[ARRAY_SIZE];
int evenArray[ARRAY_SIZE];

int main(int argc, char **argv) 
{
    int randSeed=0;
    if (argc > 1) {
       char* end;
       randSeed = (int) strtol(argv[1],&end,10);
       if (!end || *end != '\0') {
          fprintf(stderr,"Error: provide one nonzero integer argument\n");
          return -1;
       }
    }
    // initializing AppEKG library
    EKG_INITIALIZE(3, 1, 105, 55, 15, 1);
    EKG_NAME_HEARTBEAT(1,"Gen");
    EKG_NAME_HEARTBEAT(2,"Find");
    EKG_NAME_HEARTBEAT(3,"Sum");
    // generate an array with given size of random numbers
    generateArray(randSeed);
    sleep(2); // show zero data rows getting output
    // find all even numbers
    int evenArrayLength = findEvenNumbers();
    // calculate even numbers sum
    int sum = calculateSumOfEvenNumbers(evenArrayLength);
    printf("Sum of %d randomly generated even numbers are: %d\n\n",
           evenArrayLength, sum);
    // finalizing/closing AppEKG library
    EKG_FINALIZE();
    return 0;
}

void generateArray(int randomSeed)
{
    if (randomSeed == 0)
       srand(time(0));
    else
       srand(randomSeed);
    #pragma omp parallel for
    for(int i=0; i < ARRAY_SIZE; i++) {
        EKG_BEGIN_HEARTBEAT(1, 1);
        // fill array element
        valArray[i] = rand() % 100 + 1;
        // sleep for 5ms
        usleep(1000*5); 
        // end HB tracking
        EKG_END_HEARTBEAT(1);
    }
}

int findEvenNumbers()
{
    int evenArrayLength = 0;
    #pragma omp parallel for
    for(int i=0; i < ARRAY_SIZE; i++) {
        if(valArray[i] % 2 == 0) {
            EKG_BEGIN_HEARTBEAT(2, 1);
            // computation
            evenArray[i] = valArray[i];
            evenArrayLength++;
            // sleep for 5ms
            usleep(1000*5);
            // end HB tracking
            EKG_END_HEARTBEAT(2);
        } 
    }
    return evenArrayLength;  // TODO: this is not correct!
}

int calculateSumOfEvenNumbers(int length)
{
    int sum = 0;
    #pragma omp parallel for
    for(int i=0; i < length; i++) {
        EKG_BEGIN_HEARTBEAT(3, 1);
        sum = sum + evenArray[i];
        // sleep for 5ms
        usleep(1000*5);
        // end HB tracking
        EKG_END_HEARTBEAT(3);
    }
    return sum;
}






