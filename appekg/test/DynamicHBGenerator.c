/*

This program generates HBs dynamically by taking input from command line.

The following values can be given as input:
* number of HB 
* number of Cycle 
* duration of dach HB (ms) 
* interval between HBs (ms) 
* number of HB for each cycle

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

#include "appekg.h"

void logHeartbeats(int heartbeatID, int numberOfHBForEachCycle, int durationOfEachHB, int intervalBetweenHBs);
int msleep(long tms);
char* getHBName(int id);

int TO_MILLISECOND_FACTOR = 1000;

int main(int argc, char *argv[]) {

    if(argc != 6) {
        printf("\n");
        printf("Five integer arguments expected in the following order:\n\n");
        printf("numberOfHB numberOfCycle durationOfEachHB(ms) intervalBetweenHBs(ms) numberOfHBForEachCycle:\n\n");
        return 0;
    }

    // capture data in integer form from command line input
    int numberOfHB = atoi(argv[1]);
    int numberOfCycle = atoi(argv[2]);
    int durationOfEachHB = atoi(argv[3]);
    int intervalBetweenHBs = atoi(argv[4]);
    int numberOfHBForEachCycle = atoi(argv[5]);

    // initializing AppEKG library
    EKG_INITIALIZE(numberOfHB, 1, 106, 56, 16, 1);

    // naming HBs
    for(int i = 0; i < numberOfHB; i++){
        int id = i + 1;
        EKG_NAME_HEARTBEAT(id, getHBName(id));
    }
    // EKG_NAME_HEARTBEAT(1,"hb1");
    // EKG_NAME_HEARTBEAT(2,"hb2");
    // EKG_NAME_HEARTBEAT(3,"hb3");

    // this loop controls the number of cycle
    for(int i=0;i<numberOfCycle;i++){

        // this loop control the number of HBs
        for(int j=0;j<numberOfHB;j++){

            // starts a HB ID from 1
            int heartbeatID = j+1;

            // starts capturing HBs
            logHeartbeats(heartbeatID, numberOfHBForEachCycle, durationOfEachHB, intervalBetweenHBs);

            //interval between each HB
            usleep(TO_MILLISECOND_FACTOR * durationOfEachHB);
            //msleep(intervalBetweenHBs);
        }
    }

    // finalizing/closing AppEKG library
    EKG_FINALIZE();

    return 0;
}

char* getHBName(int id){
    int length = snprintf(NULL, 0, "%d", id);
    char* idStr = malloc(length + 1);
    snprintf(idStr, length + 1, "%d", id);

    static char hbName[4] = "hb";
    strcpy(hbName, "hb");
    strcat(hbName, idStr);

    //printf("%d - %s - %s\n", id, hbName, idStr);
    free(idStr);
    return hbName;
}

void logHeartbeats(int heartbeatID, int numberOfHBForEachCycle, int durationOfEachHB, int intervalBetweenHBs){
    
    for(int k=0;k<numberOfHBForEachCycle;k++){
        // start HB tracking
        EKG_BEGIN_HEARTBEAT(heartbeatID, 1);
        
        usleep(TO_MILLISECOND_FACTOR * durationOfEachHB);
        //msleep(durationOfEachHB);

        // end HB tracking
        EKG_END_HEARTBEAT(heartbeatID);

    }

}

// usleep is deprecated, need to use 'nanosleep' instead [need to understand how it works] 
int msleep(long tms){
    struct timespec ts;
    int ret;

    if (tms < 0){
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = tms / 1000;
    ts.tv_nsec = (tms % 1000) * 1000000;

    do {
        ret = nanosleep(&ts, &ts);
    } while (ret && errno == EINTR);

    return ret;
}


