#include <stdio.h>
#include <string.h>

#include "appekg.h"

void printHelloWorld();

int main(void) {
    // initializing AppEKG library
    EKG_INITIALIZE(1, 1, 101, 42, 13, 1);
    EKG_NAME_HEARTBEAT(1,"Print");
    
    // printing Hello World
    printHelloWorld();

    // printting Hello World multiple times
    /*
    for(int i=0;i<10000;i++){
        printHelloWorld();
    } 
    */

    // finalizing/closing AppEKG library
    EKG_FINALIZE();

    return 0;
}

void printHelloWorld(){
    // start HB tracking
    EKG_BEGIN_HEARTBEAT(1, 1);

    printf ("Hello from your first program!\n");

    // end HB tracking
    EKG_END_HEARTBEAT(1);
}

