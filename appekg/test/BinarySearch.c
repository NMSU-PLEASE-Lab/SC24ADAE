#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include<time.h>

#include "appekg.h"

int binarySearch(int arr[], int l, int r, int x);

int main(void) {
   // initializing AppEKG library
   EKG_INITIALIZE(1, 1, 102, 52, 12, 1);
   EKG_NAME_HEARTBEAT(1,"Bin");

   // declaring array size values
   int n = 32767;
   int arr[n];
   for(int i=0;i<n;i++)
   {
       arr[i] = i;
   }


   // finding random x value for search operation
   srand(time(0));
   int x = rand() % n;

   // performing binary search operation and storing the result
   int result = binarySearch(arr, 0, n - 1, x);

   // printing the result
   if(result == -1)
       printf("Element %d is not present in array", x);
   else
       printf("Element %d is present at index %d", x, result);

   // finalizing/closing AppEKG library
   EKG_FINALIZE();

   return 0;
}

int binarySearch(int arr[], int l, int r, int x) 
{
   // start HB tracking
   EKG_BEGIN_HEARTBEAT(1, 1);

   if (r >= l)
    {
       int mid = l + (r - l) / 2;
       // if the element is present at the middle itself
       if (arr[mid] == x){
       // end HB tracking
       EKG_END_HEARTBEAT(1);
           return mid;
    }
 
   // if element is smaller than mid, then it can only be present in left subarray
   if (arr[mid] > x)
   {
       // end HB tracking
       EKG_END_HEARTBEAT(1);
       return binarySearch(arr, l, mid - 1, x);
   }
            
      EKG_END_HEARTBEAT(1);
      // else the element can only be present in right subarray
      return binarySearch(arr, mid + 1, r, x);
   }
    
    // End HB tracking
    EKG_END_HEARTBEAT(1);

    // We reach here when element is not present in array
    return -1;
}

