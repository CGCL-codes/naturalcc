#include <stdlib.h>

int null_dereference(void) {
    int *ptr = 0;
    return *ptr;
}

int memory_leak(void) {
    int *ptr = (int *)malloc(20);
    ptr[0] = 1;
    return 0;
