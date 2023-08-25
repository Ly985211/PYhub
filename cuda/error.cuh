#ifndef ERROR_CUH_
#define ERROR_CUH_
    #include<stdio.h>

    #define CHECK(call)                                          \
    do{                                                          \
        const cudaError_t err_code = call;                       \
        if(err_code != cudaSuccess){                             \
            printf("CUDA Error:\n");                             \
            printf("  File:      %s\n", __FILE__);               \
            printf("  Line:      %d\n", __LINE__);               \
            printf("  Error code:%d\n", err_code);               \
            printf("  Error text:%s\n", cudaGetErrorString(err_code)); \
            exit(1);                                             \
        }                                                        \
    }while(0)
#endif