#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <sys/time.h>

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at %s:%d - %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Timer function
inline double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

#endif