#ifndef SPS_CU_H
#define SPS_CU_H

#include <cuda_runtime.h>

#include "constants.cu.h"

// template <int B, int Q>
// __device__ inline void threadReduce(int32_t *shd_mem, uint32_t idx) {
//     unsigned int tid = threadIdx.x;
//     int acc = 0;
// #pragma unroll
//     for (int i = 0; i < Q; i++) {
//         int tmp = shd_mem[idx + i * B + tid];
//         acc = acc + tmp;
//     }
//     // stores acc in the auxiliary array
//     aux_mem[idx + 4 * B + tid] = acc;
// }

// shd_mem is a pointer to an array in shared memory. It has size Q * B.
// idx is the id of the part of the shared memory we start scanning.
template <int B, int Q>
__device__ inline void threadScan(int32_t *shd_mem) {
    unsigned int tid = threadIdx.x;
    int acc = 0;
#pragma unroll
    for (int = 0; i < Q; i++) {
        int tmp = shd_mem[i * B + tid];
        acc = acc + tmp;
        shd_mem[i * B + tid] = acc;
    }
}

template <int B, int Q>
__device__ inline void threadAdd(int32_t *shd_mem, int32_t *shd_buf) {
    unsigned int tid = threadIdx.x;
#pragma unroll
    if (tid != 0) {
        int32_t tmp = shd_buf[tid - 1];
        for (int = 0; i < Q; i++) {
            shd_mem[i * B + tid] = shd_mem[i * B + tid] + tmp;
        }
    }
}

// Function that uses collectively scans a warp of elements in a shared buffer.
// shd_buf is a pointer to an array in shared memory. It has size B <= 1024.
// idx is the id of the part of the shared memory we start scanning.
//
// Each thread in the warp performs a scan across the shared buffer. The
// result is stored in the shared buffer.
__device__ inline int32_t warpScan(volatile int32_t *shd_buf, uint32_t idx,
                                   int32_t) {
    uint32_t lane = idx & (WARP - 1);  // WARP
    int n = WARP;
    int k = lgWARP;

#pragma unroll
    for (int d = 0; d < k; d++) {
        int h = 1 << d;
        if (lane >= h) {
            shd_buf[idx] = shd_buf[idx - h] + shd_buf[idx];
        }
    }

    int32_t res = shd_buf[idx];
    return res;
}

// Function that performs a per block scan collectively across all threads in
// the block.
// shd_buf is a pointer to an array of size B.
// idx is the id of the part of the shared memory we start scanning.
//
// Each thread copy the final value of the scan to the shd_buf. Then
// we perform a parallel scan across the shd_buf like in assignment 2.
template <int B, int Q>
__device__ inline int32_t blockScan(volatile int32_t *shd_buf, uint32_t idx) {
    uint32_t lane = idx & (WARP - 1);
    uint32_t warpid = idx >> lgWARP;

    // scan at warp level
    int64_t warp_res = warpScan(shd_buf, idx);
    __syncthreads();

    // store the results of each warp scan in the first part of the shared
    // memory
    if (lane == WARP - 1) {
        shd_buf[warpid] = warp_res;
    }
    __syncthreads();

    // scan the first warp again.
    if (warpid == 0) {
        warpScan(shd_buf, idx);
    }

    // accumulate the results from the previous step
    if (warpid > 0) {
        int32_t tmp = shd_buf[warpid - 1];
        warp_res = warp_res + tmp;
    }

    return warp_res;
}

// aux_mem is an index to an array of global memory.
// flag_mem is an array with flags corresponding to which aux_mem indices
//     are valid.
// aux_size is the size of the aux_mem array Performs a scan a across aux
//     array and stores the result in aux_mem.
//
// The function sequentially scans all elements in the aux array and stores
// the result in aux_mem. For each position in the aux_mem, wait to progress
// until the flag_mem is set to 1. This is done by checking the flag_mem
// array in a loop. Once the flag_mem is set to 1, thread can update the
// aux_mem array.
template <int B, int Q>
__device__ inline void blockLevelScan(int32_t *aux_mem, int32_t *flag_mem,
                                      uint32_t aux_size) {
    uint32_t tid = threadIdx.x;
    if (tid == 0) {
        // scan the aux array
        for (int i = 1; i < aux_size; i++) {
            // wait for the flag to be set
            while (flag_mem[i] == 0)
                ;
            // update the aux array
            aux_mem[i] = aux_mem[i] + aux_mem[i - 1];
        }
    }
}

#endif /* SPS_CU_H */