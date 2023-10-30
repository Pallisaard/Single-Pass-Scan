#ifndef SPS_CU_H
#define SPS_CU_H

#include <cuda_runtime.h>

#include "constants.cu.h"

#define X 0
#define A 1
#define P 2
#ifndef Q
#define Q 4
#endif
#ifndef B
#define B 1024
#endif

// shd_mem is a pointer to an array in shared memory. It has size Q * B.
// idx is the id of the part of the shared memory we start scanning.
__device__ inline void threadScan(int32_t* shd_mem, volatile int32_t* shd_buf, uint32_t tid) {
    // uint32_t B = blockDim.x;
    int acc = 0;
#pragma unroll
    for (int i = 0; i < Q; i++) {
        int tmp = shd_mem[tid * Q + i];
        acc = acc + tmp;
        shd_mem[tid * Q + i] = acc;
    }
    shd_buf[tid] = acc;
	__syncthreads();
}

__device__ inline void threadAdd(int32_t* shd_mem, volatile int32_t* shd_buf, uint32_t tid) {
    // uint32_t B = blockDim.x;
    if (tid != 0) {
        int32_t tmp = shd_buf[tid - 1];
#pragma unroll
        for (int i = 0; i < Q; i++) {
            shd_mem[tid * Q + i] = shd_mem[tid * Q + i] + tmp;
        }
    }
	__syncthreads();
}

__device__ inline void threadAddVal(int32_t* shd_mem, int32_t val, uint32_t tid, uint32_t dynID) {
    // uint32_t B = blockDim.x;
#pragma unroll
    for (int i = 0; i < Q; i++) {
        shd_mem[i * B + tid] = shd_mem[i * B + tid] + val;
        // printf("dynID == %d, tid == %d, Q = %d, val == %d\n", dynID, tid, i, shd_mem[i * B + tid]);
    }
	__syncthreads();
}

// Function that uses collectively scans a warp of elements in a shared buffer.
// shd_buf is a pointer to an array in shared memory. It has size B <= 1024.
// idx is the id of the part of the shared memory we start scanning.
//
// Each thread in the warp performs a scan across the shared buffer. The
// result is stored in the shared buffer.
__device__ inline int32_t warpScan(volatile int32_t* shd_buf, uint32_t tid) {
    uint32_t lane = tid & (WARP - 1);  // WARP
    int k = lgWARP;

#pragma unroll
    for (int d = 0; d < k; d++) {
        int h = 1 << d;
        if (lane >= h) {
            shd_buf[tid] = shd_buf[tid - h] + shd_buf[tid];
        }
    }

    int32_t res = shd_buf[tid];
    return res;
}

// Function that performs a per block scan collectively across all threads in
// the block.
// shd_buf is a pointer to an array of size B.
// idx is the id of the part of the shared memory we start scanning.
//
// Each thread copy the final value of the scan to the shd_buf. Then
// we perform a parallel scan across the shd_buf like in assignment 2.
__device__ inline int32_t blockScan(volatile int32_t* shd_buf, uint32_t tid) {
    // uint32_t B = blockDim.x;
    uint32_t lane = tid & (WARP - 1);
    uint32_t warpid = tid >> lgWARP;

    // step 1
    // scan at warp level
    int64_t warp_res = warpScan(shd_buf, tid);
    __syncthreads();

    // step 2
    // store the results of each warp scan in the first part of the shared
    // memory
    if (lane == (WARP - 1)) {
        shd_buf[warpid] = warp_res;
    }
    __syncthreads();

    // step 3
    // scan the first warp again.
    if (warpid == 0) {
        warpScan(shd_buf, tid);
    }
    __syncthreads();

    // step 4
    // accumulate the results from the previous step
    if (warpid > 0) {
        warp_res = shd_buf[warpid - 1] + warp_res;
    }
    __syncthreads();

    // step 5
    shd_buf[tid] = warp_res;

    __syncthreads();
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
__device__ inline void blockLevelScan(int32_t* aux_mem, int32_t* flag_mem,
                                      uint32_t aux_size, uint32_t tid) {
    // uint32_t B = blockDim.x;
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
	__syncthreads();
}

// device function for a lookback scan method.
__device__ inline int32_t lookbackScan(volatile int32_t* agg_mem,
								    volatile int32_t* pref_mem,
                                    volatile uint32_t* flag_mem,
									int32_t* shr_mem, uint32_t dyn_idx,
									uint32_t tid) {
    // uint32_t B = blockDim.x;

	// Handle lookback differently depending on dynamic id.
    if (tid == B - 1 && dyn_idx == 0) {
		int32_t agg_val = shr_mem[(Q - 1) * B + tid];
		agg_mem[dyn_idx] = agg_val;
        pref_mem[dyn_idx] = agg_val;
        __threadfence();
        flag_mem[dyn_idx] = P;

    } else if (tid == B - 1 && dyn_idx > 0) {
		int32_t agg_val = shr_mem[(Q - 1) * B + tid];
        agg_mem[dyn_idx] = agg_val;
        __threadfence();
        flag_mem[dyn_idx] = A;

        int32_t grab_id = dyn_idx - 1;
        while (flag_mem[grab_id] != P) {
            if (flag_mem[grab_id] == A && grab_id >= 0) {
				agg_val = agg_mem[grab_id] + agg_val;
                grab_id--;
            }
        }

		pref_mem[dyn_idx] = agg_mem[dyn_idx] + pref_mem[grab_id];
        __threadfence();
        flag_mem[dyn_idx] = P;
    }

	int32_t prefix = pref_mem[dyn_idx] - agg_mem[dyn_idx];

	__threadfence();  // it might work without
	__syncthreads();  // also might work without
    return prefix;
}

/**
 * Helper function that copies `Q` input elements per thread from
 *   global to shared memory, in a way that optimizes spatial locality,
 *
 * `glb_offs` is the offset in global-memory array `d_inp`
 *    from where elements should be read.
 * `d_inp` is the input array stored in global memory
 * `N` is the length of `d_inp`
 * `ne` is the neutral element of `T` (think zero). In case
 *    the index of the element to be read by the current thread
 *    is out of range, then place `ne` in shared memory instead.
 * `shmem_inp` is the shared memory. It has size
 *     `blockDim.x*CHUNK*sizeof(T)`, where `blockDim.x` is the
 *     size of the CUDA block. `shmem_inp` should be filled from
 *     index `0` to index `blockDim.x*CHUNK - 1`.
 *
 * As such, a CUDA-block B of threads executing this function would
 *   read `Q*B` elements from global memory and store them to
 *   (fast) shared memory, in the same order in which they appear
 *   in global memory, but making sure that consecutive threads
 *   read consecutive elements of `d_inp` in a SIMD instruction.
 **/
__device__ inline void copyFromGlb2ShrMem(int32_t glb_offs, const uint32_t N,
                                          int32_t ne, int32_t* d_inp,
                                          volatile int32_t* shmem_inp,
                                          uint32_t tid) {
#pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        // uint32_t loc_ind = blockDim.x * i + tid;
        uint32_t loc_ind = blockDim.x * i + tid;
        uint32_t glb_ind = glb_offs + loc_ind;
        uint32_t elm = ne;
        if (glb_ind < N) {
            elm = d_inp[glb_ind];
        }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads();  // leave this here at the end!
}

/**
 * This is very similar with `copyFromGlb2ShrMem` except
 * that you need to copy from shared to global memory, so
 * that consecutive threads write consecutive indices in
 * global memory in the same SIMD instruction.
 * `glb_offs` is the offset in global-memory array `d_out`
 *    where elements should be written.
 * `d_out` is the global-memory array
 * `N` is the length of `d_out`
 * `shmem_red` is the shared-memory of size
 *    `blockDim.x*Q*sizeof(T)`
 */
__device__ inline void copyFromShr2GlbMem(int32_t glb_offs, const uint32_t N,
                                          int32_t* d_out,
                                          int32_t* shmem_red,
                                          uint32_t tid) {
#pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        // uint32_t loc_ind = blockDim.x * i + tid;
        uint32_t loc_ind = B * i + tid;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            uint32_t elm = (shmem_red[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads();  // leave this here at the end!
}

__device__ inline int32_t getDynID(int32_t* IDAddr, uint32_t tid) { 
    __shared__ int32_t dynID;
	int32_t retDynID = 0;
    if (tid==0){
        dynID = atomicAdd(IDAddr, 1);
    }
    __syncthreads();
	retDynID = dynID;
	return retDynID;
}

/**
 * Naive memcpy kernel, for the purpose of comparing with
 * a more "realistic" bandwidth number.
 */
__global__ void naiveMemcpy(int* d_out, int* d_inp, const uint32_t N) {
    // uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = blockIdx.x * B + threadIdx.x;
    if (gid < N) {
        d_out[gid] = d_inp[gid];
    }
	__syncthreads();
}

/**
 * A single pass scan kernel that uses the three arrays described i
 * Merrill & Garland's paper.
*/
__global__ void SinglePassScanKernel2(int32_t *d_in, int32_t* d_out,
									  const size_t N, int32_t* IDAddr,
									  volatile uint32_t* flagArr,
									  volatile int32_t* aggrArr,
									  volatile int32_t* prefixArr) {
	// Step 1 get a dynamic id
	int32_t tid = threadIdx.x;
	uint32_t num_blocks = gridDim.x;
	int32_t dynID = getDynID(IDAddr, tid);
	int32_t globaloffset = dynID * B * Q;
	__syncthreads();

	// Step 2 copy the memory the block will scan into shared memory.
	__shared__ int32_t blockShrMem[Q * B];
	volatile __shared__ int32_t blockShrBuf[B];
	copyFromGlb2ShrMem(globaloffset, N, 0, d_in, blockShrMem, tid);

	// #define testid 1
	#ifdef testid
	if (tid == 0 && dynID == testid) {
		printf("glbMem initial\n");
		for (int i = 0; i < N; i++) {
			printf("%d ", d_in[i]);
		}
		printf("\n");
	}

	if (tid == 0 && dynID == testid) {
		printf("shrMem after load\n");
		for (int i = 0; i < B * Q; i++) {
			printf("%d ", blockShrMem[i]);
		}
		printf("\n");
	}

	__syncthreads();
	#endif

	// Step 3 Do the scan on the block
	// First scan each thread
	threadScan(blockShrMem, blockShrBuf, tid);

	#ifdef testid
	if (tid == 0 && dynID == testid) {
		printf("shrBuf after thread scan\n");
		for (int i = 0; i < B; i++) {
			printf("%d ", blockShrBuf[i]);
		}
		printf("\n");
	}
	__syncthreads();
	#endif

	// Do the scan on the block level
	blockScan(blockShrBuf, tid);

	#ifdef testid
	if (tid == 0 && dynID == testid) {
		printf("shrBuf after block scan\n");
		for (int i = 0; i < B; i++) {
			printf("%d ", blockShrBuf[i]);
		}
		printf("\n");
	}
	__syncthreads();
	#endif

	// Save the result in shrmem.
	threadAdd(blockShrMem, blockShrBuf, tid);

	#ifdef testid
	if (tid == 0 && dynID == testid) {
		printf("shrMem after thread add\n");
		for (int i = 0; i < Q * B; i++) {
			printf("%d ", blockShrMem[i]);
		}
		printf("\n");
	}
	__syncthreads();
	#endif

	// Step 4 use lookback scan to find the inclusive prefix value
	int32_t prefix = lookbackScan(aggrArr, prefixArr, flagArr, blockShrMem, dynID, tid);

	// Step 5 Sum the prefix into the scan
	threadAddVal(blockShrMem, prefix, tid, dynID);

	#ifdef testid
	if (tid == 0 && dynID == testid) {
		printf("shrMem after thread add val\n");
		for (int i = 0; i < Q * B; i++) {
			printf("%d ", blockShrMem[i]);
		}
		printf("\n");
	}
	__syncthreads();
	#endif

	// Step 6 Copy the result into global memory

	copyFromShr2GlbMem(globaloffset, N, d_out, blockShrMem, tid);

	#ifdef testid
	if (tid == 0 && dynID == testid) {
		printf("glbMem final after load\n");
		for (int i = 0; i < N; i++) {
			printf("%d ", d_out[i]);
		}
		printf("\n");
	}
	__syncthreads();
	#endif

}

/**
 * Single pass scan kernel using a naive auxiliary thread to sum up
 * aggregates.
*/
__global__ void SinglePassScanKernel1(int32_t* d_in, int32_t* d_out,
                                      const size_t N, int32_t* IDAddr,
                                      volatile uint32_t* flagArr, volatile int32_t* aggrArr,
                                      volatile int32_t* prefixArr) {
    // Step 1 get a dynamic id
    int32_t tid = threadIdx.x;
	uint32_t num_blocks = gridDim.x;
    int32_t dynID = getDynID(IDAddr, tid);
	// if (tid == 0) {
	// 	 printf("block_id == %d, dynID == %d\n", blockIdx.x, dynID);
	// }
	// int B = blockDim.x;
	__syncthreads();
    // If the first dynamic id, of -1 then we are the prefix block instead.
    // an optimisation might be to let id 0 do it, but it still calculates the
    // first block.
    if (dynID < 0 && tid == 0) {
        int32_t prefix = 0;
        for (uint32_t counter = 0; counter < num_blocks - 1; counter++) {  // 1 block is aux block
			while (flagArr[counter] == X) {
				// printf("dynid == -1, counter == %d -> flagArr[%d] == X\n", counter, counter);
				// printf("-1\n");
			}
			// printf("out\n");
            // Flag should be A
            int32_t tmp = aggrArr[counter];
            prefix = prefix + tmp;
            aggrArr[counter] = prefix;
            __threadfence();
            flagArr[counter] = P;
            __threadfence();
			// printf("flagArr:\n");
			// for (int i = 0; i < 25; i++) {
			// 	printf("%d ", flagArr[i]);
			// }
			// printf("\n");
        }
    } else if (dynID >= 0) {

        // Step 1.5 calculate some id's and stuff we will use
        int32_t globaloffset = dynID * B * Q;
        // printf("dynID == %d, globaloffset == %d\n", dynID, globaloffset);
        // Step 2 copy the memory the block will scan into shared memory.
		__shared__ int32_t blockShrMem[Q * B];
        volatile __shared__ int32_t blockShrBuf[B];
        copyFromGlb2ShrMem(globaloffset, N, 0, d_in, blockShrMem, tid);

        // __threadfence();
        // #define testid 1
		__syncthreads();

		#ifdef testid
        if (tid == 0 && dynID == testid) {
            printf("glbMem before load\n");
            for (int i = 0; i < B * Q; i++) {
                printf("%d ", d_in[i]);
            }
            printf("\n");
        }

        if (tid == 0 && dynID == testid) {
            printf("shrMem after load\n");
            for (int i = 0; i < B * Q; i++) {
                printf("%d ", blockShrMem[i]);
            }
            printf("\n");
        }
		#endif

        __syncthreads();

        // Step 3 Do the scan on the block
        // First scan each thread
        threadScan(blockShrMem, blockShrBuf, tid);
		__syncthreads();
		#ifdef testid
        if (tid == 0 && dynID == testid) {
            printf("shrBuf after thread scan\n");
            for (int i = 0; i < B; i++) {
                printf("%d ", blockShrBuf[i]);
            }
            printf("\n");
        }
        __syncthreads();
		#endif

        // Do the scan on the block level
        blockScan(blockShrBuf, tid);
        __syncthreads();

		#ifdef testid
        if (tid == 0 && dynID == testid) {
            printf("shrBuf after block scan\n");
            for (int i = 0; i < B; i++) {
                printf("%d ", blockShrBuf[i]);
            }
            printf("\n");
        }
        __syncthreads();
		#endif

        // Save the result in shrmem.
        threadAdd(blockShrMem, blockShrBuf, tid);
		// __threadfence();
		__syncthreads();

		#ifdef testid
        if (tid == 0 && dynID == testid) {
            printf("shrMem after thread add\n");
            for (int i = 0; i < Q * B; i++) {
                printf("%d ", blockShrMem[i]);
            }
            printf("\n");
        }
		__syncthreads();
		#endif

        // Step 4 Update aggregate array
        if (tid == B - 1 && dynID < num_blocks - 1) {
            int32_t res = blockShrMem[(Q - 1) * B + tid];
            aggrArr[dynID] = res;
            __threadfence();
            flagArr[dynID] = A;
        }
		// if (tid == 0) {
		// 	printf("into flagArr: dynID == %d\n", dynID);
		// }
        while (flagArr[dynID] != P) {
            // printf("%d\n", dynID);
        }
        // Let block 0 calculate the prefix, we wait for it.
		__syncthreads();
        // printf("dynid == %d - flagArr[%d] == %d - aggrArr[%d] == %d\n", dynID, dynID, flagArr[dynID], dynID, aggrArr[dynID]);
        
        // Step 5 calculate prefixArr value, might block or wait.

        // Step 6 Update prefix array

        // Get the prefix value as it is ready.
        // int32_t prefix = aggrArr[dynID];
		int32_t prefix = 0;
		if (dynID > 0) {
	        prefix = aggrArr[dynID - 1];
		}
		__threadfence();
        // printf("dynID == %d, tid == %d, prefix == %d\n", dynID, tid, prefix);
        // Step 7 Sum the prefix into the scan

        threadAddVal(blockShrMem, prefix, tid, dynID);
        __syncthreads();
		#ifdef testid
        if (tid == 0 && dynID == testid) {
            printf("shrMem after thread add val\n");
            for (int i = 0; i < Q * B; i++) {
                printf("%d ", blockShrMem[i]);
            }
            printf("\n");
        }
		__syncthreads();
		#endif

        // Step 8 Copy the result into global memory

        copyFromShr2GlbMem(globaloffset, N, d_out, blockShrMem, tid);

        __syncthreads();
		#ifdef testid
        if (tid == 0 && dynID == testid) {
            printf("glbMem final after load\n");
            for (int i = 0; i < N; i++) {
                printf("%d ", d_out[i]);
            }
            printf("\n");
        }
		#endif

    }
	// if (tid == 0) {
	// 	printf("dyn block %d done\n", dynID);
	// }
    // Step 9 Die!
}

/*** Steps for the kernel in general and for other Kernel ***/

// Step 0 calculate some id's and stuff we will use

// Step 1 get a dynamic id

// Step 2 copy the memory the block will scan into shared memory.

// Step 3 Do the scan on the block

// Step 4 Update aggregate array

// Step 5 calculate prefixArr value, might block or wait.

// Step 6 Update prefix array

// Step 7 Sum the prefix into the scan

// Step 8 Copy the result into global memory

// Step 9 Die!

#endif /* SPS_CU_H */