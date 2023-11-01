#ifndef SPS_CU_H
#define SPS_CU_H

#include <cuda_runtime.h>

#define X 0
#define A 1
#define P 2
#ifndef Q
#define Q 4
#endif
#ifndef B
#define B 1024
#endif

template<typename T>
class Quad {
public:
	T x;
	T y;
	T z;
	T w;

	// initialize x y z q to default values of type T
	__device__ __host__ inline Quad() : x(T()), y(T()), z(T()), w(T()) {}

	__device__ __host__ inline Quad(const T& x, const T& y,
									const T& z, const T& w)
									: x(x), y(y), z(z), w(w) {}

	__device__ __host__ inline Quad(const Quad<T>& q) : x(q.x), y(q.y), z(q.z), w(q.w) {}

	__device__ __host__ inline Quad<T> operator=(const Quad<T>& q) {
		x = q.x;
		y = q.y;
		z = q.z;
		w = q.w;
	}

	__device__ __host__ inline Quad<T> operator+(const Quad<T>& q) {
		return Quad<T>(x + q.x, y + q.y, z + q.z, w + q.w);
	}
};

// shd_mem is a pointer to an array in shared memory. It has size Q * B.
// idx is the id of the part of the shared memory we start scanning.
template<typename T>
__device__ inline void threadScan(T* shd_mem, volatile T* shd_buf, uint32_t tid) {
    T acc = 0;
#pragma unroll
    for (int i = 0; i < Q; i++) {
        int tmp = shd_mem[tid * Q + i];
        acc = acc + tmp;
        shd_mem[tid * Q + i] = acc;
    }
    shd_buf[tid] = acc;
	__syncthreads();
}

template<typename T>
__device__ inline void threadAdd(T* shd_mem, volatile T* shd_buf, uint32_t tid) {
    if (tid != 0) {
        T tmp = shd_buf[tid - 1];
#pragma unroll
        for (int i = 0; i < Q; i++) {
            shd_mem[tid * Q + i] = shd_mem[tid * Q + i] + tmp;
        }
    }
	__syncthreads();
}

template<typename T>
__device__ inline void threadAddVal(T* shd_mem, T val, uint32_t tid, uint32_t dynID) {
#pragma unroll
    for (int i = 0; i < Q; i++) {
        shd_mem[i * B + tid] = shd_mem[i * B + tid] + val;
    }
	__syncthreads();
}

// Function that uses collectively scans a warp of elements in a shared buffer.
// shd_buf is a pointer to an array in shared memory. It has size B <= 1024.
// idx is the id of the part of the shared memory we start scanning.
//
// Each thread in the warp performs a scan across the shared buffer. The
// result is stored in the shared buffer.
template<typename T>
__device__ inline T warpScan(volatile T* shd_buf, uint32_t tid) {
    uint32_t lane = tid & (WARP - 1);  // WARP
    int k = lgWARP;

#pragma unroll
    for (int d = 0; d < k; d++) {
        int h = 1 << d;
        if (lane >= h) {
            shd_buf[tid] = shd_buf[tid - h] + shd_buf[tid];
        }
    }

    T res = shd_buf[tid];
    return res;
}

// Function that performs a per block scan collectively across all threads in
// the block.
// shd_buf is a pointer to an array of size B.
// idx is the id of the part of the shared memory we start scanning.
//
// Each thread copy the final value of the scan to the shd_buf. Then
// we perform a parallel scan across the shd_buf like in assignment 2.
template<typename T>
__device__ inline void blockScan(volatile T* shd_buf, uint32_t tid) {
    uint32_t lane = tid & (WARP - 1);
    uint32_t warpid = tid >> lgWARP;

    // step 1
    // scan at warp level
    T warp_res = warpScan(shd_buf, tid);
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

// device function for a lookback scan method.
template<typename T>
__device__ inline T lookbackScan(volatile T* agg_mem,
								 volatile T* pref_mem,
								 volatile uint32_t* flag_mem,
								 T* shr_mem, uint32_t dyn_idx,
								 uint32_t tid) {
	// Handle lookback differently depending on dynamic id.
    if (tid == B - 1 && dyn_idx == 0) {
		T agg_val = shr_mem[(Q - 1) * B + tid];
		agg_mem[dyn_idx] = agg_val;
        pref_mem[dyn_idx] = agg_val;
        __threadfence();
        flag_mem[dyn_idx] = P;

    } else if (tid == B - 1 && dyn_idx > 0) {
		T agg_val = shr_mem[(Q - 1) * B + tid];
        agg_mem[dyn_idx] = agg_val;
        __threadfence();
        flag_mem[dyn_idx] = A;
        uint32_t grab_id = dyn_idx - 1;
        while (flag_mem[grab_id] != P) {
            if (flag_mem[grab_id] == A && grab_id > 0) {
				agg_val = agg_mem[grab_id] + agg_val;
                grab_id--;
            }
        }
		pref_mem[dyn_idx] = agg_val + pref_mem[grab_id];
        __threadfence();
        flag_mem[dyn_idx] = P;
    }

	__syncthreads();
	T prefix = pref_mem[dyn_idx] - agg_mem[dyn_idx];

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
template<typename T>
__device__ inline void copyGlb2Shr(T glb_offs, const uint32_t N,
								   T ne, T* d_inp,
								   volatile T* shmem_inp,
								   uint32_t tid) {
#pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
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
 * This is very similar with `copyGlb2Shr` except
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
template<typename T>
__device__ inline void copyShr2Glb(T glb_offs, const uint32_t N,
								   T* d_out,
								   T* shmem_red,
								   uint32_t tid) {
#pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
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

template<typename T>
__global__ void glbShrMemcpy(T* d_out, T* d_inp, const uint32_t N)
{
	uint32_t tid = threadIdx.x;
	uint32_t globaloffset = blockIdx.x * B * Q;
	__shared__ T blockShrMem[Q * B];
	copyGlb2Shr<T>(globaloffset, N, 0, d_inp, blockShrMem, tid);

	copyShr2Glb<T>(globaloffset, N, d_out, blockShrMem, tid);
}

/**
 * A single pass scan kernel that uses the three arrays described i
 * Merrill & Garland's paper.
*/
template<typename T>
__global__ void SinglePassScanKernel2(T *d_in, T* d_out,
									  const size_t N, int32_t* IDAddr,
									  volatile uint32_t* flagArr,
									  volatile T* aggrArr,
									  volatile T* prefixArr) {
	// Allocate shared memory
	__shared__ T blockShrMem[Q * B];
	volatile __shared__ T blockShrBuf[B];

	// Step 1 get ids and initialize global arrays
	uint32_t tid = threadIdx.x;
	int32_t dynID = getDynID(IDAddr, tid);
	uint32_t globaloffset = dynID * B * Q;

	// Step 2 copy the memory the block will scan into shared memory.
	copyGlb2Shr<T>(globaloffset, N, 0, d_in, blockShrMem, tid);

	// Step 3 Do the scan on the block
	// First scan each thread
	threadScan<T>(blockShrMem, blockShrBuf, tid);

	// Do the scan on the block level
	blockScan<T>(blockShrBuf, tid);

	// Save the result in shrmem.
	threadAdd<T>(blockShrMem, blockShrBuf, tid);

	// Step 4 use lookback scan to find the inclusive prefix value
	T prefix = lookbackScan<T>(aggrArr, prefixArr, flagArr, blockShrMem, dynID, tid);

	// Step 5 Sum the prefix into the scan
	threadAddVal<T>(blockShrMem, prefix, tid, dynID);

	// Step 6 Copy the result into global memory
	copyShr2Glb<T>(globaloffset, N, d_out, blockShrMem, tid);
}

/**
 * Single pass scan kernel using a naive auxiliary thread to sum up
 * aggregates.
*/
template<typename T>
__global__ void SinglePassScanKernel1(T* d_in, T* d_out,
                                      const size_t N, int32_t* IDAddr,
                                      volatile uint32_t* flagArr,
									  volatile T* aggrArr,
                                      volatile T* prefixArr) {
    // Step 1 get a dynamic id
    uint32_t tid = threadIdx.x;
	uint32_t num_blocks = gridDim.x;
    int32_t dynID = getDynID(IDAddr, tid);

	__syncthreads();
    // If the first dynamic id, of -1 then we are the prefix block instead.
    // an optimisation might be to let id 0 do it, but it still calculates the
    // first block.
    if (dynID < 0 && tid == 0) {
        int32_t prefix = 0;
        for (uint32_t counter = 0; counter < num_blocks - 1; counter++) {  // 1 block is aux block
			while (flagArr[counter] == X) {}
            // Flag should be A
            int32_t tmp = aggrArr[counter];
            prefix = prefix + tmp;
            aggrArr[counter] = prefix;
            __threadfence();
            flagArr[counter] = P;
        }
    } else if (dynID >= 0) {

        // Step 1.5 calculate some id's and stuff we will use
        int32_t globaloffset = dynID * B * Q;

        // Step 2 copy the memory the block will scan into shared memory.
		__shared__ T blockShrMem[Q * B];
        volatile __shared__ T blockShrBuf[B];
        copyGlb2Shr<T>(globaloffset, N, 0, d_in, blockShrMem, tid);

        // Step 3 Do the scan on the block
        // First scan each thread
        threadScan<T>(blockShrMem, blockShrBuf, tid);

        // Do the scan on the block level
        blockScan<T>(blockShrBuf, tid);

        // Save the result in shrmem.
        threadAdd<T>(blockShrMem, blockShrBuf, tid);

        // Step 4 Update aggregate array
        if (tid == B - 1 && dynID < num_blocks - 1) {
            int32_t res = blockShrMem[(Q - 1) * B + tid];
            aggrArr[dynID] = res;
            __threadfence();
            flagArr[dynID] = A;
        }

        // Let block 0 calculate the prefix, we wait for it.
        while (flagArr[dynID] != P) {}

		int32_t prefix = 0;
		if (dynID > 0) {
	        prefix = aggrArr[dynID - 1];
		}
		__threadfence();

        // Step 7 Sum the prefix into the scan
        threadAddVal<T>(blockShrMem, prefix, tid, dynID);

        // Step 8 Copy the result into global memory
        copyShr2Glb<T>(globaloffset, N, d_out, blockShrMem, tid);
    }
    // Step 9 Die!
}

/**
 * Naive memcpy kernel, for the purpose of comparing with
 * a more "realistic" bandwidth number.
 */
template<typename T>
__global__ void naiveMemcpy(T* d_out, T* d_inp, const uint32_t N)
{
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < N) {
		d_out[gid] = d_inp[gid];
	}
	__syncthreads();
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