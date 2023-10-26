#ifndef SCAN_HOST
#define SCAN_HOST

#include "constants.cu.h"
// #include "pbbKernels.cu.h"
#include "sps.cu.h"

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

uint32_t closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

void log2UB(uint32_t n, uint32_t* ub, uint32_t* lg) {
    uint32_t r = 0;
    uint32_t m = 1;
    if( n <= 0 ) { printf("Error: log2(0) undefined. Exiting!!!"); exit(1); }
    while(m<n) {
        r = r + 1;
        m = m * 2;
    }
    *ub = m;
    *lg = r;
}




/**
 * Host Wrapper orchestraiting the execution of scan:
 * d_in  is the input array
 * d_out is the result array (result of scan)
 * t_tmp is a temporary array (used to scan in-place across the per-block results)
 * Implementation consist of three phases:
 *   1. elements are partitioned across CUDA blocks such that the number of
 *      spawned CUDA blocks is <= 1024. Each CUDA block reduces its elements
 *      and publishes the per-block partial result in `d_tmp`. This is 
 *      implemented in the `redAssocKernel` kernel.
 *   2. The per-block reduced results are scanned within one CUDA block;
 *      this is implemented in `scan1Block` kernel.
 *   3. Then with the same partitioning as in step 1, the whole scan is
 *      performed again at the level of each CUDA block, and then the
 *      prefix of the previous block---available in `d_tmp`---is also
 *      accumulated to each-element result. This is implemented in
 *      `scan3rdKernel` kernel and concludes the whole scan.
 */
// template<class OP>                     // element-type and associative operator properties
// void scanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
//             , const size_t       N     // length of the input array
//             , typename OP::RedElTp* d_out // device array of length: N
//             , typename OP::InpElTp* d_in  // device array of length: N
//             , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
// ) {
//     const uint32_t inp_sz = sizeof(typename OP::InpElTp);
//     const uint32_t red_sz = sizeof(typename OP::RedElTp);
//     const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
//     const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
//     uint32_t num_seq_chunks;
//     const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
//     const size_t   shmem_size = B * max_tp_size * CHUNK;

//     //
//     redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks);

//     {
//         const uint32_t block_size = closestMul32(num_blocks);
//         const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
//         scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
//     }

//     scan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks);
// }


#endif //SCAN_HOST
