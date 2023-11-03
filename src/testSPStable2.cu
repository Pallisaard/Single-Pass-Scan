#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostSkel.cu.h"


// R seems to be max value of the elements of the array
void initArray(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 */ 
int bandwidthCudaMemcpy( const size_t   N     // length of the input array
                   	   , int* d_in            // device input  of length N
                       , int* d_out           // device result of length N
) {
    // dry run to exercise the d_out allocation!
    const size_t mem_size = N * sizeof(int32_t);
	cudaMemcpy(d_out, d_in, mem_size, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<RUNS_GPU; i++) {
			cudaMemcpy(d_out, d_in, mem_size, cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int32_t) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );
    return 0;
}

/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 */ 
int bandwidthMemcpy( const size_t   N     // length of the input array
                   , int* d_in            // device input  of length N
                   , int* d_out           // device result of length N
) {
    // dry run to exercise the d_out allocation!
    // const uint32_t num_blocks = (N + B - 1) / B;
    const uint32_t num_blocks = (N + 1024 - 1) / 1024;
    // const uint32_t num_blocks = (N + B * Q - 1) / (B * Q);
    naiveMemcpy<<< num_blocks, 1024 >>>(d_out, d_in, N);
	cudaDeviceSynchronize();

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<<< num_blocks, 1024 >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int32_t) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );
    return 0;
}

/*
 * Measure a memcpy like kernel using registers in the GPU akin to
 * what the other methods use.
 */ 
 int bandwidthRegMemcpy( const size_t   N     // length of the input array
	, int* h_in			   // host input    of length N
	, int* d_in            // device input  of length N
	, int* d_out           // device result of length N
) {
    const size_t mem_size = N * sizeof(int32_t);
	int32_t* h_out = (int32_t*)malloc(mem_size);
    int32_t* h_ref = (int32_t*)malloc(mem_size);

	// dry run to exercise the d_out allocation!
	// const uint32_t num_blocks = (N + B - 1) / B;
	// const uint32_t num_blocks = (N + 1024 - 1) / 1024;
	const uint32_t num_blocks = (N + B * Q - 1) / (B * Q);
	regMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
	cudaDeviceSynchronize();

	double gigaBytesPerSec;
	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;

	{ // timing the GPU implementations
		gettimeofday(&t_start, NULL); 

		for(int i=0; i<RUNS_GPU; i++) {
			regMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
		}
		cudaDeviceSynchronize();

		gettimeofday(&t_end, NULL);
		timeval_subtract(&t_diff, &t_end, &t_start);
		elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
		gigaBytesPerSec = 2 * N * sizeof(int32_t) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
	}

	gpuAssert( cudaPeekAtLastError() );

    free(h_out);
    free(h_ref);

	return 0;
}

int bandwidthGlgShrMemcpy( const size_t   N     // length of the input array
	, int* h_in			   // host input    of length N
	, int* d_in            // device input  of length N
	, int* d_out           // device result of length N
) {
    const size_t mem_size = N * sizeof(int32_t);
	int32_t* h_out = (int32_t*)malloc(mem_size);
    int32_t* h_ref = (int32_t*)malloc(mem_size);

	// dry run to exercise the d_out allocation!
	// const uint32_t num_blocks = (N + B - 1) / B;
	// const uint32_t num_blocks = (N + 1024 - 1) / 1024;
	const uint32_t num_blocks = (N + B * Q - 1) / (B * Q);
	glbShrMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
	cudaDeviceSynchronize();

	double gigaBytesPerSec;
	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;

	{ // timing the GPU implementations
		gettimeofday(&t_start, NULL); 

		for(int i=0; i<RUNS_GPU; i++) {
			glbShrMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
		}
		cudaDeviceSynchronize();

		gettimeofday(&t_end, NULL);
		timeval_subtract(&t_diff, &t_end, &t_start);
		elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
		gigaBytesPerSec = 2 * N * sizeof(int32_t) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
	}

	gpuAssert( cudaPeekAtLastError() );

    free(h_out);
    free(h_ref);

	return 0;
}

/*
* singlePassScanAuxBlock performs a single pass scan using an auxiliary block.
* N - length of the input array
* h_in - host input    of size: N * sizeof(int)
* d_in - device input  of size: N * sizeof(ElTp)
* d_out - device result of size: N * sizeof(int)
* Returns 0 if the scan was successful, 1 otherwise.
*/

int singlePassScanAuxBlock(const size_t N, int32_t* h_in,
                           int32_t* d_in, int32_t* d_out) {
    const size_t mem_size = N * sizeof(int32_t);
    int32_t* h_out = (int32_t*)malloc(mem_size);
    int32_t* h_ref = (int32_t*)malloc(mem_size);
    uint32_t num_blocks = (N+B*Q-1)/(B*Q) + 1;  // We add 1 to be our auxiliary block.
		size_t f_array_size = num_blocks - 1;
    int32_t* IDAddr;
    uint32_t* flagArr;
    int32_t* aggrArr;
    int32_t* prefixArr;
    cudaMalloc((void**)&IDAddr, sizeof(int32_t));
    cudaMemset(IDAddr, -1, sizeof(int32_t));
    cudaMalloc(&flagArr, f_array_size * sizeof(uint32_t));
    cudaMemset(flagArr, X, f_array_size * sizeof(uint32_t));
    cudaMalloc(&aggrArr, f_array_size * sizeof(int32_t));
    cudaMemset(aggrArr, 0, f_array_size * sizeof(int32_t));
    cudaMalloc(&prefixArr, f_array_size * sizeof(uint32_t));
    cudaMemset(prefixArr, 0, f_array_size * sizeof(uint32_t));

    // dry run to exercise the d_out allocation!
    SinglePassScanKernel1<<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
    // printf("done 0\n");
    cudaDeviceSynchronize();

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
		// time the GPU computation
    // Need to reset the dynID and flag arr each time we call the kernel
    // Before we can start to run it multiple times and get a benchmark.
    {
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_GPU; i++) {
            cudaMemset(IDAddr, -1, sizeof(int32_t));
            cudaMemset(flagArr, X, f_array_size * sizeof(uint32_t));
            cudaMemset(aggrArr, 0, f_array_size * sizeof(int32_t));
            cudaMemset(prefixArr, 0, f_array_size * sizeof(uint32_t));
            SinglePassScanKernel1<<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
            // printf("gpu %d\n", i + 1);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        elapsed = elapsed / RUNS_GPU;
        double gigaBytesPerSec = N  * 2 * sizeof(int32_t) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
    }
    gpuAssert( cudaPeekAtLastError() );

    free(h_out);
    free(h_ref);
    cudaFree(IDAddr);
    cudaFree(flagArr);
    cudaFree(aggrArr);
    cudaFree(prefixArr);

    return 0;
}

/*
* singlePassScanLookback performs a single pass scan using lookback.
* N - length of the input array
* h_in - host input    of size: N * sizeof(int)
* d_in - device input  of size: N * sizeof(ElTp)
* d_out - device result of size: N * sizeof(int)
* Returns 0 if the scan was successful, 1 otherwise.
*/

int singlePassScanLookback(const size_t N, int32_t* h_in,
                           int32_t* d_in, int32_t* d_out) {
    const size_t mem_size = N * sizeof(int32_t);
    int32_t* h_out = (int32_t*)malloc(mem_size);
    int32_t* h_ref = (int32_t*)malloc(mem_size);
    uint32_t num_blocks = (N+B*Q-1)/(B*Q);  // We add 1 to be our auxiliary block.
    size_t f_array_size = num_blocks;
    int32_t* IDAddr;
    uint32_t* flagArr;
    int32_t* aggrArr;
    int32_t* prefixArr;
    cudaMalloc((void**)&IDAddr, sizeof(int32_t));
    cudaMemset(IDAddr, 0, sizeof(int32_t));
    cudaMalloc(&flagArr, f_array_size * sizeof(uint32_t));
    cudaMemset(flagArr, X, f_array_size * sizeof(uint32_t));
    cudaMalloc(&aggrArr, f_array_size * sizeof(int32_t));
    cudaMemset(aggrArr, 0, f_array_size * sizeof(int32_t));
    cudaMalloc(&prefixArr, f_array_size * sizeof(uint32_t));
    cudaMemset(prefixArr, 0, f_array_size * sizeof(uint32_t));

    // dry run to exercise the d_out allocation!
    SinglePassScanKernel2<<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
    // printf("done 0\n");
    cudaDeviceSynchronize();

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    // time the GPU computation
    // Need to reset the dynID and flag arr each time we call the kernel
    // Before we can start to run it multiple times and get a benchmark.
    {
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_GPU; i++) {
            cudaMemset(IDAddr, 0, sizeof(int32_t));
            cudaMemset(flagArr, X, f_array_size * sizeof(uint32_t));
            cudaMemset(aggrArr, 0, f_array_size * sizeof(int32_t));
            cudaMemset(prefixArr, 0, f_array_size * sizeof(uint32_t));
            SinglePassScanKernel2<<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
            // printf("gpu %d\n", i + 1);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        elapsed = elapsed / RUNS_GPU;
        double gigaBytesPerSec = N  * 2 * sizeof(int32_t) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
    }
    gpuAssert( cudaPeekAtLastError() );

    free(h_out);
    free(h_ref);
    cudaFree(IDAddr);
    cudaFree(flagArr);
    cudaFree(aggrArr);
    cudaFree(prefixArr);

    return 0;
}

/*
* cpuSeqScan computes a sequencial scan on the cpu.
* N - length of the input array
* h_in - host input of size: N * sizeof(int)
* d_in - device input of size: N * sizeof(ElTp)
* d_out - device result of size: N * sizeof(int)
* Returns 0 if the scan was successful, 1 otherwise.
*/

int cpuSeqScan(const size_t N, int32_t* h_in, int32_t* d_in, int32_t* d_out){
    const size_t mem_size = N * sizeof(int32_t);
    int32_t* h_out = (int32_t*)malloc(mem_size);

    // dry run to exercise the h_out allocation!
    // printf("done 0\n");
    cudaDeviceSynchronize();
    int acc = 0;
    for(uint32_t i=0; i<N; i++) {
        acc = h_in[i] + acc;
        h_out[i] = acc;
    }
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    // The CPU we might as well just add the benchmark
    { // sequential computation
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_CPU; i++) {
            int acc = 0;
            for(uint32_t i=0; i<N; i++) {
                acc = h_in[i] + acc;
                h_out[i] = acc;
            }
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU + 1.0e-7f;
        double gigaBytesPerSec = N * (sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("%.2f", gigaBytesPerSec);
    }

    free(h_out);

    return 0;
}

// int scanIncAddI32( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
//                  , const size_t   N     // length of the input array
//                  , int* h_in            // host input    of size: N * sizeof(int)
//                  , int* d_in            // device input  of size: N * sizeof(ElTp)
//                  , int* d_out           // device result of size: N * sizeof(int)
// ) {
//     const size_t mem_size = N * sizeof(int);
//     int* d_tmp;
//     int* h_out = (int*)malloc(mem_size);
//     int* h_ref = (int*)malloc(mem_size);
//     cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(int));
//     cudaMemset(d_out, 0, N*sizeof(int));

//     // dry run to exercise d_tmp allocation
//     scanInc< Add<int> > ( B, N, d_out, d_in, d_tmp );

//     // time the GPU computation
//     unsigned long int elapsed;
//     struct timeval t_start, t_end, t_diff;
//     gettimeofday(&t_start, NULL); 

//     for(int i=0; i<RUNS_GPU; i++) {
//         scanInc< Add<int> > ( B, N, d_out, d_in, d_tmp );
//     }
//     cudaDeviceSynchronize();

//     gettimeofday(&t_end, NULL);
//     timeval_subtract(&t_diff, &t_end, &t_start);
//     elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
//     double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
//     printf("Scan Inclusive AddI32 GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
//           , elapsed, gigaBytesPerSec);

//     gpuAssert( cudaPeekAtLastError() );

//     { // sequential computation
//         gettimeofday(&t_start, NULL);
//         for(int i=0; i<RUNS_CPU; i++) {
//             int acc = 0;
//             for(uint32_t i=0; i<N; i++) {
//                 acc += h_in[i];
//                 h_ref[i] = acc;
//             }
//         }
//         gettimeofday(&t_end, NULL);
//         timeval_subtract(&t_diff, &t_end, &t_start);
//         elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
//         double gigaBytesPerSec = N * (sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
//         printf("Scan Inclusive AddI32 CPU Sequential runs in: %lu microsecs, GB/sec: %.2f\n"
//               , elapsed, gigaBytesPerSec);
//     }

//     { // Validation
//         cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);
//         for(uint32_t i = 0; i<N; i++) {
//             if(h_out[i] != h_ref[i]) {
//                 printf("!!!INVALID!!!: Scan Inclusive AddI32 at index %d, dev-val: %d, host-val: %d\n"
//                       , i, h_out[i], h_ref[i]);
//                 exit(1);
//             }
//         }
//         printf("Scan Inclusive AddI32: VALID result!\n\n");
//     }

//     free(h_out);
//     free(h_ref);
//     cudaFree(d_tmp);

//     return 0;
// }


int main (int argc, char * argv[]) {
    if (argc != 1) {
        printf("Usage: %s\n", argv[0]);
        exit(1);
    }

    initHwd();

    printf("Testing parallel basic blocks for CUDA-block size: %d and Q: %d\n\n\n", B, Q);

	printf("N (2^),naiveMemcpy,registerMemcpy,glbShrMemcpy,scanIncAddI32,singlePassScanAuxBlock,singlePassScanLookback\n");

	for (int32_t i = 10; i < 31; i++) {
		printf("%d", i);

		int32_t N = pow(2, i);
		const size_t mem_size = N*sizeof(int32_t);
		int32_t* h_in    = (int32_t*) malloc(mem_size);
		int32_t* d_in;
		int32_t* d_out;
		cudaMalloc((void**)&d_in ,   mem_size);
		cudaMalloc((void**)&d_out,   mem_size);

		initArray(h_in, N, 13);

			

		cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

		printf(",");
		// computing a "realistic/achievable" bandwidth figure
		{
			bandwidthMemcpy(N, d_in, d_out);
		}

		printf(",");
		// computing a bandwidth figure using registers
		{
			bandwidthRegMemcpy(N, h_in, d_in, d_out);
		}

		printf(",");
		{
			bandwidthGlgShrMemcpy(N, h_in, d_in, d_out);
		}

		printf(",");
		{
			bandwidthCudaMemcpy(N, d_in, d_out);
		}

		printf(",");
		// computing a sequential scan on the cpu
		{
			cpuSeqScan(N, h_in, d_in, d_out);
		}

		printf(",");
		// Computing a base single pass scan using an auxiliary block
		// to add dynamically allocated blocks to the scan.
		// This function also validates that the single pass scan works.
		{
			singlePassScanAuxBlock(N, h_in, d_in, d_out);
		}

		printf(",");
		// Computing a single pass scan using lookback.
		// This function also validates that the single pass scan works.
		{
			singlePassScanLookback(N, h_in, d_in, d_out);
		}

		// cleanup memory
		free(h_in);
		cudaFree(d_in);
		cudaFree(d_out);

		printf("\n");
	}
}
