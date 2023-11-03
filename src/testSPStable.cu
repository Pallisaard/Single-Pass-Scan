#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostSkel.cu.h"

#ifndef INCLUDE_QUAD
#define INCLUDE_QUAD 0
#endif

// Initialize an array of int32_t with random values between -R and R.
// Array has length N.
// R seems to be max value of the elements of the array.
void initArrayInt32(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

// Initialize an array of Quad<int32_t> with random values between -R and R.
// Array has length N.
// R seems to be max value of the elements of the array.
void initArrayQuadInt32(Quad<int32_t>* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i].x = (rand() % M) - R;
        inp_arr[i].y = (rand() % M) - R;
        inp_arr[i].z = (rand() % M) - R;
        inp_arr[i].w = (rand() % M) - R;
    }
}

/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 * N - length of the input array
 * h_in - host input of size: N * sizeof(int)
 * d_in - device input of size: N * sizeof(ElTp)
 */ 
template<typename T>
int bandwidthCudaMemcpy(const size_t N, T* d_in, T* d_out) {
    // dry run to exercise the d_out allocation!
    const size_t mem_size = N * sizeof(T);
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
        gigaBytesPerSec = 2 * N * sizeof(T) * 1.0e-3f / elapsed;
        printf("%.2f,", gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );

    return 0;
}

/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 * N - length of the input array
 * h_in - host input of size: N * sizeof(int)
 * d_in - device input of size: N * sizeof(ElTp)
 */
 template<typename T>
int bandwidthMemcpy(const size_t N, T* d_in, T* d_out) {
    // dry run to exercise the d_out allocation!
    const uint32_t num_blocks = (N + 1024 - 1) / 1024;
    naiveMemcpy<T><<< num_blocks, 1024>>>(d_out, d_in, N);
	cudaDeviceSynchronize();

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<T><<< num_blocks, 1024 >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(T) * 1.0e-3f / elapsed;
        printf("%.2f,", gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );

    return 0;
}

/*
 * singlePassScanAuxBlock performs a single pass scan using an auxiliary block.
 * N - length of the input array
 * h_in - host input of size: N * sizeof(int)
 * d_in - device input of size: N * sizeof(ElTp)
 * d_out - device result of size: N * sizeof(int)
 * Returns 0 if the scan was successful, 1 otherwise.
 */
template<typename T>
int singlePassScanAuxBlock(const size_t N, T* h_in,
	                       T* d_in, T* d_out) {
    const size_t mem_size = N * sizeof(T);
    T* h_out = (T*)malloc(mem_size);
    T* h_ref = (T*)malloc(mem_size);
    cudaMemset(d_out, 0, N*sizeof(T));

    uint32_t num_blocks = (N+B*Q-1)/(B*Q) + 1;  // We add 1 to be our auxiliary block.
	size_t f_array_size = num_blocks - 1;
    int32_t* IDAddr;
    uint32_t* flagArr;
    T* aggrArr;
    T* prefixArr;
    cudaMalloc((void**)&IDAddr, sizeof(int32_t));
    cudaMemset(IDAddr, -1, sizeof(int32_t));
    cudaMalloc(&flagArr, f_array_size * sizeof(uint32_t));
    cudaMemset(flagArr, X, f_array_size * sizeof(uint32_t));
    cudaMalloc(&aggrArr, f_array_size * sizeof(T));
    cudaMemset(aggrArr, 0, f_array_size * sizeof(T));
    cudaMalloc(&prefixArr, f_array_size * sizeof(T));
    cudaMemset(prefixArr, 0, f_array_size * sizeof(T));

    // dry run to exercise the d_out allocation!
    SinglePassScanKernel1<T><<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
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
            cudaMemset(aggrArr, 0, f_array_size * sizeof(T));
            cudaMemset(prefixArr, 0, f_array_size * sizeof(T));
            SinglePassScanKernel1<T><<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
            // printf("gpu %d\n", i + 1);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        elapsed = elapsed / RUNS_GPU;
        double gigaBytesPerSec = N  * 2 * sizeof(T) * 1.0e-3f / elapsed;
        printf("%.2f,", gigaBytesPerSec);
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
template<typename T>
int singlePassScanLookback(const size_t N, T* h_in,
                                T* d_in, T* d_out) {
    const size_t mem_size = N * sizeof(T);
    T* h_out = (T*)malloc(mem_size);
    T* h_ref = (T*)malloc(mem_size);
	cudaMemset(d_out, 0,  N *sizeof(T));

	uint32_t num_blocks = (N+B*Q-1)/(B*Q);
    size_t f_array_size = num_blocks;
    int32_t* IDAddr;
    uint32_t* flagArr;
    T* aggrArr;
    T* prefixArr;
    cudaMalloc((void**)&IDAddr, sizeof(int32_t));
    cudaMemset(IDAddr, 0, sizeof(int32_t));
    cudaMalloc(&flagArr, f_array_size * sizeof(uint32_t));
    cudaMemset(flagArr, X, f_array_size * sizeof(uint32_t));
    cudaMalloc(&aggrArr, f_array_size * sizeof(T));
    cudaMemset(aggrArr, 0, f_array_size * sizeof(T));
    cudaMalloc(&prefixArr, f_array_size * sizeof(T));
    cudaMemset(prefixArr, 0, f_array_size * sizeof(T));

    // dry run to exercise the d_out allocation!
    SinglePassScanKernel2<T><<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
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
            cudaMemset(aggrArr, 0, f_array_size * sizeof(T));
            cudaMemset(prefixArr, 0, f_array_size * sizeof(T));
            SinglePassScanKernel2<T><<< num_blocks, B>>>(d_in, d_out, N, IDAddr, flagArr, aggrArr, prefixArr);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        elapsed = elapsed / RUNS_GPU;
        double gigaBytesPerSec = N  * 2 * sizeof(T) * 1.0e-3f / elapsed;
        printf("%.2f,", gigaBytesPerSec);
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
template<typename T>
int cpuSeqScan(const size_t N, T* h_in,
	           T* d_in, T* d_out) {
    const size_t mem_size = N * sizeof(T);
    T* h_out = (T*)malloc(mem_size);

    // dry run to exercise the h_out allocation!
    cudaDeviceSynchronize();
    T acc = T();
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
            T acc = T();
            for(uint32_t i=0; i<N; i++) {
                acc = h_in[i] + acc;
                h_out[i] = acc;
            }
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        double gigaBytesPerSec = N * 2 * sizeof(T) * 1.0e-3f / elapsed;
        printf("%.2f,", gigaBytesPerSec);
    }

    free(h_out);
    return 0;
}

/*
 * scanIncAddI32 computes a scan inclusive add on the GPU using the
 * scanInc kernel we implemented in assignment 2.
 * b_size - desired CUDA block size ( <= 1024, multiple of 32)
 * N - length of the input array
 * h_in - host input of size: N * sizeof(int)
 * d_in - device input of size: N * sizeof(ElTp)
 * d_out - device result of size: N * sizeof(int)
 */
template<typename T>
int scanIncAdd(const uint32_t b_size, const size_t N, T* h_in,
				  T* d_in, T* d_out) {
    const size_t mem_size = N * sizeof(T);
    T* d_tmp;
    T* h_out = (T*)malloc(mem_size);
    T* h_ref = (T*)malloc(mem_size);
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(T));
    cudaMemset(d_out, 0, N*sizeof(T));

    // dry run to exercise d_tmp allocation
    scanInc< Add<T> > ( b_size, N, d_out, d_in, d_tmp );

    // time the GPU computation
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int i=0; i<RUNS_GPU; i++) {
        scanInc< Add<T> > ( b_size, N, d_out, d_in, d_tmp );
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
    double gigaBytesPerSec = N  * 2 * sizeof(T) * 1.0e-3f / elapsed;
	printf("%.2f", gigaBytesPerSec);

    gpuAssert( cudaPeekAtLastError() );

    free(h_out);
    free(h_ref);
    cudaFree(d_tmp);

    return 0;
}

int i32Experiments(const uint32_t N) {
    const size_t mem_size = N*sizeof(int32_t);
    int32_t* h_in    = (int32_t*) malloc(mem_size);
    int32_t* d_in;
    int32_t* d_out;
    cudaMalloc((void**)&d_in ,   mem_size);
    cudaMalloc((void**)&d_out,   mem_size);

    initArrayInt32(h_in, N, 13);
	cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // Scan experiments.
    {
		// computing a "realistic/achievable" bandwidth figure
		bandwidthCudaMemcpy<int32_t>(N, d_in, d_out);
		bandwidthMemcpy<int32_t>(N, d_in, d_out);
		// Scan experiments.
        cpuSeqScan<int32_t>(N, h_in, d_in, d_out);
        singlePassScanAuxBlock<int32_t>(N, h_in, d_in, d_out);
        singlePassScanLookback<int32_t>(N, h_in, d_in, d_out);
		if (N != 2 << 21)
			scanIncAdd<int32_t>(B, N, h_in, d_in, d_out);
    }

    // cleanup memory
    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);

	return 0;
}

#if INCLUDE_QUAD == 1
int quadInt32Experiments(const uint32_t N) {
	const size_t mem_size = N*sizeof(Quad<int32_t>);
	Quad<int32_t>* h_in    = (Quad<int32_t>*) malloc(mem_size);
	Quad<int32_t>* d_in;
	Quad<int32_t>* d_out;
	cudaMalloc((void**)&d_in ,   mem_size);
	cudaMalloc((void**)&d_out,   mem_size);

	initArrayQuadInt32(h_in, N, 13);
	cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

	// Scan experiments.
	{
		// computing a "realistic/achievable" bandwidth figure
		bandwidthCudaMemcpy<Quad<int32_t>>(N, d_in, d_out);
		bandwidthMemcpy<Quad<int32_t>>(N, d_in, d_out);
		// bandwidthGlgShrMemcpyInt32(N, h_in, d_in, d_out);
		// Scan experiments.
		cpuSeqScan<Quad<int32_t>>(N, h_in, d_in, d_out);
		singlePassScanAuxBlock<Quad<int32_t>>(N, h_in, d_in, d_out);
		singlePassScanLookback<Quad<int32_t>>(N, h_in, d_in, d_out);
		if (N != 2 << 19)
			scanIncAdd<Quad<int32_t>>(B, N, h_in, d_in, d_out);
	}

	// cleanup memory
	free(h_in);
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
#endif

int main (int argc, char * argv[]) {
    if (argc != 1) {
        printf("Usage: %s\n", argv[0]);
        exit(1);
    }

    initHwd();

    printf("Testing parallel basic blocks for CUDA-block size: %d and Q: %d\n\n\n", B, Q);

	printf("N (2^i),cudaMemcpy,naiveMemcpy,cpuSeqScan,singlePassScanAuxBlock,singlePassScanLookback,scanIncAdd\n");
	for (uint32_t i = 10; i < 31; i++) {
		printf("%d,", i);
		uint32_t N = 2 << i;
		// printf("N: %d,", N);
		i32Experiments(N);
		printf("\n");
	}
	
	#if INCLUDE_QUAD
	printf("N,cudaMemcpy,naiveMemcpy,cpuSeqScan,singlePassScanAuxBlock,singlePassScanLookback,scanIncAdd\n");
	for (uint32_t i = 10; i < 30; i++) {
		printf("%d,", i);
		uint32_t N = 2 << i;
		quadInt32Experiments(N);
		printf("\n");
	}
	#endif

	return 0;
}
