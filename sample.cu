#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

__global__ void generateRandomData(int* data, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        // Generate random data for each thread
        data[tid] = rand() % 1000;
    }
}

__global__ void computeSamples(int* data, int* samples, int num_elements, int num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = num_elements / num_samples;
    int sample_idx = tid * stride;
    samples[tid] = data[sample_idx];
}

__global__ void distributeData(int* data, int* samples, int* bucket_offsets, int num_elements, int num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int element = data[tid];

    int bucket_id = 0;
    while (bucket_id < num_samples - 1 && element > samples[bucket_id]) {
        bucket_id++;
    }

    int bucket_start = (bucket_id == 0) ? 0 : bucket_offsets[bucket_id - 1];
    int bucket_offset = atomicAdd(&bucket_offsets[bucket_id], 1);
    data[bucket_start + bucket_offset] = element;
}

void sampleSort(int* data, int num_elements, int num_samples) {
    int* d_data;
    int* d_samples;
    int* d_bucket_offsets;

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    cudaMalloc((void**)&d_data, num_elements * sizeof(int));
    cudaMalloc((void**)&d_samples, num_samples * sizeof(int));
    cudaMalloc((void**)&d_bucket_offsets, num_samples * sizeof(int));

    cudaMemcpy(d_data, data, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    computeSamples<<<blocks, threads>>>(d_data, d_samples, num_elements, num_samples);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    distributeData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_data, d_samples, d_bucket_offsets, num_elements, num_samples);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_samples);
    cudaFree(d_bucket_offsets);
}

bool isSorted(int* data, int num_elements) {
    for (int i = 1; i < num_elements; i++) {
        if (data[i - 1] > data[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    int* data = (int*)malloc(num_elements * sizeof(int));
    int* d_data;

    cudaMalloc((void**)&d_data, num_elements * sizeof(int));

    // Launch the CUDA kernel for data generation only once
    generateRandomData<<<blocks, threads>>>(d_data, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    sampleSort(data, num_elements, num_samples);

    // Check if the data is sorted correctly
    if (isSorted(data, num_elements)) {
        printf("Sorting is successful.\n");
    } else {
        printf("Sorting is not correct.\n");
    }

    free(data);
    return 0;
}
