#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <atomic>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <curand_kernel.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int flag = 0;
int NUM_VALS, BLOCKS, THREADS;

const char* main_function = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";
const char* memcpyDeviceToHost = "memcpyDeviceToHost";
const char* sortedInput = "sorted";
const char* randomInput = "random";
const char* reverseSortedInput = "reverse_sorted";
const char* perturbed = "perturbed";


/*
Source: https://github.com/domkris/CUDA-Bubble-Sort/blob/master/CUDABubbleSort/kernel.cu
AI (Chat GPT) was used to create functions for data generation and correctness checking
*/



__host__ void bubbleSortHost(int *array, int index)
{
	int temp;
	do {

		for (int i = 0; i < NUM_VALS - 1 - index * 2 - flag; i++) {
			if (array[index * 2 + i] > array[index * 2 + 1 + i]) {
				temp = array[index * 2 + 1 + i];
				array[index * 2 + 1 + i] = array[index * 2 + i];
				array[index * 2 + i] = temp;
			}
		}

		flag++;

	} while (NUM_VALS - 1 - index * 2 - flag> 0);
}

__global__ void bubbleSortDeviceParallel(int *array, int offSet, int THREADS, int BLOCKS)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int indexPerBlock = threadIdx.x;
	int temp;

	if (index  < THREADS* BLOCKS) {

		// FIRST STEP
		if (offSet == 0) {

			// DO THREAD SORTING IN CORRESPONDING BLOCK 
			for (int j = 0; j < THREADS / 2; j++) {

				for (int i = 0; i < THREADS * 2 - 1 - indexPerBlock * 2; i++) {

					if (array[index * 2 + i] > array[index * 2 + 1 + i]) {
						temp = array[index * 2 + 1 + i];
						array[index * 2 + 1 + i] = array[index * 2 + i];
						array[index * 2 + i] = temp;
					}
				}
				__syncthreads();
			}
		}
		// ALL OTHER STEPS, INDEX/THREADS/BLOCKS SHIFTED FOR int offSet
		// LAST BLOCK SKIPPED
		else {
			if (blockIdx.x != BLOCKS - 1) {
				for (int j = 0; j < THREADS / 2; j++) {
					for (int i = offSet; i < THREADS * 2 - 1 + offSet - indexPerBlock * 2; i++) {

						if (array[index * 2 + i] > array[index * 2 + 1 + i]) {
							temp = array[index * 2 + 1 + i];
							array[index * 2 + 1 + i] = array[index * 2 + i];
							array[index * 2 + i] = temp;
						}

					}
					__syncthreads();
				}
			}
		}
	}

}

__global__ void initialize_reverse_sorted(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        array[tid] = size - tid; // Reverse sorted data
    }
}

__global__ void initialize_sorted(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        array[tid] = tid; // sorted data
    }
}

__global__ void initialize_random(int* array, int size, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    // Generate random numbers and scale them to the desired range
    if (tid < size) {
        array[tid] = static_cast<int>(curand_uniform(&state) * static_cast<float>(INT_MAX));
        //array[tid] = 1;
    }
}

__global__ void initialize_perturbed(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        if (tid % 100 == 0)
            array[tid] = 1; // 1%perturbed
    }
}

bool isSorted(int* array, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            return false;  // Array is not sorted
        }
    }
    return true;  // Array is sorted
}

int main(int argc, char* argv[])
{
    printf("main function beginning\n");
    CALI_MARK_BEGIN(main_function);
    
    cali::ConfigManager mgr;
    mgr.start();

	//THREADS = atoi(argv[1]);
    //NUM_VALS = atoi(argv[2]);
    char* input = argv[3];
    THREADS = 128;
    NUM_VALS = 1048576;
    // const char* input = "random";
    const char* input_type;
    size_t size = NUM_VALS * sizeof(int);
    BLOCKS = NUM_VALS / THREADS;
	int *h_array;
	int *d_array;
	int offSet;
	
	printf("allocating arrays\n");

	h_array = new int[NUM_VALS];

    if (cudaMalloc(&d_array, size) != cudaSuccess)
    {
        cout << "D_ARRAY ALLOCATING NOT WORKING!" << endl;
        return 0;
    }

    CALI_MARK_BEGIN(data_init);

    // Generate the data in parallel
    // Generate in reverse sorted order
    if (strcmp(input, sortedInput) == 0) {
        printf("sorted input starting\n");
        initialize_sorted<<<BLOCKS, THREADS>>>(d_array, NUM_VALS);
        input_type = "sorted";
    }
    else if (strcmp(input, randomInput) == 0) {
        printf("generating data randomly\n");
        initialize_random<<<BLOCKS, THREADS>>>(d_array, NUM_VALS, time(NULL));
        input_type = "random";
    }
    else if (strcmp(input, reverseSortedInput) == 0) {
        initialize_reverse_sorted<<<BLOCKS, THREADS>>>(d_array, NUM_VALS);
        input_type = "reverse_sorted";
    }
    else if (strcmp(input, perturbed) == 0) {
        initialize_perturbed<<<BLOCKS, THREADS>>>(d_array, NUM_VALS);
        input_type = "perturbed";
    }

    cudaDeviceSynchronize();
    
    printf("data generated\n");
    
    // cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost); 
    
    // for (int i = 0; i < NUM_VALS; i++) {
    //     printf("%d ", h_array[i]);
    // }
    // printf("\n");

    // CALI_MARK_END(data_init);

    // CALI_MARK_BEGIN(comp);
    // CALI_MARK_BEGIN(comp_large);


    // do {

    //     for (int i = 0; i < THREADS * 2; i++) {
    //         offSet = i;
    //         // POSSIBLE CHANGE: if offset != 0 USE bubbleSortDeviceParallel << < BLOCKS-1, THREADS >> > (d_array, offSet);
    //         bubbleSortDeviceParallel << < BLOCKS, THREADS >> > (d_array, offSet, THREADS, BLOCKS);
    //     }

    //     BLOCKS--;
    // } while (BLOCKS > 0);

    // cudaDeviceSynchronize();
    // printf("comp done\n");

    // CALI_MARK_END(comp_large);
    // CALI_MARK_END(comp);

    // CALI_MARK_BEGIN(comm);
    // CALI_MARK_BEGIN(comm_large);
    // CALI_MARK_BEGIN("cudaMemcpy");

    // cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // CALI_MARK_END("cudaMemcpy");
    // CALI_MARK_END(comm_large);
    // CALI_MARK_END(comm);

    // CALI_MARK_BEGIN(correctness_check);

    // if (isSorted(h_array, NUM_VALS)) {
    //     printf("Ending array is properly sorted!\n");
    // } else {
    //     printf("Ending array is not properly sorted :(\n");
    // }

    // CALI_MARK_END(correctness_check);

	// // FREEING MEMORY OF CPU & GPU
	// delete[] h_array;
	// cudaFree(d_array);
	// cudaDeviceReset();

    // CALI_MARK_END(main_function);

    // const char* algorithm = "BubbleSort";
    // const char* programmingModel = "CUDA";
    // const char* datatype = "int";
    // int sizeOfDatatype = sizeof(int);
    // int inputSize = NUM_VALS;
    // int num_procs = 1;
    // int num_threads = THREADS;
    // int num_blocks = NUM_VALS / THREADS;
    // int group_number = 18;
    // const char* implementation_source = "Online and AI";

    // adiak::init(NULL);
    // adiak::launchdate();    // launch date of the job
    // adiak::libraries();     // Libraries used
    // adiak::cmdline();       // Command line used to launch the job
    // adiak::clustername();   // Name of the cluster
    // adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    // adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    // adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    // adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    // adiak::value("input_size", inputSize); // The number of elements in input dataset (1000)
    // adiak::value("input_type", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    // adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    // adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    // mgr.stop();
    // mgr.flush();

	return 0;
}