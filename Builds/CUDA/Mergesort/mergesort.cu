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

int NUM_VALS, BLOCKS, THREADS;

const char *main_function = "main";
const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *correctness_check = "correctness_check";
const char *Memcpy = "cudaMemcpy";

__device__ void merge(int *array, int *temp, int left, int mid, int right)
{
	int i = left;
	int j = mid + 1;
	int k = left;

	while (i <= mid && j <= right)
	{
		if (array[i] <= array[j])
		{
			temp[k++] = array[i++];
		}
		else
		{
			temp[k++] = array[j++];
		}
	}

	while (i <= mid)
	{
		temp[k++] = array[i++];
	}

	while (j <= right)
	{
		temp[k++] = array[j++];
	}

	for (i = left; i <= right; i++)
	{
		array[i] = temp[i];
	}
}

__device__ void mergeSort(int *array, int *temp, int left, int right)
{
	if (left < right)
	{
		int mid = left + (right - left) / 2;

		// Loop to simulate recursion
		mergeSort(array, temp, left, mid);
		mergeSort(array, temp, mid + 1, right);

		merge(array, temp, left, mid, right);
	}
}

__global__ void mergeSortWrapper(int *array, int *temp, int left, int right)
{
	mergeSort(array, temp, left, right);
}

void launchMergeSort(int *array, int *temp, int size)
{
	mergeSortWrapper<<<1, 1>>>(array, temp, 0, size - 1);
}

__global__ void generateDataKernel(int *array, int size, int init_type, unsigned long long seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int thread_seed = seed + tid;

	curandState state;
	curand_init(thread_seed, tid, 0, &state);

	if (init_type == 0)
	{
		// Random initialization
		if (tid < size)
		{
			array[tid] = curand(&state) % size;
		}
	}
	else if (init_type == 1)
	{
		// Sorted initialization
		if (tid < size)
		{
			array[tid] = tid + 1;
		}
	}
	else if (init_type == 2)
	{
		// Reverse sorted initialization
		if (tid < size)
		{
			array[tid] = size - tid;
		}
	}
	else if (init_type == 3)
	{
		// 1% perturbed initialization (mostly sorted except 1% of values)
		if (tid < size)
		{
			if (curand_uniform(&state) < 0.01)
			{
				array[tid] = curand(&state) % size;
			}
			else
			{
				array[tid] = tid + 1;
			}
		}
	}
}

bool isSorted(int *array, int size)
{
	for (int i = 0; i < size - 1; i++)
	{
		if (array[i] > array[i + 1])
		{
			return false; // Array is not sorted
		}
	}
	return true; // Array is sorted
}

int main(int argc, char *argv[])
{
	CALI_MARK_BEGIN(main_function);

	srand(time(NULL));
	THREADS = atoi(argv[1]);
	NUM_VALS = atoi(argv[2]);
	size_t size = NUM_VALS * sizeof(int);
	BLOCKS = NUM_VALS / THREADS;
	int *h_array;
	int *d_array;
	int *temp_array;

	h_array = new int[NUM_VALS];

	if (cudaMalloc(&d_array, size) != cudaSuccess)
	{
		cout << "D_ARRAY ALLOCATING NOT WORKING!" << endl;
		return 0;
	}

	if (cudaMalloc(&temp_array, size) != cudaSuccess)
	{
		cout << "TEMP_ARRAY ALLOCATING NOT WORKING!" << endl;
		return 0;
	}
	unsigned int seed = static_cast<unsigned int>(time(NULL));

	CALI_MARK_BEGIN(data_init);

	// Generate the data in parallel
	// Generate in reverse sorted order
	int init_type = atoi(argv[3]);
	generateDataKernel<<<BLOCKS, THREADS>>>(d_array, NUM_VALS, init_type, seed);
	cudaDeviceSynchronize();

	CALI_MARK_END(data_init);

	CALI_MARK_BEGIN(comp);
	CALI_MARK_BEGIN(comp_large);

	launchMergeSort(d_array, temp_array, NUM_VALS);

	cudaDeviceSynchronize();

	CALI_MARK_END(comp_large);
	CALI_MARK_END(comp);

	CALI_MARK_BEGIN(comm);
	CALI_MARK_BEGIN(comm_large);
	CALI_MARK_BEGIN(Memcpy);

	cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

	CALI_MARK_END(Memcpy);
	CALI_MARK_END(comm_large);
	CALI_MARK_END(comm);

	CALI_MARK_BEGIN(correctness_check);

	if (isSorted(h_array, NUM_VALS))
	{
		printf("Ending array is properly sorted!\n");
	}
	else
	{
		printf("Ending array is not properly sorted :(\n");
	}

	CALI_MARK_END(correctness_check);

	// FREEING MEMORY OF CPU & GPU
	delete[] h_array;
	cudaFree(d_array);
	cudaFree(temp_array);
	cudaDeviceReset();

	CALI_MARK_END(main_function);

	// Remaining code for adiak remains unchanged
	const char *algorithm = "MergeSort";
	const char *programmingModel = "CUDA";
	const char *datatype = "int";
	int sizeOfDatatype = sizeof(int);
	int inputSize = NUM_VALS;
	const char *inputType;
	switch (init_type)
	{
	case 0:
		inputType = "Random";
		break;
	case 1:
		inputType = "Sorted";
		break;
	case 2:
		inputType = "ReverseSorted";
		break;
	case 3:
		inputType = "1%Perturbed";
		break;
	default:
		inputType = "Unknown";
	}
	int num_procs = 1;
	int num_threads = THREADS;
	int num_blocks = NUM_VALS / THREADS;
	int group_number = 18;
	const char *implementation_source = "Online and AI";

	adiak::init(NULL);
	adiak::launchdate();										  // launch date of the job
	adiak::libraries();											  // Libraries used
	adiak::cmdline();											  // Command line used to launch the job
	adiak::clustername();										  // Name of the cluster
	adiak::value("Algorithm", algorithm);						  // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
	adiak::value("ProgrammingModel", programmingModel);			  // e.g., "MPI", "CUDA", "MPIwithCUDA"
	adiak::value("Datatype", datatype);							  // The datatype of input elements (e.g., double, int, float)
	adiak::value("SizeOfDatatype", sizeOfDatatype);				  // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("InputSize", inputSize);						  // The number of elements in input dataset (1000)
	adiak::value("InputType", inputType);						  // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
	adiak::value("num_procs", num_procs);						  // The number of processors (MPI ranks)
	adiak::value("num_threads", num_threads);					  // The number of CUDA or OpenMP threads
	adiak::value("num_blocks", num_blocks);						  // The number of CUDA blocks
	adiak::value("group_num", group_number);					  // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
	return 0;
}
