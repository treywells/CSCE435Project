#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <atomic>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int flag = 0;
int SIZE, BLOCKS, THREADS;


__host__ void bubbleSortHost(int *array, int index)
{
	int temp;
	do {

		for (int i = 0; i < SIZE - 1 - index * 2 - flag; i++) {
			if (array[index * 2 + i] > array[index * 2 + 1 + i]) {
				temp = array[index * 2 + 1 + i];
				array[index * 2 + 1 + i] = array[index * 2 + i];
				array[index * 2 + i] = temp;
			}
		}

		flag++;

	} while (SIZE - 1 - index * 2 - flag> 0);
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

__global__ void generateDataKernel(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        array[tid] = size - tid; // Reverse sorted data
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
	srand(time(NULL));
	THREADS = atoi(argv[1]);
    SIZE = atoi(argv[2]);
    BLOCKS = SIZE / THREADS;
	int *h_array;
	int *d_array;
	int offSet;

	h_array = new int[SIZE];
	
	// for (int i = 0; i < SIZE; i++) {
	//     h_array[i] = rand() % SIZE;
	// }


    if (cudaMalloc(&d_array, sizeof(int) * SIZE) != cudaSuccess)
    {
        cout << "D_ARRAY ALLOCATING NOT WORKING!" << endl;
        return 0;
    }

    // Generate the data in parallel
    // Generate in reverse sorted order
    generateDataKernel<<<BLOCKS, THREADS>>>(d_array, SIZE);
    cudaDeviceSynchronize();

    if (isSorted(h_array, SIZE)) {
        printf("Starting array is already sorted!\n");
    } else {
        printf("Starting array is definitely not already sorted!\n");
    }

    if (cudaMemcpy(d_array, h_array, sizeof(int)* SIZE, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "cudaMemcpyHostToDevice ERROR!" << endl;
        cudaFree(d_array);
        return 0;
    }


    do {

        for (int i = 0; i < THREADS * 2; i++) {
            offSet = i;
            // POSSIBLE CHANGE: if offset != 0 USE bubbleSortDeviceParallel << < BLOCKS-1, THREADS >> > (d_array, offSet);
            bubbleSortDeviceParallel << < BLOCKS, THREADS >> > (d_array, offSet, THREADS, BLOCKS);
        }

        BLOCKS--;
    } while (BLOCKS > 0);

    cudaDeviceSynchronize();


    if (cudaMemcpy(h_array, d_array, sizeof(int)* SIZE, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        delete[] h_array;
        cudaFree(d_array);
        cout << "cudaMemcpyDeviceToHost Error" << endl;
        system("pause");
        return 0;
    }

    if (isSorted(h_array, SIZE)) {
        printf("Ending array is properly sorted!\n");
    } else {
        printf("Ending array is not properly sorted :(\n");
    }

	// FREEING MEMORY OF CPU & GPU
	delete[] h_array;
	cudaFree(d_array);
	cudaDeviceReset();

    char algorithm = "BubbleSort"
    char programmingModel = "CUDA"
    char datatype = "int"
    int sizeOfDatatype = sizeof(int);
    int inputSize = SIZE;
    char inputType = "ReverseSorted";
    int num_procs = 1;
    int num_threads = THREADS;
    int num_blocks = BLOCKS;
    int group_number = 18;
    char implementation_source = "Online and AI";



    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

	return 0;
}