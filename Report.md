# CSCE 435 Group project

## 0. Group number: 18

## 1. Group members:
1. Trey Wells
2. Aaron Weast
3. Jacob Miller
4. David Vilenchouk

## 2. Project topic (e.g., parallel sorting algorithms)

 Sorting Algorithm Performance Comparisons for MPI and CUDA

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

For the duration of this project, our team plans on communicating via Slack. 

For our algorithms, we plan on implementing various sorting algorithms. The four sorting algorithms we are planning on implementing are Bubble sort, Merge Sort, quick sort, and insertion sort. 

For each of the algorithms, we are planning on implementating in both MPI and CUDA so that we can compare the differences in CPU vs. GPU parallelization. Not only will we be comparing the differences in CPU and GPU speed but we will also be testing the differences in the algorithms on various types of inputs. For example, we might run each algorithm on a completely random input, then on a partially sorted one, then on a completely sorted one. 

- Quicksort (MPI):
	We have implemented the quicksort algorithm that uses MPI to communicate and function in parallel. It uses a recursive tree method in order to disperse segments to multiple proccsess to sort. It will combine the arrays after sorted.

- Quicksort (CUDA):
	We have implemented the quicksort algorithm using CUDA on the GPU to communicate and function in parallel. It uses an iterative approach to call the kernal which will create partitions of the data for multiple proccess to use until there are not anymore proccesses. It combines the sorted data afterwards.

- BubbleSort (MPI):
	We have implemented the buble sort algorithm using MPI to communiate and function parallel. It is implemented using an even-odd transposition where each processor is responsible for a range of values in the data and then after sorting its personal data, it will communicate and trade data with the processors that are "next" to it. After going through all the phases, it gathers all the data into the root rank. 

- BubbleSort (CUDA):
	We have implemented the bubble sort algorithm using CUDA on the GPU to communicate and function in parallel. This algorithm functions similar to an actual bubble sort where each thread is given a range of values in the data to perform a bubble sort on. Then we call the sorting function again with one less block so that each thread is sorting more and more values till we reach the very end. 

- Mergesort (MPI):
  	We have implemented the mergesort algorithm that uses MPI to communicate in parallel. It uses recursive calls on consecutively smaller halves of an array before gathering the data from the processes to recombine it.
  	Given the Grace outage, I have been unable to test whether this code currently works or not. However, you can clearly see in the Git commit history that I have been trying despite this so that I can test it as soon as Grace is back up.

- Mergesort (CUDA):
  	We have implemented the mergesort algorithm that uses CUDA to function on the GPU. It will partition the array data to multiple processes by splitting the array in half iteratively before recombining the sorted portions.
  	Given the Grace outage, I have been unable to test whether this code currently works or not. However, you can clearly see in the Git commit history that I have been trying despite this so that I can test it as soon as Grace is back up.

- Insertionsort (MPI):
  	We have implemented the insertionsort algorithm that uses MPI to communicate in parallel. It will transfer elements in the array one at a time to the right position.
  	Given the Grace outage, I have been unable to test whether this code currently works or not. However, you can clearly see in the Git commit history that I have been trying despite this so that I can test it as soon as Grace is back up.

- Insertionsort (CUDA):
  	We have implemented the insertionsort algorithm that uses CUDA to function on the GPU. It will transfer elements in the array one at a time to the right position.
  	Given the Grace outage, I have been unable to test whether this code currently works or not. However, you can clearly see in the Git commit history that I have been trying despite this so that I can test it as soon as Grace is back up.

### 2b. Pseudocode for each parallel algorithm

For example:

**- Bubble Sort (MPI)**
	```

	int findPartner(int phase, int rank) {
		int partner;

		/* if it's an even phase */
		if (phase % 2 == 0) {
			/* if we are an even process */
			if (rank % 2 == 0) {
				partner = rank + 1;
			} else {
				partner = rank - 1;
			}
		} else {
			/* it's an odd phase - do the opposite */
			if (rank % 2 == 0) {
				partner = rank - 1;
			} else {
				partner = rank + 1;
			}
		}
		return partner;
	}

	void parallel_sort(int *data, int rank, int count_processes, unsigned long data_size)
	{

		CALI_MARK_BEGIN(comp);
		CALI_MARK_BEGIN(comp_large);

		const unsigned long concat_data_size = data_size * 2;

		auto *other      = new int[data_size];
		auto *concatData = new int[concat_data_size];

		for (int i=0; i<count_processes; i++)
		{
			int partner = findPartner(i, rank);
			if (partner < 0 || partner >= count_processes)
			continue;

			CALI_MARK_END(comp_large);
			CALI_MARK_END(comp);

			CALI_MARK_BEGIN(comm);
			CALI_MARK_BEGIN(comm_large);

			if (rank % 2 == 0) {
				CALI_MARK_BEGIN(MPI_Send);
				MPI_Send(data, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
				CALI_MARK_END(MPI_Send);

				CALI_MARK_BEGIN(MPI_Recv);
				MPI_Recv(other, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				CALI_MARK_END(MPI_Recv);
			} else {
				CALI_MARK_BEGIN(MPI_Recv);
				MPI_Recv(other, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				CALI_MARK_END(MPI_Recv);

				CALI_MARK_BEGIN(MPI_Send);
				MPI_Send(data, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
				CALI_MARK_END(MPI_Send);
			}

			CALI_MARK_END(comm_large);
			CALI_MARK_END(comm);

			CALI_MARK_BEGIN(comp);
			CALI_MARK_BEGIN(comp_large);

			merge(data,  data  + data_size,
				other, other + data_size,
				concatData);
			qsort(concatData, concat_data_size, sizeof(int), compare);

			auto posHalfConcatData = concatData + data_size;
			if (rank < partner)
				copy(concatData, posHalfConcatData, data);
			else
				copy(posHalfConcatData, concatData + concat_data_size, data);
		}

		CALI_MARK_END(comp);
		CALI_MARK_END(comp_large);

	}

	int main(int argc, char** argv)
	{
		MPI_Init(&argc, &argv);

		CALI_MARK_BEGIN(main_function);

		CALI_MARK_BEGIN(data_init);
		int status = fill_vector_with_numbers(data, rank, count_processes, localSize);
		CALI_MARK_END(data_init);

		parallel_sort(data, rank, count_processes, localSize);

		CALI_MARK_BEGIN(comm);
		CALI_MARK_BEGIN(comm_large);
		CALI_MARK_BEGIN(MPI_Gather);

		// Gather data from all processes onto rank 0
		MPI_Gather(data, localSize, MPI_INT, gathered_data, localSize, MPI_INT, 0, MPI_COMM_WORLD);

		CALI_MARK_END(MPI_Gather);
		CALI_MARK_END(comm_large);
		CALI_MARK_END(comm);
		
		// Print the sorted data on rank 0
		if (rank == 0) {
			CALI_MARK_BEGIN(correctness_check);
			if gathered_data is sorted:
				display message to output

			CALI_MARK_END(correctness_check);
		}

		CALI_MARK_END(main_function);

		MPI_Finalize();
	}

	```
    Source: https://github.com/erenalbayrak/Odd-Even-Sort-mit-MPI/blob/master/implementation/c%2B%2B/OddEvenSort.cpp

**- Bubble Sort (CUDA)**
	```

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

	int main(int argc, char* argv[])
	{
		CALI_MARK_BEGIN(main_function);

		srand(time(NULL));
		THREADS = atoi(argv[1]);
		NUM_VALS = atoi(argv[2]);
		size_t size = NUM_VALS * sizeof(int);
		BLOCKS = NUM_VALS / THREADS;
		int *h_array;
		int *d_array;
		int offSet;

		h_array = new int[NUM_VALS];

		if (cudaMalloc(&d_array, size) != cudaSuccess)
		{
			cout << "D_ARRAY ALLOCATING NOT WORKING!" << endl;
			return 0;
		}

		CALI_MARK_BEGIN(data_init);

		// Generate the data in parallel
		// Generate in reverse sorted order
		Generate data in parallel;

		cudaDeviceSynchronize();

		CALI_MARK_END(data_init);

		CALI_MARK_BEGIN(comp);
		CALI_MARK_BEGIN(comp_large);


		do {

			for (int i = 0; i < THREADS * 2; i++) {
				offSet = i;
				// POSSIBLE CHANGE: if offset != 0 USE bubbleSortDeviceParallel << < BLOCKS-1, THREADS >> > (d_array, offSet);
				bubbleSortDeviceParallel << < BLOCKS, THREADS >> > (d_array, offSet, THREADS, BLOCKS);
			}

			BLOCKS--;
		} while (BLOCKS > 0);

		cudaDeviceSynchronize();

		CALI_MARK_END(comp_large);
		CALI_MARK_END(comp);

		CALI_MARK_BEGIN(comm);
		CALI_MARK_BEGIN(comm_large);
		CALI_MARK_BEGIN(memcpyDeviceToHost);

		cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

		CALI_MARK_END(memcpyDeviceToHost);
		CALI_MARK_END(comm_large);
		CALI_MARK_END(comm);

		CALI_MARK_BEGIN(correctness_check);

		if h_array is sorted:
			display message to output

		CALI_MARK_END(correctness_check);

		CALI_MARK_END(main_function);

	}
 	```

    Source: https://github.com/domkris/CUDA-Bubble-Sort/blob/master/CUDABubbleSort/kernel.cu

**- Quick Sort (MPI)**
```  
	#include "mpi.h"
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <stdbool.h>
	
	#include <caliper/cali.h>
	#include <caliper/cali-manager.h>
	#include <adiak.hpp>
	
	using namespace std;
	
	#define SIZE 1000000
	
	/*
	    Divides the array given into two partitions
	    - Lower than pivot
	    - Higher than pivot
	    and returns the Pivot index in the array
	*/
	int partition(int *arr, int low, int high) {
	    int pivot = arr[high];
	    int i = (low - 1);
	    int j, temp;
	    for (j = low; j <= high - 1; j++) {
	        if (arr[j] < pivot) {
	            i++;
	            temp = arr[i];
	            arr[i] = arr[j];
	            arr[j] = temp;
	        }
	    }
	    temp = arr[i + 1];
	    arr[i + 1] = arr[high];
	    arr[high] = temp;
	    return (i + 1);
	}
	
	/*
	    Hoare Partition - Starting pivot is the middle point
	    Divides the array given into two partitions
	    - Lower than pivot
	    - Higher than pivot
	    and returns the Pivot index in the array
	*/
	int hoare_partition(int *arr, int low, int high) {
	    int middle = floor((low + high) / 2);
	    int pivot = arr[middle];
	    int j, temp;
	    // move pivot to the end
	    temp = arr[middle];
	    arr[middle] = arr[high];
	    arr[high] = temp;
	
	    int i = (low - 1);
	    for (j = low; j <= high - 1; j++) {
	        if (arr[j] < pivot) {
	            i++;
	            temp = arr[i];
	            arr[i] = arr[j];
	            arr[j] = temp;
	        }
	    }
	    // move pivot back
	    temp = arr[i + 1];
	    arr[i + 1] = arr[high];
	    arr[high] = temp;
	
	    return (i + 1);
	}
	
	/*
	    Simple sequential Quicksort Algorithm
	*/
	void quicksort(int *number, int first, int last) {
	    if (first < last) {
	        int pivot_index = partition(number, first, last);
	        quicksort(number, first, pivot_index - 1);
	        quicksort(number, pivot_index + 1, last);
	    }
	}
	
	/*
	    Functions that handle the sharing of subarrays to the right clusters
	*/
	int quicksort_recursive(int *arr, int arrSize, int currProcRank, int maxRank, int rankIndex) {
	    MPI_Status status;
	
	    // Calculate the rank of the Cluster to which I'll send the other half
	    int shareProc = currProcRank + pow(2, rankIndex);
	    // Move to a lower layer in the tree
	    rankIndex++;
	
	    // If no Cluster is available, sort sequentially by yourself and return
	    if (shareProc > maxRank) {
	        MPI_Barrier(MPI_COMM_WORLD);
	        CALI_MARK_BEGIN("quicksort");
	        quicksort(arr, 0, arrSize - 1);
	        CALI_MARK_END("quicksort");
	        return 0;
	    }
	    // Divide the array into two parts with the pivot in between
	    int j = 0;
	    int pivotIndex;
	
	    
	    
	    // CALI_MARK_BEGIN("comp_small");
	    CALI_MARK_BEGIN("partition");
	    pivotIndex = hoare_partition(arr, j, arrSize - 1);
	    CALI_MARK_END("partition");
	    // CALI_MARK_END("comp_small");
	    
	    
	
	
	    // Send a partition based on size (always send the smaller part),
	    // Sort the remaining partitions,
	    // Receive the sorted partition
	    if (pivotIndex <= arrSize - pivotIndex) {
	        MPI_Send(arr, pivotIndex, MPI_INT, shareProc, pivotIndex, MPI_COMM_WORLD);
	        quicksort_recursive((arr + pivotIndex + 1), (arrSize - pivotIndex - 1), currProcRank, maxRank, rankIndex);
	        MPI_Recv(arr, pivotIndex, MPI_INT, shareProc, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	    } else {
	        MPI_Send((arr + pivotIndex + 1), arrSize - pivotIndex - 1, MPI_INT, shareProc, pivotIndex + 1, MPI_COMM_WORLD);
	        quicksort_recursive(arr, (pivotIndex), currProcRank, maxRank, rankIndex);
	        MPI_Recv((arr + pivotIndex + 1), arrSize - pivotIndex - 1, MPI_INT, shareProc, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	    }
	}
	
	// Function to initialize the data (generate random input array)
	void data_init(int *arr, int size) {
	    int j;
	    for (j = 0; j < size; ++j) {
	        arr[j] = (int) rand() % 1000;
	    }
	}
	
	// Function to check the correctness of the sorted array
	bool correctness_check(int *arr, int size) {
	    int i;
	    for (i = 0; i < size - 1; i++) {
	        if (arr[i] > arr[i + 1]) {
	            return false;
	        }
	    }
	    return true;
	}
	
	int main(int argc, char *argv[]) {
	    CALI_MARK_BEGIN("main");
	    
	    if (argc != 2) {
	        printf("Usage: %s <array_size>\n", argv[0]);
	        return 1;
	    }
	
	    int array_size = atoi(argv[1]);
	
	    int unsorted_array[array_size];
	    int size, rank;
	
	    // Start Parallel Execution
	    CALI_MARK_BEGIN("MPI_init");
	    MPI_Init(&argc, &argv);
	    CALI_MARK_END("MPI_init");
	    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	    cali::ConfigManager mgr;
	    mgr.start();
	
	    if (rank == 0) {
	        // --- RANDOM ARRAY GENERATION ---
	        printf("Creating a Random List of %d elements\n", array_size);
	        CALI_MARK_BEGIN("data_init");
	        data_init(unsorted_array, array_size);
	        CALI_MARK_END("data_init");
	        printf("Created\n");
	    }
	
	    // Calculate in which layer of the tree each Cluster belongs
	    int rankPower = 0;
	    while (pow(2, rankPower) <= rank) {
	        rankPower++;
	    }
	    // Wait for all clusters to reach this point
	    CALI_MARK_BEGIN("comm");
	    CALI_MARK_BEGIN("MPI_Barrier");
	    MPI_Barrier(MPI_COMM_WORLD);
	    CALI_MARK_END("MPI_Barrier");
	    CALI_MARK_END("comm");
	    double start_timer, finish_timer;
	
	    if (rank == 0) {
	        start_timer = MPI_Wtime();
	        // Cluster Zero (Master) starts the Execution and
	        // always runs recursively and keeps the left bigger half
	        CALI_MARK_BEGIN("comp");
	        CALI_MARK_BEGIN("comp_large");
	        CALI_MARK_BEGIN("recursive_proccess_0");
	        quicksort_recursive(unsorted_array, array_size, rank, size - 1, rankPower);
	        CALI_MARK_END("recursive_proccess_0");
	        CALI_MARK_END("comp_large");
	        CALI_MARK_END("comp");
	    } else {
	        // All other Clusters wait for their subarray to arrive,
	        // they sort it and they send it back.
	        MPI_Status status;
	        int subarray_size;
	        CALI_MARK_BEGIN("comm");
	        CALI_MARK_BEGIN("MPI_probe");
	        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	        CALI_MARK_END("MPI_probe");
	
	        // Capturing the size of the array to receive
	        CALI_MARK_BEGIN("comm_small");
	        CALI_MARK_BEGIN("MPI_get_count");
	        MPI_Get_count(&status, MPI_INT, &subarray_size);
	        CALI_MARK_END("MPI_get_count");
	        CALI_MARK_END("comm_small");
	
	        int source_process = status.MPI_SOURCE;
	        int subarray[subarray_size];
	
	        CALI_MARK_BEGIN("comm_large");
	        CALI_MARK_BEGIN("MPI_recv_>0");
	        MPI_Recv(subarray, subarray_size, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	        CALI_MARK_END("MPI_recv_>0");
	        CALI_MARK_END("comm_large");
	        CALI_MARK_END("comm");
	
	        CALI_MARK_BEGIN("comp");
	        CALI_MARK_BEGIN("comp_large");
	        CALI_MARK_BEGIN("recursive>0");
	        quicksort_recursive(subarray, subarray_size, rank, size - 1, rankPower);
	        CALI_MARK_END("recursive>0");
	        CALI_MARK_END("comp_large");
	        CALI_MARK_END("comp");
	
	        CALI_MARK_BEGIN("comm");
	        CALI_MARK_BEGIN("comm_large");
	        CALI_MARK_BEGIN("MPI_Send>0");
	        MPI_Send(subarray, subarray_size, MPI_INT, source_process, 0, MPI_COMM_WORLD);
	        CALI_MARK_END("MPI_Send>0");
	        CALI_MARK_END("comm_large");
	        CALI_MARK_END("comm");
	    };
	
	    if (rank == 0) {
	        finish_timer = MPI_Wtime();
	        printf("Total time for %d Proccesses: %2.2f sec\n", size, finish_timer - start_timer);
	
	        // --- VALIDATION CHECK ---
	        printf("Checking...\n");
	        CALI_MARK_BEGIN("correctness_check");
	        bool sorted_correctly = correctness_check(unsorted_array, array_size);
	        CALI_MARK_END("correctness_check");
	        if (sorted_correctly) {
	            printf("Sorted correctly!\n");
	        } else {
	            printf("Error: Not sorted correctly\n");
	        }
	    }
	    string algorithm = "QuickSort";
	    string programmingModel = "MPI";
	    string datatype = "int";
	    int sizeOfDatatype = sizeof(int);
	    int inputSize = array_size;
	    string inputType = "Random";
	    int num_procs = size;
	    string group_number = "18";
	    string implementation_source = "Online";
	
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
	    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
	    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
	
	    CALI_MARK_END("main");
	    mgr.stop();
	    mgr.flush();
	
	    MPI_Finalize();
	    // End of Parallel Execution
	    
	}

 	```
 
source: https://github.com/triasamo1/Quicksort-Parallel-MPI/blob/master/quicksort_mpi.c

**- Quick Sort (CUDA)**
  
	```   
	#include <stdio.h>
	#include <stdlib.h>
	#include <time.h>
	
	#include <caliper/cali.h>
	#include <caliper/cali-manager.h>
	#include <adiak.hpp>
	
	using namespace std;
	
	
	void printArr(int arr[], int n) {
	    for (int i = 0; i < n; ++i)
	        printf("%d ", arr[i]);
	    printf("\n");
	}
	
	__device__ int d_size;
	
	__global__ void partition(int *arr, int *arr_l, int *arr_h, int n) {
	    
	    int z = blockIdx.x * blockDim.x + threadIdx.x;
	    d_size = 0;
	    __syncthreads();
	    if (z < n) {
	        int h = arr_h[z];
	        int l = arr_l[z];
	        int x = arr[h];
	        int i = (l - 1);
	        int temp;
	        for (int j = l; j <= h - 1; j++) {
	            if (arr[j] <= x) {
	                i++;
	                temp = arr[i];
	                arr[i] = arr[j];
	                arr[j] = temp;
	            }
	        }
	        temp = arr[i + 1];
	        arr[i + 1] = arr[h];
	        arr[h] = temp;
	        int p = (i + 1);
	        if (p - 1 > l) {
	            int ind = atomicAdd(&d_size, 1);
	            arr_l[ind] = l;
	            arr_h[ind] = p - 1;
	        }
	        if (p + 1 < h) {
	            int ind = atomicAdd(&d_size, 1);
	            arr_l[ind] = p + 1;
	            arr_h[ind] = h;
	        }
	    }
	    
	}
	
	// void quickSortIterative(int arr[], int l, int h, int threads_per_block, int size) {
	    
	    
	//     int lstack[h - l + 1], hstack[h - l + 1];
	
	//     int top = -1, *d_d, *d_l, *d_h;
	
	//     lstack[++top] = l;
	//     hstack[top] = h;
	
	//     CALI_MARK_BEGIN("comm");
	//     CALI_MARK_BEGIN("comm_large");
	
	//     cudaMalloc(&d_d, (h - l + 1) * sizeof(int));
	//     cudaMemcpy(d_d, arr, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	//     cudaMalloc(&d_l, (h - l + 1) * sizeof(int));
	//     cudaMemcpy(d_l, lstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	//     cudaMalloc(&d_h, (h - l + 1) * sizeof(int));
	//     cudaMemcpy(d_h, hstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	//     CALI_MARK_END("comm_large");
	//     CALI_MARK_END("comm");
	
	//     int n_t = threads_per_block;
	//     int n_b = size / threads_per_block;
	//     int n_i = 1;
	    
	//     CALI_MARK_BEGIN("comp");
	//     CALI_MARK_BEGIN("comp_large");
	//     while (n_i > 0) {
	//         partition<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
	//         int answer;
	//         CALI_MARK_BEGIN("comm");
	//         CALI_MARK_BEGIN("comm_small");
	//         cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
	//         n_t = threads_per_block;
	//         n_i = answer;
	//         cudaMemcpy(arr, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	//         CALI_MARK_END("comm_small");
	//         CALI_MARK_END("comm");
	//     }
	//     CALI_MARK_END("comp_large");
	//     CALI_MARK_END("comp");
	    
	// }
	
	void data_init(int arr[], int n) {
	    srand(time(NULL));
	    for (int i = 0; i < n; i++) {
	        arr[i] = rand() % 10000;
	    }
	}
	
	bool isSorted(int arr[], int n) {
	    for (int i = 0; i < n - 1; ++i) {
	        if (arr[i] > arr[i + 1]) {
	            return false;
	        }
	    }
	    return true;
	}
	
	void correctness_check(int arr[], int n) {
	    if (isSorted(arr, n)) {
	        printf("Array is correctly sorted.\n");
	    } else {
	        printf("Array is NOT correctly sorted.\n");
	    }
	}
	
	int main(int argc, char *argv[]) {
	    CALI_MARK_BEGIN("main");
	    
	    if (argc != 3) {
	        printf("Usage: %s <threads_per_block> <input_size>\n", argv[0]);
	        return 1;
	    }
	
	    int threads_per_block = atoi(argv[1]);
	    int input_size = atoi(argv[2]);
	
	    cali::ConfigManager mgr;
	    mgr.start();
	
	    int arr[input_size];
	
	    CALI_MARK_BEGIN("data_init");
	    data_init(arr, input_size);
	    CALI_MARK_END("data_init");
	
	    int n = sizeof(arr) / sizeof(*arr);
	    printf("Number of threads per block: %d\n", threads_per_block);
	    printf("Input size: %d\n", input_size);
	
	    // printf("Array before sorting: ");
	    // printArr(arr, n);
	
	    // quickSortIterative(arr, 0, n - 1, threads_per_block, input_size);
	
	    int l = 0;
	    int h = n-1;
	    int size = input_size;
	
	    int lstack[h - l + 1], hstack[h - l + 1];
	
	    int top = -1, *d_d, *d_l, *d_h;
	
	    lstack[++top] = l;
	    hstack[top] = h;
	
	    CALI_MARK_BEGIN("comm");
	    CALI_MARK_BEGIN("comm_large");
	    CALI_MARK_BEGIN("cudaMemcpy");
	
	    cudaMalloc(&d_d, (h - l + 1) * sizeof(int));
	    cudaMemcpy(d_d, arr, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	    cudaMalloc(&d_l, (h - l + 1) * sizeof(int));
	    cudaMemcpy(d_l, lstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	    cudaMalloc(&d_h, (h - l + 1) * sizeof(int));
	    cudaMemcpy(d_h, hstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	    CALI_MARK_END("cudaMemcpy");
	    CALI_MARK_END("comm_large");
	    CALI_MARK_END("comm");
	
	    int n_t = threads_per_block;
	    int n_b = size / threads_per_block;
	    int n_i = 1;
	    
	    
	    while (n_i > 0) {
	        CALI_MARK_BEGIN("comp");
	        CALI_MARK_BEGIN("comp_large");
	        CALI_MARK_BEGIN("kernal_partition");
	        partition<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
	        CALI_MARK_END("kernal_partition");
	        CALI_MARK_END("comp_large");
	        CALI_MARK_END("comp");
	        int answer;
	        CALI_MARK_BEGIN("comm");
	        CALI_MARK_BEGIN("comm_small");
	        CALI_MARK_BEGIN("cudaMemcpySymbol");
	        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
	        CALI_MARK_END("cudaMemcpySymbol");
	        CALI_MARK_END("comm_small");
	        
	        n_t = threads_per_block;
	        n_i = answer;
	        CALI_MARK_BEGIN("comm_large");
	        CALI_MARK_BEGIN("cudaMemcpySortedinLoop");
	        cudaMemcpy(arr, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	        CALI_MARK_END("cudaMemcpySortedinLoop");
	        CALI_MARK_END("comm_large");
	        CALI_MARK_END("comm");
	    }
	    
	
	    string algorithm = "QuickSort";
	    string programmingModel = "CUDA";
	    string datatype = "int";
	    int sizeOfDatatype = sizeof(int);
	    int inputSize = input_size;
	    string inputType = "Random";
	    int num_threads = threads_per_block;
	    int num_blocks = input_size / threads_per_block;
	    string group_number = "18";
	    string implementation_source = "Online";
	
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
	    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
	    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks
	    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
	    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
	
	    // printf("Array after sorting: ");
	    // printArr(arr, n);
	    CALI_MARK_BEGIN("correctness_check");
	    correctness_check(arr, n);
	    CALI_MARK_END("correctness_check");
	    CALI_MARK_END("main");
	    
	    mgr.stop();
	    mgr.flush();
	
	    return 0;
	}
	```
 
source: https://github.com/GreyVader1993/Cuda-Programs/blob/main/QuickSort.cu

**- Merge Sort (MPI)**

	```
	**function merge**(a, b, l, m, r):
	    h = l
	    i = l
	    j = m + 1
	
	    while (h <= m) and (j <= r):
	        if a[h] <= a[j]:
	            b[i] = a[h]
	            h++
	        else:
	            b[i] = a[j]
	            j++
	        i++
	
	    if m < h:
	        for k = j to r:
	            b[i] = a[k]
	            i++
	    else:
	        for k = h to m:
	            b[i] = a[k]
	            i++
	
	    for k = l to r:
	        a[k] = b[k]
	
	
	**function mergeSort**(a, b, l, r):
	    if l < r:
	        m = (l + r) / 2
	
	        mergeSort(a, b, l, m)
	        mergeSort(a, b, (m + 1), r)
	        merge(a, b, l, m, r)
	
	
	**function main**(argc, argv):
	    n = atoi(argv[1])
	    original_array = malloc(n * sizeof(int))
	
	    // Populate the array with random values
	    for c = 0 to n - 1:
	        original_array[c] = rand() % n
	
	    // Initialize MPI
	    MPI_INIT(&argc, &argv)
	    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank)
	    MPI_Comm_size(MPI_COMM_WORLD, &world_size)
	
	    // Divide the array into equal-sized chunks
	    size = n / world_size
	
	    // Scatter the subarrays to each process
	    sub_array = malloc(size * sizeof(int))
	    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD)
	
	    // Perform mergesort on each process
	    tmp_array = malloc(size * sizeof(int))
	    mergeSort(sub_array, tmp_array, 0, (size - 1))
	
	    // Gather the sorted subarrays into one
	    sorted = NULL
	    if world_rank == 0:
	        sorted = malloc(n * sizeof(int))
	
	    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD)
	
	    // Make the final mergesort call on the root process
	    if world_rank == 0:
	        other_array = malloc(n * sizeof(int))
	        mergeSort(sorted, other_array, 0, (n - 1))
	
	        // Check sorted array
	        for c = 0 to n - 1:
	            if sorted[c] > sorted[c + 1]:
	                print("Sort Failed")
	                break
	
	        print("Sort Successful")
	
	
	    // Finalize MPI
	    MPI_Barrier(MPI_COMM_WORLD)
	    MPI_Finalize()

	```

    Source: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c
  

**- Merge Sort (CUDA)**
  
        function solve(tempList, left_start, right_start, old_left_start, my_start, my_end, left_end, right_end, headLoc, walkLen):
    minRemaining = min(right_end - right_start, left_end - left_start)

    for i = 0 to minRemaining - 1:
        if tempList[current_list][left_start] < tempList[current_list][right_start]:
            tempList[!current_list][headLoc] = tempList[current_list][left_start]
            left_start++
        else:
            tempList[!current_list][headLoc] = tempList[current_list][right_start]
            right_start++

        headLoc++

function Device_Merge(d_list, length, elementsPerThread):
    my_start, my_end, left_start, right_start, old_left_start, left_end, right_end, headLoc = 0, 0, 0, 0, 0, 0, 0, 0
    current_list = 0

    shared_memory tempList[2][SHARED / sizeof(int)]

    index = blockIdx.x * blockDim.x + threadIdx.x

    for i = 0 to elementsPerThread - 1:
        if index + i < length:
            tempList[current_list][elementsPerThread * threadIdx.x + i] = d_list[index + i]

    synchronize_threads()

    for walkLen = 1 to length - 1 step 2:
        my_start = elementsPerThread * threadIdx.x
        my_end = my_start + elementsPerThread
        left_start = my_start

        while left_start < my_end:
            old_left_start = left_start

            if left_start > my_end:
                left_start = length
                break

            left_end = left_start + walkLen
            if left_end > my_end:
                left_end = length

            right_start = left_end

            if right_start > my_end:
                right_end = length

            right_end = right_start + walkLen

            if right_end > my_end:
                right_end = length

            solve(tempList, left_start, right_start, old_left_start, my_start, my_end, left_end, right_end, headLoc, walkLen)
            left_start = old_left_start + 2 * walkLen
            current_list = !current_list

    synchronize_threads()

    for i = 0 to elementsPerThread - 1:
        if index + i < length:
            d_list[index + i] = tempList[current_list][elementsPerThread * threadIdx.x + i]

    synchronize_threads()

function MergeSort(h_list, len, threadsPerBlock, blocks):
    d_list = allocate_device_memory(len)

    copy_host_to_device(h_list, d_list, len)

    elementsPerThread = ceil(static_cast<float>(NUM_VALS) / (threadsPerBlock * blocks))

    Device_Merge<<<blocks, threadsPerBlock>>>(d_list, NUM_VALS, elementsPerThread)

    copy_device_to_host(d_list, h_list, len)

    free_device_memory(d_list)

function main(argc, argv):
    THREADS = atoi(argv[1])
    NUM_VALS = atoi(argv[2])
    BLOCKS = NUM_VALS / THREADS

    initialize_cali_ConfigManager()

    h_list = allocate_host_memory(NUM_VALS)

    for i = 0 to NUM_VALS - 1:
        h_list[i] = random() % NUM_VALS

    MergeSort(h_list, NUM_VALS, THREADS, BLOCKS)

    check_correctness(h_list, NUM_VALS)

    adiak_setup_and_logging()

    return 0

    
Source: https://github.com/pushkarkrp/Parallel-Computing/blob/master/BLOG6/merge.cu


**- Insertion Sort (MPI)**
```
	int main ( int argc, char *argv[] )
	{
		int i,p,*n,j,g,s;
		MPI_Status status;
		MPI_Init(&argc,&argv);
		MPI_Comm_size(MPI_COMM_WORLD,&p);
		MPI_Comm_rank(MPI_COMM_WORLD,&i);
		if(i==0) /* manager generates p random numbers */
		{
			n = (int*)calloc(p,sizeof(int));
			srand(time(NULL));
			for(j=0; j<p; j++) n[j] = rand() % 100;
			if(verbose>0)
			{
				printf("The %d numbers to sort : ",p);
				for(j=0; j<p; j++) printf(" %d", n[j]);
				printf("\n"); fflush(stdout);
			}
		}
		for(j=0; j<p-i; j++) /* processor i performs p-i steps */
		if(i==0)
		{
			g = n[j];
			if(verbose>0){
				printf("Manager gets %d.\n",n[j]); fflush(stdout);
			}
			Compare_and_Send(i,j,&s,&g);
		}
		else
		{
			MPI_Recv(&g,1,MPI_INT,i-1,tag,MPI_COMM_WORLD,&status);
			if(verbose>0){
				printf("Node %d receives %d.\n",i,g); fflush(stdout);
			}
			Compare_and_Send(i,j,&s,&g);
		}
		MPI_Barrier(MPI_COMM_WORLD); /* to synchronize for printing */
		Collect_Sorted_Sequence(i,p,s,n);
		MPI_Finalize();
		return 0;
	}
	void Compare_and_Send
	( int myid, int step, int *smaller, int *gotten )
	/* Processor "myid" initializes smaller with gotten
	* at step zero, or compares smaller to gotten and
	* sends the larger number through. */
	{
		if(step==0)
			*smaller = *gotten;
		else
			if(*gotten > *smaller)
			{
				MPI_Send(gotten,1,MPI_INT,myid+1,tag,MPI_COMM_WORLD);
				if(verbose>0)
				{
					printf("Node %d sends %d to %d.\n",
					myid,*gotten,myid+1);
					fflush(stdout);
				}
			}
		else
		{
			MPI_Send(smaller,1,MPI_INT,myid+1,tag,
			MPI_COMM_WORLD);
			if(verbose>0)
			{
				printf("Node %d sends %d to %d.\n",
				myid,*smaller,myid+1);
				fflush(stdout);
			}
			*smaller = *gotten;
		}
	}
	void Collect_Sorted_Sequence
		( int myid, int p, int smaller, int *sorted ) {
		/* Processor "myid" sends its smaller number to the
		* manager who collects the sorted numbers in the
		* sorted array, which is then printed. */
		MPI_Status status;
		int k;
		if(myid==0) {
			sorted[0] = smaller;
			for(k=1; k<p; k++)
			MPI_Recv(&sorted[k],1,MPI_INT,k,tag,
			MPI_COMM_WORLD,&status);
			printf("The sorted sequence : ");
			for(k=0; k<p; k++) printf(" %d",sorted[k]);
			printf("\n");
		}
		else
			MPI_Send(&smaller,1,MPI_INT,0,tag,MPI_COMM_WORLD);
	}
```
Source: https://homepages.math.uic.edu/~jan/mcs572/pipelinedsort.pdf

**- Insertion Sort (CUDA)**
```
	#include <stdio.h>
	#include <stdlib.h>
	
	#define N 16
	
	__global__ void insertionsort(int n, const float *values, int *indices) { 
	  int key_i, j; 
	  for (int i = blockIdx.x; i < n; i += gridDim.x) {
	    key_i = indices[i];
	    j = i - 1; 
	    while (j >= 0 && values[indices[j]] > values[key_i]) { 
	      indices[j + 1] = indices[j];
	      j = j - 1; 
	    } 
	    indices[j + 1] = key_i; 
	  } 
	}
	
	/**
	  * Indices need not to be copied. They will be set in the function itself.
	  */
	__global__ void argsort(int n, const float* values, int *indices) {
	  for (int i = blockIdx.x; i < n; i += gridDim.x) {
	    indices[i] = i;
	  }
	  __syncthreads();
	}
	
	int main() {
	    // The h prefix stands for host
	    float h_values[N];
	    int h_indices[N];
	
	    // The d prefix stands for device
	    float *d_values;
	    int *d_indices;
	    cudaMalloc((void **)&d_values, N*sizeof(float));
	    cudaMalloc((void **)&d_indices, N*sizeof(int));
	
	    // Random d_valuesta
	    for (int i = 0; i<N; ++i) {
	      h_values[i] = rand() % 100;
	    }
	
	    // Copy values to GPU
	    cudaMemcpy(d_values, h_values, N*sizeof(float), cudaMemcpyHostToDevice);
	
	    // Launch GPU with N threads
	    argsort<<<N, 1>>>(N, d_values, d_indices);
	    insertionsort<<<N, 1>>>(N, d_values, d_indices);
	
	    // Copy indices back
	    cudaMemcpy(h_indices, d_indices, N*sizeof(int), cudaMemcpyDeviceToHost);
	
	    printf("Indices:\n");
	    for (int i = 0; i<N; ++i) {
	        printf("%i\n", h_indices[i]);
	    }
	    
	    printf("Values (should now be sorted):\n");
	    for (int i = 0; i<N; ++i) {
	        printf("%f\n", h_values[h_indices[i]]);
	    }
	
	    // Free up the arrays on the GPU.
	    cudaFree(d_values);
	    cudaFree(d_indices);
	
	    return 0;
	}
```
Source: https://gist.github.com/mrquincle/f738daa6bd27367c09d0f6ae81fd6ca2
```
## 4. Project Evaluation 

Quicksort (MPI):

Looking at these graphs, we can see that for all the input types, the time decreases as we increase the number of proccesses for main and comp. Specifically, sorted and random input seemed to scale the best with drastic decreases in runtime for higher number of proccesses. Additionally, we can see that communication time increases as we increase the number of processes.

Note: this algorithm could only scale to 2^20 input size due to the maximum value allowed for "tag" in MPI_SEND. The algorithm sends the pivot index which is the middle of the array to divide into halves so each half goes to one side of the recursive tree. Therefore, this number reaches 2000000 for 2^22 input which causes it to be above the max allowed value.

-Sorted Input:

![image](https://github.com/treywells/CSCE435Project/assets/98286168/d23359f5-3ea4-4d3b-b146-c4d59158d015)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/a2be2362-357a-4cc7-a917-ff019740b954)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/bbe3616e-1ceb-4ef3-91fa-ed8ac833e42f)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/f895f6c0-cca5-4e87-874a-71ccbdda5ef9)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/cfa150ad-fe67-482e-bf77-f20a328ec854)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/3a77849e-14d0-4879-ad9b-5481796eb2b5)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/e676d98c-855f-4402-834b-9147c988b13f)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/9d8afda9-f2d3-4bd3-af65-035dee9a42d9)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/b6546adf-4d4b-416f-af3e-29778b59e664)



- Random Input:

![image](https://github.com/treywells/CSCE435Project/assets/98286168/bae5c5ca-3ef7-4819-bd40-99f6146e4517)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/5902b593-f067-4138-8500-c81ded2e9756)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/2b547bf0-5ecb-46b0-9a95-1b28cbc490d1)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/659e0c52-6316-4be5-921f-f470caaa582a)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/3c510257-7d4c-44f7-9fef-30b0ebc4ab4a)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/da05aef6-cef3-4327-9a97-532e8ca7cce4)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/806c90f1-f619-4f51-b9da-27b3bc434ad8)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/db7d7925-2283-4b04-8d58-897805695cbb)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/ff4f335c-ca31-4a88-8e10-ccf884710139)

- Reverse Sorted:
![image](https://github.com/treywells/CSCE435Project/assets/98286168/b89d6a83-489e-426c-bcb5-6092f697c68c)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/77648296-cb57-4603-81f9-0113ee3baa23)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/18b69f0c-c367-4f47-8e32-b710b398a1cc)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/a7d4c80c-1429-4d23-a235-ef24017f9b05)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/54a3cc9b-a549-4819-b014-8ada6a32cdac)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/36f84b6e-86d5-405e-ba15-9bdbc221e6f4)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/af04b6ac-6a5f-4cb4-bdc7-033a2ddc976c)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/319f6d7a-0f38-4ea5-9260-4f0cdd272e7b)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/6fb7dc8f-33fd-490b-9323-39f12a9e17e6)

 

- 1% Perturbed:
![image](https://github.com/treywells/CSCE435Project/assets/98286168/d57c8196-6e60-4213-88a6-cfbc52a113b4)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/07e153ea-981d-49cf-b854-ccf361614808)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/9fe5d1fe-62e2-4ea4-b498-9a90eda35172)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/83837ec8-02f2-40b1-a0c6-590637650eb8)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/8f11da51-d889-4471-9871-1872ea5fe312)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/a494508b-7548-4d45-8e12-fb83fe8bb162)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/e4640989-e75f-4db6-9c6d-27debf2066c0)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/728b9596-1782-443b-bf91-8f9ec007e539)
![image](https://github.com/treywells/CSCE435Project/assets/98286168/ff618321-dadc-4fd0-b72c-1bdd09304f86)
  
 
  Quicksort (CUDA):

  The time of the GPU increased for smaller input sizes, but then started to decrease as we get to larger problems. The optimal thread per block size seemed to 
  be 256 in these larger problem cases. Additionally, we can see that the time for the Comm regions increased as we increased the number of threads per block. 

  - Random Input:
    
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/8dbb80ef-6bd2-4b93-83cf-7666736e9dce)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/425e2cbd-a0df-4214-834e-285eff514b03)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/bfddbdc1-9c2e-4165-95c4-a23575aecded)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/805dd24a-304d-4dc8-a4d0-8555f42a083f)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/4f81a708-771b-461c-9f45-ce1852fc4e53)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/0a2bad7e-72c3-49a2-8f38-6338e9d9c9a2)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/40cf6870-7cb8-42dc-bc79-7d514c2e9a2e)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/27cd2858-d096-4ec4-9f19-e7a3f4384f41)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/76d8e572-fa5e-45c6-9f31-0a850d52b262)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/e1d85ce6-a2f1-4398-928e-0fffc12c8a4b)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/50c62b8f-9329-45d0-86b9-d706d7541ed2)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/0981be80-17e1-4654-9e25-f09ab9755e35)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/4d1ed903-2716-4438-bc9a-e1996920ce3d)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/b74aa991-58e1-4442-a4fb-0c7b335be1c3)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/b07af483-d955-441d-ae35-e128d5ec25fe)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/b31a2711-de52-4e55-953a-1fc69a930c02)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/23297f18-bd35-453c-b08d-2ddbe9e324df)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/5737e82e-5a58-4537-ad82-f93015aa20c6)

  - Sorted
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/e6b03607-9af0-4809-8f18-5c869a3ae36f)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/7d596565-6d4c-4c85-b824-40b0f388d0f3)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/3a129cc9-f60b-49d9-a4c8-2bfdc23d0f3f)

  - Reverse Sorted
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/4f8a2709-fce9-41fc-a3f6-e8431200821d)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/e762415f-718b-4abd-90ad-40429d3a4ff3)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/9820e3de-6ddd-48db-b30d-dac28c41383f)

  - 1% Perturbed
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/f8031dd9-f14c-4962-a010-8c1e1627cde4)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/d6ff150b-ac12-497d-9209-f71878e8a194)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/569bef3b-8cea-4142-8743-fe736271929e)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/2428e5ee-3726-423e-bf75-061133c3f975)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/a0bb5db1-1b61-43c9-bdc6-a2797548496c)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/02d70231-8201-4c3e-8fa2-6e862cf11742)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/6c05a5dd-cc9a-40ea-86ee-2e5d5848c0f5)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/0f19ef98-5381-4625-a71b-e865f780e65c)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/2f1fc65d-0db7-402c-b69e-4c2513cb7faf)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/602b2d37-a6c5-4206-bfce-59652f36dd3a)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/bfd71f93-7802-4248-992c-e6ce891507f8)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/4956ad00-f547-4ca9-9fe7-3f25d43de771)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/eeea7301-7a7e-4fad-98c4-d325fd17cddc)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/2dc5e75c-8d0a-4092-8d01-18a6f378a347)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/295daaf0-7ba7-4a85-a3be-8e93e4fb9d8a)
  ![image](https://github.com/treywells/CSCE435Project/assets/98286168/1d688c72-a4c3-4a59-bfb8-37467b520aff)
  



  
Bubble (MPI):

  The Max time/rank for practically every input size increased when we added more and more processes. This leads us to believe that the optimal number of processors for our implementation of the MPI Odd-Even Sort is approximately 2 or 4 processors. 
  We of course expect the communication time to increase as we add more processors but even the computation time is increasing, which is definitely not expected. I personally believe that this is due to the fact that more processors means more phases and thus a longer sorting time.

  - Random Input:

	![Alt text](image-3.png)
	![Alt text](image-4.png)
	![Alt text](image-5.png)
	![Alt text](image-6.png)
	![Alt text](image-7.png)
	![Alt text](image-8.png)
	![Alt text](image-9.png)
	![Alt text](image-10.png)
	![Alt text](image-11.png)
	![Alt text](image-12.png)
	![Alt text](image-13.png)
	![Alt text](image-14.png)
	![Alt text](image-15.png)
	![Alt text](image-16.png)
	![Alt text](image-17.png)
	![Alt text](image-18.png)
	![Alt text](image-19.png)
	![Alt text](image-20.png)
	![Alt text](image-21.png)
	![Alt text](image-22.png)
	![Alt text](image-23.png)
	![Alt text](image-24.png)

Bubble (CUDA):

  The Max time/rank for the CUDA implementation of the Bubble sort followed a similar pattern to the MPI implementation where increasing the threads/block roughly increases the execution time for the algorithm as a whole. This leads us to believe that the optimal amount of threads for the CUDA Bubble implementation is approximately 128 to 256 threads. 
  Like MPI, the communication time is expected to increase, and it did, but the computation time also increase which leads us to believe that Bubble Sort as a whole is not the best option for parallel computing.

  When we do parallelize it, it will perform much better than the serial counterpart but adding more and more parallelization will create an overhead that is not outweighed by the increase in speed.

  - Random Input:

	![Alt text](image-24.png)
	![Alt text](image-25.png)
	![Alt text](image-26.png)
	![Alt text](image-27.png)
	![Alt text](image-28.png)
	![Alt text](image-29.png)
	![Alt text](image-30.png)
	![Alt text](image-31.png)
	![Alt text](image-32.png)
	![Alt text](image-33.png)
	![Alt text](image-34.png)
	![Alt text](image-35.png)

- Mergesort(MPI):
  
  The below graphs were generated using my mergesort MPI algorithm on a sorted list of integers. Initial analysis seems to indicate that my mergesort MPI implementation does not scale well in the slightest apart from a couple of interesting exceptions. For whatever reason, the average time per rank decreases signficantly for only a few types numbers of threads. This is inconsistent which could indicate an error on my part when generating the data, but it is known that mergesort is ineffecient for parallelism in general so the graphs could just be a consequence of that.  
  
![image](https://github.com/treywells/CSCE435Project/assets/112406802/cadd4801-282b-45b8-9959-fc29f139c16a)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/8a290c56-4eae-436f-b105-f939840af21b)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/ddc5a9c6-0c78-4028-8f37-03b49c82cbcd)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/f01b39d5-d1f8-4a9c-8c39-572fe5529b92)

- Mergesort Cuda

The Graphs below were generated by a GPU parallelization of mergesort. Initial analysis seems to indicate that the average computation time was steadily decreasing until reaching 1024 threads at which point there was a drastic dropoff

![image](https://github.com/treywells/CSCE435Project/assets/112406802/1837c59a-5d69-46d4-ba1f-e88ccdcefcdf)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/4d4fd01f-4ec0-4ffa-aed5-7675ed24d528)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/08736190-6bb4-4143-a4a6-bb015f4eab63)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/8645f0de-0f24-44c6-909d-4e45b8aa4d14)



# 5. Presentation Graphs:
   
## Quicksort MPI
  
![image](https://github.com/treywells/CSCE435Project/assets/98286168/3db3088a-4509-4b77-908d-6a41c70993c0)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/d01bf3e1-87df-4acc-acd7-99ddf25b299c)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/468242e8-1df7-47e3-9085-1fe0a82fd1e6)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/2a97fcc9-e460-4f96-94b3-9715443ce221)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/481efdb8-903a-4ed2-a6d6-4f8f0baa7e7a)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/c961fa8d-c4ea-4516-a7de-ed63d8f79d2f)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/4ab2f232-4d41-4345-85d3-ab12048d19a2)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/a366de21-fe99-4e66-aa06-95af702dee2e)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/53e6f7b8-d64b-40c1-ac1f-9a7c421d91b5)










## Quicksort CUDA
![image](https://github.com/treywells/CSCE435Project/assets/98286168/f60fbf6d-1ade-4cfd-8f3c-c9a9709fd1c7)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/108e97e8-c65c-4447-a013-d1b001722fa1)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/8051939b-bba7-4249-8809-02c00ae80e86)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/16905871-035e-444c-ba04-e308d71c4cb4)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/488460cc-9c27-4bb5-8927-a8ff358e6c43)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/e26dd7a3-082b-472a-ba74-bbb091d7819f)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/0bbe554b-c2d7-4449-9043-82650d90141f)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/ca46ff25-30fc-4a22-8c7e-3e35a222000d)
     ![image](https://github.com/treywells/CSCE435Project/assets/98286168/dfcc8e3b-4f33-4c38-8fd2-2bab0c8740aa)

## Odd-Even MPI
### Strong Scaling

![image](https://github.com/treywells/CSCE435Project/assets/95384144/1d7829f2-66d1-4d6e-b9d1-60aa151e3b5f)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/0214b9b7-26aa-4220-ac88-62b364138dbd)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/a2a49869-6c71-4543-b8c7-9eb1994eca76)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/dc55f513-565f-4f73-a11e-cff9fa22cff0)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/449cef0c-d035-4107-8310-f6d271b3cceb)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/55984e7c-11cb-4356-9ad7-20b31390241c)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/29906a41-cc43-4252-a98e-eab2509f9bb8)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/cdb5f336-c37b-4771-b36b-8a0815fed25e)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/aaccc0b0-43f5-48cd-9948-8bc16cf1e676)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/11bbda9c-6530-4ab5-9ca5-950062305e13)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/2e3693bf-8187-4e5c-a216-8bb31b269e69)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/420a76b7-729d-4376-82ab-423a153a933e)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/e6cfa230-1146-47d6-a9e0-099401715d49)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/ab7fa484-fc33-4337-baf4-627e55dee24d)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/af85efd7-2ef0-4056-af24-0287405d27a4)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/1016995b-26b1-41a0-85ab-4c5e218bccf6)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/e67a0ec8-db07-4abb-86df-5428936be445)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/59f2bb6e-b341-43b4-b8a1-f53437e732ec)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/710acbcc-b7ca-4e5e-a0da-f741f6ac5684)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/987ef827-2216-4e56-bb23-9e99710d7b39)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/2ead769e-1cdd-4f43-8cf7-75d3362189c7)

### Weak Scaling
![image](https://github.com/treywells/CSCE435Project/assets/95384144/1fc111a1-df39-4f8b-86ba-34ff40f1f887)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/8ca64f0c-5567-4021-a121-ffe473a808fb)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/6e928f46-55d4-4b58-a9cf-dfcff414bfa8)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/63222335-205e-4af7-9ad4-916c716300e6)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/b2b08036-718a-4058-a290-4d0b2124f4e2)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/bd11d2aa-6e25-4527-8ba2-247d0570e6b4)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/8427e110-d116-49b1-b97c-8f15d080c7c9)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/72744daa-b729-41b7-9d93-0bc0ecbfde28)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/f7db1544-ea48-45bc-96bc-b6382cefdbf6)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/21d5cfee-0494-4d76-9c73-005624048328)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/4249cdf2-9069-4221-b8ad-17d3c5c5e534)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/34253de2-c66c-4b75-ac32-401f3fa0ed79)

### Speedup
![image](https://github.com/treywells/CSCE435Project/assets/95384144/b4806df8-01d9-4363-85c3-16fdc979ad5c)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/de3aef85-6cbd-4d9a-927a-ac27f82d17d5)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/17a8bc77-6fac-4259-becc-609b89fcac39)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/d66710ee-8dbd-409b-80ad-102194ca8d13)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/ea1c6b55-aa35-445c-85ba-8861084ab749)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/76f65ac3-cba4-41b0-9ec3-ecb4007ed6df)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/79d4c06a-a3a3-476e-9eb0-4a9527e7a8c0)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/34e16ab4-bbe4-46b9-9833-17e20933b6c0)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/8cf5985a-937e-4789-bca5-63d6004cfe24)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/d33d21b4-43fd-4670-8e6b-e27777d5d059)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/24734bf8-8bda-4534-bb92-8e65704a137e)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/08d7cc01-1125-4c50-b8b5-70171c91f0c0)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/5fe63f70-46b6-4c7e-8aa0-bbcd76e0108c)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/b3d5562e-148a-4ddb-b9ec-4c2aff1868e7)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/01d1ce6d-ad2d-4d98-897d-7c7f6ea52004)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/f52f02f3-e1ec-4f2e-ba12-2916a5a7f584)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/e1a2b2e2-be34-406c-8b98-6eacc0dbc7b6)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/372192c0-4747-4b3c-af61-140d5428c684)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/f1f2ca8b-d10d-480d-b874-08e39ed3fbb6)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/9492d631-0a9d-41ce-96ca-fcae7cbe4919)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/9421c4f7-7e77-4e60-be97-d169b1b517e5)

## Odd-Even CUDA
### Strong Scaling
![image](https://github.com/treywells/CSCE435Project/assets/95384144/325bead8-fce0-451b-b8e3-2dfe3d5aed5c)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/87f0cd5f-6b16-43bd-ade7-d62ac66172f1)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/0a3fd155-1aa4-4a19-bdb2-42ba305dae50)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/7099c334-561e-4145-bf4f-92d124da14c9)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/ef200f91-ec83-4244-b269-3c77a120f189)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/5fc65c54-15ef-43cc-b797-e59413110079)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/ef4834e5-8b5e-4f00-a06b-60f06f37bba0)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/107868d1-40cc-43e1-a3ca-a79efa3ed1b4)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/984bfd33-35fb-4a8e-9143-215e9f25ebfb)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/c9217dc8-9953-4bca-b679-e718b05f64f0)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/43447dbf-6725-43d8-b0c9-4407e2468f98)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/38436a54-34ee-4d78-a65f-1a9de86f8b43)

### Weak Scaling
![image](https://github.com/treywells/CSCE435Project/assets/95384144/e17430c5-9568-48e9-9fe8-ad4b369ddcc4)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/46941fa0-1665-4f0d-b3d6-d7830c1984d5)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/5f8dfa5e-2fed-4013-bbc3-0633265e1775)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/08be3e0f-83bf-418f-85ac-091d349395b4)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/82ddf9e2-fbd8-4efe-862b-4ed18f0d05ea)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/2f00fb55-25a7-4719-98d8-776aada89dc4)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/e593e0ec-fc09-4c81-af95-15d0296f8b76)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/36eae79a-bc30-4ddf-8703-8fdf3429f761)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/b5df7968-0e81-4630-9542-082179dc983b)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/d70695d2-5562-48bb-bcd5-bf50dfe7ace1)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/3aa4dcdf-8a24-4d68-9c9e-6d2f2f3c421c)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/3621a5c2-d4ba-4f94-8d8a-eeef2bac6f50)

### Speedup
![image](https://github.com/treywells/CSCE435Project/assets/95384144/80129cd0-612a-4375-89d0-9c8da6e82d10)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/e2155310-c84b-4122-8f62-b490286a05ff)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/2609e079-fcf9-431c-87a7-c645f1ca7da3)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/40df4bfa-d6e4-470b-855a-535951357f93)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/6e909280-20f7-4f9c-a7df-96e3bceef219)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/1549d859-0203-42fc-abc5-9d57c3cd67f8)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/bab64923-ed97-4fc0-b4a8-578a11df1193)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/a1a01ebc-1bc4-4700-bc1c-e4f4f4ee1f95)
     ![image](https://github.com/treywells/CSCE435Project/assets/95384144/aa71a0ec-b3f6-4cfd-a5c1-d9d79d62f318)

## MergeSort MPI
### Weak Scaling
**Sorted**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/aa9750c4-7b72-4cda-86b2-f8578ebec29d)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/7d167f4a-bddf-4c34-aa62-937ffb6bb735)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/785d9cb2-79f2-445f-b790-504ed2ef594f)

**Random**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/a2d74e21-5bc4-446a-bebc-0cbb347d25a6)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/084808bb-6b1f-4b1e-b1e7-45e97a027927)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/fa284aa9-0755-4ea4-89f1-2455621a28a6)

**Reverse Sorted**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/8875d532-a4ae-4863-9396-09f6103ec023)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/f601d178-1efd-4f75-83c5-ff9cd820a297)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/4a5e901c-7e24-42b3-a687-bd30fcf6d67d)

**1% Perturbed**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/a23213e6-ea70-48cc-8110-87111288d6e9)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2fc84392-c97d-4961-97c6-48a928328a38)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/0b324866-8bdf-4980-8e30-062cf16e4f82)


### Strong Scaling
**main**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/bf568f26-cab9-4f95-9c7b-eb8a7da290be)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/aff59a34-2aba-459b-830f-dd40f93fa712)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/8e4c1af9-7af2-4485-ad71-f20db944856a)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/f8c177af-f220-44d9-9084-2fb90e4a7614)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/5991182e-c1fd-4605-8f7f-1c8eb53977d1)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/c0b97262-9f3e-4d14-8575-06e4fa4c4892)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/0226b7e1-f695-4da4-8e15-4054a9498c75)

**comp**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/09b5f6c8-5fde-43bb-948e-3a99b99693f8)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/4b7e2ee5-71f1-4a7e-85ba-d2639dee47f0)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/66ac5b37-47d6-4ed6-b9e0-d8264782e577)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/263cbb75-702d-4609-a868-bf68c3d89f46)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/3fbfebdf-352f-46cb-b151-3381a57826d9)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/9fd14624-acd4-40ce-8b69-6995c3edbd55)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2d3bda17-7175-40c8-8184-c20a36fa1d22)

**comm**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/6f41c55b-47c7-41a9-910a-6bdf1b9e944c)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/13b1c6ba-d7ea-4ef3-9af7-d7b38634fc8d)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/f29f2adb-474f-43f6-a1a9-67b7d544dfbf)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/8d8a4026-614f-4a9a-b569-1cb154284dd0)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/4bbc9e14-2884-4456-9053-bb0005cc4c5a)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/67d0cd69-5c23-45da-8909-1224d6e500fe)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/1e8ef0c6-b9ca-4609-802d-9c7fe933935b)

### Speedup
**main**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/a26c0303-7e0b-464f-b2f8-0bb7bdf83491)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/c2eb17d2-57de-485b-af12-dc995a8b8de0)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/d8277d20-58d8-479e-a437-8265a1b89467)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/a5ecf9c4-3f64-4600-9965-f0a8d75bfe20)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2b4017ae-2d14-4936-b110-7e4f160a37d1)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/db8f2c85-2297-474d-b399-2a858ce849cf)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/1e212996-f86c-4afa-83ba-dce652c57f3f)

**comp**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/8d740cbb-0efb-445b-8149-804aafc81f9b)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/e1428a92-e763-4585-8858-8966e0491572)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/19eee48d-5f65-44cf-b2e8-dfe3412dcdf5)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/9b53fbc9-b30b-43ca-bc0b-41b2217dedeb)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/81fc5d7b-71c6-4302-8fcd-8dce6ef160e4)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2000a839-f0d2-4fee-b115-87a6466bc426)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/fad01756-048a-43a3-b7ab-71a681aedee4)


**comm**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/bf330857-f381-4f6a-a6bb-39a89842714c)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/77fc6b7f-2765-4db3-a111-8c272581b6bb)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/d7d49a7a-4636-4519-87af-e92fc5faff67)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/c5fd6895-e268-4fd5-baec-657d26a103cf)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2117f58d-7b4b-40a5-a338-90f9eaa49b17)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/997de17d-53cc-423a-b781-393cfe048f24)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/ea193e58-996f-432e-8cf6-3303140a5846)

## Mergesort CUDA
### Weak Scaling
**main**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/a877218d-bcf5-4537-91cc-67229197c0c4)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/ffdcdf4c-a63d-4af9-bffc-b4b123a0b8f8)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/e7c7ea3c-7fc9-407a-912d-1b76b01b1cf6)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2ae522f9-b850-4f2f-a354-1c543d4054ad)

**comp**
![image](https://github.com/treywells/CSCE435Project/assets/112406802/c4a1c534-5cd6-412e-af6d-6b1223ad46d0)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/0b52fbee-a0a4-4f4e-a29c-a99c624a033a)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/251cf1e8-5aec-4e9e-853b-2f9e4078bd62)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/2f449789-471c-475b-806e-4be6b6d62888)

**comm**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/1b46b0d0-9054-4903-83cd-ded34f71f4db)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/a9bfb2cd-885b-4b47-8465-96dcf64456bf)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/a37dd1e9-1275-424c-99b1-cbc22ed2dc7c)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/30ecea7f-d889-4b92-b74c-6c338f99efab)

### Strong Scaling
**main**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/c8e5346f-435f-4850-a342-c34142cf438e)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/dab1aaa4-6596-4815-9cf7-ebd088bb0ac5)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/3d996321-47b2-44cf-97db-6b314a65e489)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/6ec7b2d0-bb1d-494d-ada7-7690a1efbc96)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/e4ba7c8e-9c43-48bf-adc4-fbf86df83014)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/080a2d5d-3783-4cf0-b5ee-630608633d35)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/0d3e09f3-6a41-4491-8924-2321d0ee2b49)

**comp**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/e176cc8e-ae8a-4b0c-b6af-092b8fef4b51)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/e0403dae-ceae-4302-9990-da3ef4ea39d9)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/095b05e3-9375-4635-9576-dbac24261cc0)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/d7584037-087d-409b-bc57-e67f70cfe89f)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/bb28b86c-c9f0-4356-941f-dc785db835bc)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/1638268e-3b17-4190-9a19-454943cb55d9)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/cbd8c9ea-af67-4b13-ba24-5d65a865899c)

**comm**

![image](https://github.com/treywells/CSCE435Project/assets/112406802/22633dcc-46f5-4f41-9740-e0a8d769cbd7)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/7b7566ca-7348-4edb-b441-f3312c3509cd)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/f287d850-630c-4437-945d-ef68012bfcee)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/6e479491-3294-4c08-9289-f19379019a7f)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/8f0023a7-170b-45c2-b50e-8116192e7365)
![image](https://github.com/treywells/CSCE435Project/assets/112406802/a0591e82-55cf-42ca-aaa7-944e86d52cb0)
![image](https://github.com/treywells/CSCE435Project/blob/master/Builds/MPI/Mergesort/1download.png)
### Speedup
**main**
![image](https://github.com/treywells/CSCE435Project/blob/master/Builds/MPI/Mergesort/download.png)
![image](https://github.com/treywells/CSCE435Project/blob/master/Builds/MPI/Mergesort/download(1).png)

**comp**
**comm**



## Comparison of MPI Algorithms
### Speedup
**Random**

![image](https://github.com/treywells/CSCE435Project/assets/95384144/5c207667-3503-4319-96bc-120d136b3500)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/3f1c7064-2abb-431e-a358-95955066d695)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/83c67cc9-e4bb-4011-9471-0ffd76b12110)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/acf2b071-e901-41a8-aaa2-f62f9271a650)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/7756f2a0-fe50-422b-baac-ad8ad0151f4c)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/fb45217a-8f4f-4c87-87b4-91b1820d41c2)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/7c755c18-9a92-442c-8e0e-b0e54b95b334)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/bc548476-4486-4c3e-b15f-513b4e3167bf)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/08cb40a6-9d58-45b4-98b7-615e9e708429)

## Comparison of CUDA Algorithms
### Speedup
**Random**

![image](https://github.com/treywells/CSCE435Project/assets/95384144/5237e800-df2e-42b5-aff4-130825726ba9)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/d3732a9d-b955-4028-95b2-012a00eed3cf)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/e1af2e13-ec58-4c3c-b5d9-2af0da60b939)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/ca80f30f-1828-411b-98ea-d66642a7675e)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/6a06b88d-0950-4650-b950-e01e79c71f81)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/0857a4c0-87c2-4e2c-a46f-684547823389)

![image](https://github.com/treywells/CSCE435Project/assets/95384144/9ee9c1be-23aa-4c06-bb52-ebd500131370)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/40c170ea-53fe-44be-a517-23496b5de3da)
![image](https://github.com/treywells/CSCE435Project/assets/95384144/2ec61dd5-fd53-42b8-b402-bff8630f09c3)







     



     




















     














































































  




























































