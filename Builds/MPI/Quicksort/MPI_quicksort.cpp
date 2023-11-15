#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

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
        quicksort(arr, 0, arrSize - 1);
        return 0;
    }
    // Divide the array into two parts with the pivot in between
    int j = 0;
    int pivotIndex;

    
    
    // CALI_MARK_BEGIN("comp_small");
    
    pivotIndex = hoare_partition(arr, j, arrSize - 1);
    
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
        arr[j] = (int) rand() % 10000;
    }
}


// Function to initialize the data (generate random, reverse sorted, sorted, or perturbed input array)
void parallel_data_init(int *arr, int arr_size, string inputType, int rank, int size) {
    int j, chunk_size;

    // Calculate the chunk size for each process
    chunk_size = arr_size / size;

    if (inputType == "Random") {
        for (j = rank * chunk_size; j < (rank + 1) * chunk_size; ++j) {
            arr[j] = rand() % 1000;
        }
    } else if (inputType == "ReverseSorted") {
        for (j = rank * chunk_size; j < (rank + 1) * chunk_size; ++j) {
            arr[j] = arr_size - j;
        }
    } else if (inputType == "Sorted") {
        for (j = rank * chunk_size; j < (rank + 1) * chunk_size; ++j) {
            arr[j] = j;
        }
    } else if (inputType == "Perturbed") {
        for (j = rank * chunk_size; j < (rank + 1) * chunk_size; ++j) {
            arr[j] = (j % 100 == 0) ? (rand() % 1000) : j;
        }
    } else {
        // Handle unsupported input type or default to random
        for (j = rank * chunk_size; j < (rank + 1) * chunk_size; ++j) {
            arr[j] = rand() % 1000;
        }
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
    
    if (argc != 3) {
        printf("Usage: %s <array_size> <input_type>\n", argv[0]);
        return 1;
    }

    int array_size = atoi(argv[1]);
    string inputType = argv[2];

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
    // if (rank == 0) {
    //     // --- RANDOM ARRAY GENERATION ---
    //     printf("Creating a Random List of %d elements\n", array_size);
    //     CALI_MARK_BEGIN("data_init");
    //     data_init(unsorted_array, array_size);
    //     CALI_MARK_END("data_init");
    //     printf("Created\n");
    // }
     // --- PARALLEL RANDOM ARRAY GENERATION ---
    CALI_MARK_BEGIN("data_init");
    parallel_data_init(unsorted_array, array_size, inputType, rank, size);
    CALI_MARK_END("data_init");

    //  // Print the array before sorting
    // if (rank == 0) {
    //     printf("Array Before Sorting:\n");
    //     for (int i = 0; i < array_size; ++i) {
    //         printf("%d ", unsorted_array[i]);
    //     }
    //     printf("\n");
    // }

    // // Use separate arrays for sending and receiving in MPI_Gather
    // int* send_buffer = unsorted_array + rank * (array_size / size);

    // CALI_MARK_BEGIN("comm");
    // CALI_MARK_BEGIN("MPI_Gather");
    // MPI_Gather(send_buffer, array_size / size, MPI_INT,
    //         unsorted_array, array_size / size, MPI_INT, 0, MPI_COMM_WORLD);
    // CALI_MARK_END("MPI_Gather");
    // CALI_MARK_END("comm");

    // Allocate a temporary array for gathering data at the root
    int* gathered_array = NULL;
    if (rank == 0) {
        gathered_array = (int*)malloc(array_size * sizeof(int));
    }
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("MPI_Gather");
    // Gather the generated data from all processes
    MPI_Gather(unsorted_array + rank * (array_size / size), array_size / size, MPI_INT,
            gathered_array, array_size / size, MPI_INT, 0, MPI_COMM_WORLD);

    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm");
    // If root, copy the gathered data back to the original array
    if (rank == 0) {
        memcpy(unsorted_array, gathered_array, array_size * sizeof(int));
        free(gathered_array); // Free the temporary array
    }



    // // Print the array after gathering but before sorting
    // if (rank == 0) {
    //     printf("Array After Gathering (Before Sorting):\n");
    //     for (int i = 0; i < array_size; ++i) {
    //         printf("%d ", unsorted_array[i]);
    //     }
    //     printf("\n");
    // }
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
        
        quicksort_recursive(unsorted_array, array_size, rank, size - 1, rankPower);
        
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");
    } else {
        // All other Clusters wait for their subarray to arrive,
        // they sort it and they send it back.
        MPI_Status status;
        int subarray_size;
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("MPI_Probe");
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        CALI_MARK_END("MPI_Probe");

        // Capturing the size of the array to receive
        CALI_MARK_BEGIN("comm_small");
        CALI_MARK_BEGIN("MPI_Get_Count");
        MPI_Get_count(&status, MPI_INT, &subarray_size);
        CALI_MARK_END("MPI_Get_Count");
        CALI_MARK_END("comm_small");

        int source_process = status.MPI_SOURCE;
        int subarray[subarray_size];

        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Recv");
        MPI_Recv(subarray, subarray_size, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END("MPI_Recv");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");

        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        
        quicksort_recursive(subarray, subarray_size, rank, size - 1, rankPower);
        
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(subarray, subarray_size, MPI_INT, source_process, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
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

        // Print the array after sorting
        // printf("Array After Sorting:\n");
        // for (int i = 0; i < array_size; ++i) {
        //     printf("%d ", unsorted_array[i]);
        // }
        // printf("\n");
    }
    string algorithm = "QuickSort";
    string programmingModel = "MPI";
    string datatype = "int";
    int sizeOfDatatype = sizeof(int);
    int inputSize = array_size;
    // string inputType = "Random";
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
