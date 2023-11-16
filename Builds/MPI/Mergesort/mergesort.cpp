#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *main_function = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *barrier = "MPI_Barrier";
const char *comm_large = "comm_large";
const char *gather = "MPI_Gather";
const char *scatter = "MPI_Scatter";
const char *comp = "comp";
const char *comp_large = "comp_large";

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r)
{
    int h, i, j, k;
    h = l;
    i = l;
    j = m + 1;

    while ((h <= m) && (j <= r))
    {
        if (a[h] <= a[j])
        {
            b[i] = a[h];
            h++;
        }
        else
        {
            b[i] = a[j];
            j++;
        }

        i++;
    }

    if (m < h)
    {
        for (k = j; k <= r; k++)
        {
            b[i] = a[k];
            i++;
        }
    }
    else
    {
        for (k = h; k <= m; k++)
        {
            b[i] = a[k];
            i++;
        }
    }

    for (k = l; k <= r; k++)
    {
        a[k] = b[k];
    }
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r)
{
    int m;

    if (l < r)
    {
        m = (l + r) / 2;

        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
    }
}

/********** Parallel Data Initialization **********/
void parallelDataInit(int *array, int size, int world_rank, int world_size, int type)
{
    int chunk_size = size / world_size;
    int *temp_array = (int *)malloc(chunk_size * sizeof(int));

    // Generate random numbers in parallel
    srand(time(NULL) + world_rank);
    for (int c = 0; c < chunk_size; c++)
    {
        switch (type)
        {
        case 0: // Sorted list
            temp_array[c] = world_rank * chunk_size + c + 1;
            break;
        case 1: // Reverse sorted list
            temp_array[c] = world_rank * chunk_size + chunk_size - c;
            break;
        case 2:                                // 1% perturbed list
            if (rand() % 100 == 0)             // 1% chance
                temp_array[c] = rand() % size; // Random value
            else
                temp_array[c] = world_rank * chunk_size + c + 1;
            break;
        default:
            temp_array[c] = rand() % size; // Default to random list
            break;
        }
    }

    // Gather all chunks to the root
    MPI_Gather(temp_array, chunk_size, MPI_INT, array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Free temporary array
    free(temp_array);
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN(main_function);

    /********** Initialize MPI **********/
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /********** Create and populate the array **********/
    int n = atoi(argv[1]);
    int *original_array = (int *)malloc(n * sizeof(int));

    CALI_MARK_BEGIN(data_init);
    parallelDataInit(original_array, n, world_rank, world_size, 3);
    CALI_MARK_END(data_init);

    /********** Divide the array into equal-sized chunks **********/
    int size = n / world_size;

    /********** Send each subarray to each process **********/
    int *sub_array = (int *)malloc(size * sizeof(int));
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(scatter);
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    /********** Perform the mergesort on each process **********/
    int *tmp_array = (int *)malloc(size * sizeof(int));
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeSort(sub_array, tmp_array, 0, (size - 1));
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    /********** Gather the sorted subarrays into one **********/
    int *sorted = NULL;
    if (world_rank == 0)
    {
        sorted = (int *)malloc(n * sizeof(int));
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(gather);
    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    /********** Make the final mergeSort call **********/
    if (world_rank == 0)
    {
        int *other_array = (int *)malloc(n * sizeof(int));
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        mergeSort(sorted, other_array, 0, (n - 1));
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        /********** Check sorted array **********/
        CALI_MARK_BEGIN(correctness_check);
        for (int c = 0; c < n - 1; c++)
        {
            if (sorted[c] > sorted[c + 1])
            {
                printf("Sort Failed\n");
                break;
            }
        }
        CALI_MARK_END(correctness_check);
        printf("Sort Successful\n");

        /********** Clean up root **********/
        free(sorted);
        free(other_array);
    }
    CALI_MARK_END(main_function);

    /********** Clean up rest **********/
    free(original_array);
    free(sub_array);
    free(tmp_array);

    /********** Finalize MPI **********/
    if (world_rank == 0)
    {
        const char *algorithm = "Mergesort";
        const char *programmingModel = "MPI";
        const char *datatype = "int";
        int sizeOfDatatype = sizeof(int);
        int inputSize = n;
        const char *inputType = "Random";
        int num_procs = world_size;
        const char *num_threads = "N/A";
        const char *num_blocks = "N/A";
        int group_number = 18;
        const char *implementation_source = "Online and AI";

        adiak::init(NULL);
        adiak::launchdate();                                          // launch date of the job
        adiak::libraries();                                           // Libraries used
        adiak::cmdline();                                             // Command line used to launch the job
        adiak::clustername();                                         // Name of the cluster
        adiak::value("Algorithm", algorithm);                         // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", programmingModel);           // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", datatype);                           // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeOfDatatype);               // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", inputSize);                         // The number of elements in input dataset (1000)
        adiak::value("InputType", inputType);                         // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", num_procs);                         // The number of processors (MPI ranks)
        adiak::value("num_threads", num_threads);                     // The number of CUDA or OpenMP threads
        adiak::value("num_blocks", num_blocks);                       // The number of CUDA blocks
        adiak::value("group_num", group_number);                      // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
