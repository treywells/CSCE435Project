#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cstdlib>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char* main_function = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";
const char* MPI_send = "MPI_Send";
const char* MPI_recv = "MPI_Recv";
const char* sortedInput = "sorted";
const char* randomInput = "random";
const char* reverseSortedInput = "reverse_sorted";
const char* perturbed = "perturbed";


/*
Source: https://github.com/erenalbayrak/Odd-Even-Sort-mit-MPI/blob/master/implementation/c%2B%2B/OddEvenSort.cpp
AI (Chat GPT) was used to create functions for data generation and correctness checking
*/

void initialize_reverse_sorted(int *data,
                                 long rank,
                                 int procs,
								 int localSize)
{

    for (int i = 0; i < localSize; i++) {
		data[i] = localSize * procs - (i + localSize * rank);    // reverse order
	}
}

void initialize_sorted(int *data,
                                 long rank,
                                 int procs,
								 int localSize)
{

    for (int i = 0; i < localSize; i++) {
		data[i] = i + localSize * rank;    // in sorted order
	}
}

void initialize_random(int *data,
                                 long rank,
                                 int procs,
								 int localSize)
{
    srand(time(NULL));
    for (int i = 0; i < localSize; i++) {
		data[i] = rand() % 2147483648;    // in random order
	}
}

void initialize_perturbed(int *data,
                                 long rank,
                                 int procs,
								 int localSize)
{

    for (int i = 0; i < localSize; i++) {
		data[i] = i + localSize * rank;    // in sorted order
	}

    int perturbationCount = localSize / 100; // 1% of the local array size

    for (int i = 0; i < perturbationCount; i++) {
        int indexToPerturb = rand() % localSize; // Choose a random index to perturb
        data[indexToPerturb] = rand() % 10000; // Modify this as needed for your specific range
    }
}

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

int compare (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
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
            CALI_MARK_BEGIN(MPI_send);
            MPI_Send(data, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
            CALI_MARK_END(MPI_send);

            CALI_MARK_BEGIN(MPI_recv);
            MPI_Recv(other, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END(MPI_recv);
        } else {
            CALI_MARK_BEGIN(MPI_recv);
            MPI_Recv(other, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END(MPI_recv);

            CALI_MARK_BEGIN(MPI_send);
            MPI_Send(data, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
            CALI_MARK_END(MPI_send);
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
	
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

}

int isSorted(int* nums, int vals) {

	for (int i = 0; i < vals - 1; i++) {
		if (nums[i] > nums[i + 1]) return 0;
	}
	return 1;
}

int main(int argc, char** argv)
{
    int rank, count_processes;
    
    cali::ConfigManager mgr;
    mgr.start();
	
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &count_processes);

    CALI_MARK_BEGIN(main_function);


    int vals = atoi(argv[1]);
    char* input = argv[2];
    const char* input_type;
	int localSize = vals/count_processes;
    int *data = new int[localSize];
    int* gathered_data = nullptr;

	CALI_MARK_BEGIN(data_init);
    if (strcmp(input, sortedInput) == 0) {
        initialize_sorted(data, rank, count_processes, localSize);
        input_type = "sorted";
    }
    else if (strcmp(input, randomInput) == 0) {
        initialize_random(data, rank, count_processes, localSize);
        input_type = "random";
    }
    else if (strcmp(input, reverseSortedInput) == 0) {
        initialize_reverse_sorted(data, rank, count_processes, localSize);
        input_type = "reverse_sorted";
    }
    else if (strcmp(input, perturbed) == 0) {
        initialize_perturbed(data, rank, count_processes, localSize);
        input_type = "perturbed";
    }
	CALI_MARK_END(data_init);
	
	MPI_Barrier(MPI_COMM_WORLD);

    parallel_sort(data, rank, count_processes, localSize);

	// Create a buffer to gather data on rank 0
    //int *gathered_data = nullptr;
    if (rank == 0) {
        gathered_data = new int[vals];
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN("MPI_Gather");

    // Gather data from all processes onto rank 0
    MPI_Gather(data, localSize, MPI_INT, gathered_data, localSize, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    CALI_MARK_END("MPI_Gather");
	CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
	
    // Print the sorted data on rank 0
    if (rank == 0) {
		CALI_MARK_BEGIN(correctness_check);
        // gathered_data now contains all data from all processes
		if (isSorted(gathered_data, vals)) {
			printf("Data is definitely now sorted\n");
		} else {
			printf("Data is definitely not sorted :(\n");
		}

    	CALI_MARK_END(correctness_check);
        // Clean up gathered_data
        delete[] gathered_data;
	    delete[] data;
    }




    CALI_MARK_END(main_function);
	if (rank == 0) {
		const char* algorithm = "BubbleSort";
		const char* programmingModel = "MPI";
		const char* datatype = "int";
		int sizeOfDatatype = sizeof(int);
		int inputSize = vals;
		int num_procs = count_processes;
		const char* num_threads = "N/A";
		const char* num_blocks = "N/A";
		int group_number = 18;
		const char* implementation_source = "Online and AI";

		adiak::init(NULL);
		adiak::launchdate();    // launch date of the job
		adiak::libraries();     // Libraries used
		adiak::cmdline();       // Command line used to launch the job
		adiak::clustername();   // Name of the cluster
		adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
		adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
		adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
		adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
		adiak::value("input_size", inputSize); // The number of elements in input dataset (1000)
		adiak::value("input_type", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
		adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
		adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
		adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
		adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
		adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
		
		mgr.stop();
        mgr.flush();
	}

    MPI_Finalize();
    return EXIT_SUCCESS;
}