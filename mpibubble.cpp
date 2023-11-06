#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <cstring>

using namespace std;

/**
 * @param binary_file: the binary file on Disk.
 * @param vector: the values from binary file will store in the vector.
 * @return EXIT_SUCCESS or EXIT_FAILURE
 * */
int fill_vector_with_numbers(int *data,
                                 long rank,
                                 int procs,
								 int localSize)
{

    for (int i = 0; i < localSize; i++) {
		data[i] = localSize * procs - (i + localSize * rank);
	}

    return EXIT_SUCCESS;
}

void print(int *data, int rank, unsigned long data_size)
{
    cout << "rank " << rank << " : ";
    for(unsigned long int i=0; i<data_size; i++)
        cout << data[i] << " ";
    cout << endl;
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
    const unsigned long concat_data_size = data_size * 2;

    auto *other      = new int[data_size];
    auto *concatData = new int[concat_data_size];

    for (int i=0; i<count_processes; i++)
    {
        int partner = findPartner(i, rank);
        if (partner < 0 || partner >= count_processes)
          continue;

        if (rank % 2 == 0) {
            MPI_Send(data, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Recv(other, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(other, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(data, (int) data_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
        }

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
}

int isSorted(int* nums, int vals) {

	for (int i = 0; i < vals - 1; i++) {
		if (nums[i] > nums[i + 1]) return 0;
	}
	return 1;
}

/**
 * Compile:      mpic++ OddEvenSort.cpp -o OddEvenSort -std=gnu++0x
 * Example-Call: mpirun -np 4 ./OddEvenSort "<numbers_file.bin>" <y>
 * <y> output on console
 * */
int main(int argc, char** argv)
{
    int rank, count_processes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &count_processes);

    int vals = atoi(argv[1]);
	int localSize = vals/count_processes;
    int *data = new int[localSize];
    int status = fill_vector_with_numbers(data, rank, count_processes, localSize);

    parallel_sort(data, rank, count_processes, localSize);

	// Create a buffer to gather data on rank 0
    int *gathered_data = nullptr;
    if (rank == 0) {
        gathered_data = new int[vals];
    }

    // Gather data from all processes onto rank 0
    MPI_Gather(data, localSize, MPI_INT, gathered_data, localSize, MPI_INT, 0, MPI_COMM_WORLD);


    // Print the sorted data on rank 0
    if (rank == 0) {
        // gathered_data now contains all data from all processes
		if (isSorted(gathered_data, vals)) {
			printf("Data is definitely now sorted\n");
		} else {
			printf("Data is definitely not sorted :(\n");
		}

        // Clean up gathered_data
        delete[] gathered_data;
    }

	delete[] data;

    MPI_Finalize();
    return EXIT_SUCCESS;
}