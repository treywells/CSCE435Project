#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <cstring>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char* main_function = "main";

/*
Source: https://homepages.math.uic.edu/~jan/mcs572/pipelinedsort.pdf
AI (Chat GPT) was used to create functions for data generation and correctness checking
*/

void Compare_and_Send( int myid, int step, int *smaller, int *gotten ){
/* Processor "myid" initializes smaller with gotten
* at step zero, or compares smaller to gotten and
* sends the larger number through. */
	if(step==0){
		*smaller = *gotten;
	}
	else{
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
	}
	else{
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

void Collect_Sorted_Sequence( int myid, int p, int smaller, int *sorted ) {
	/* Processor "myid" sends its smaller number to the
	* manager who collects the sorted numbers in the
	* sorted array, which is then printed. */
	MPI_Status status;
	int k;
	if(myid==0){
		sorted[0] = smaller;
		for(k=1; k<p; k++){
		    MPI_Recv(&sorted[k],1,MPI_INT,k,tag,MPI_COMM_WORLD,&status);
		}
		printf("The sorted sequence : ");
		for(k=0; k<p; k++) printf(" %d",sorted[k]);
		printf("\n");
	}
	else{
		MPI_Send(&smaller,1,MPI_INT,0,tag,MPI_COMM_WORLD);
	}
}

int main(int argc, char** argv)
{
    int i,p,*n,j,g,s;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	MPI_Comm_rank(MPI_COMM_WORLD,&i);
	
	CALI_MARK_BEGIN(main_function);

	if(i==0) /* manager generates p random numbers */
	{
		n = (int*)calloc(p,sizeof(int));
		srand(time(NULL));
		for(j=0; j<p; j++) n[j] = rand() % 100;
		if(verbose>0){
			printf("The %d numbers to sort : ",p);
			for(j=0; j<p; j++) printf(" %d", n[j]);
			printf("\n"); fflush(stdout);
		}
	}
	for(j=0; j<p-i; j++){ /* processor i performs p-i steps */
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
	}
	MPI_Barrier(MPI_COMM_WORLD); /* to synchronize for printing */
	Collect_Sorted_Sequence(i,p,s,n);
	
    CALI_MARK_END(main_function);
    
	delete[] data;


	if (rank == 0) {

		const char* algorithm = "BubbleSort";
		const char* programmingModel = "MPI";
		const char* datatype = "int";
		int sizeOfDatatype = sizeof(int);
		int inputSize = p;
		const char* inputType = "ReverseSorted";
		int num_procs = p-i;
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
		adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
		adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
		adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
		adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
		adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
		adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
		adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
	}

    MPI_Finalize();
    return EXIT_SUCCESS;
}
