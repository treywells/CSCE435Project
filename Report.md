# CSCE 435 Group project

## 0. Group number: 

## 1. Group members:
1. Trey Wells
2. Aaron Weast
3. Jacob Miller
4. David Vilenchouk

## 2. Project topic (e.g., parallel sorting algorithms)

 Sorting Algorithm Performance Comparisons for OpenMP and CUDA

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For the duration of this project, our team plans on communicating via Slack. 

For our algorithms, we plan on implementing various sorting algorithms. The four sorting algorithms we are planning on implementing are Bubble sort, Merge Sort, quick sort, and insertion sort. 

For each of the algorithms, we are planning on implementating in both OpenMP and CUDA so that we can compare the differences in CPU vs. GPU parallelization. Not only will we be comparing the differences in CPU and GPU speed but we will also be testing the differences in the algorithms on various types of inputs. For example, we might run each algorithm on a completely random input, then on a partially sorted one, then on a completely sorted one. 

## Psuedocode for Algorithms

For example:

- Bubble Sort (MPI)
	```
	evenOddSort(array, n):
		for phase from 0 --> n:
	        if phase is even:
	            # pragma omp parallelize
	            for even values of i from 1 --> n:
	                if array[i] > array[i + 1]:
	                    swap()
	        else:
	            #pragma omp parallelize
	            for odd values of i from 1 --> n:
	                if array[i] > array[i + 1]:
	                    swap()

	```
    Source: https://people.cs.pitt.edu/~bmills/docs/teaching/cs1645/lecture_par_sort.pdf

- Bubble Sort (CUDA)
	```
    __global__ evenOddSort(array, length):
        while unsorted:
            if threadID is even:
                if array[ID] > array[ID + 1]:
                    swap()
            
            __syncthreads()

            if threadID is odd:
                if array[ID] > array[ID + 1]:
                    swap()

            __syncthreads()

    main():
        evenOddSort<<< blocks, threads >>>(array, length)
 	```

    Source: https://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/SLIDES/s19.html

- Quick Sort (MPI)
	```  
	// C program to implement the Quick Sort
	// Algorithm using MPI
	#include <mpi.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <time.h>
	#include <unistd.h>
	using namespace std;
 
	// Function to swap two numbers 
	void swap(int* arr, int i, int j)
	{
	    int t = arr[i];
	    arr[i] = arr[j];
	    arr[j] = t;
	}
 
	// Function that performs the Quick Sort
	// for an array arr[] starting from the
	// index start and ending at index end
	void quicksort(int* arr, int start, int end)
	{
    int pivot, index;
 
    // Base Case
    if (end <= 1)
        return;
 
    // Pick pivot and swap with first
    // element Pivot is middle element
    pivot = arr[start + end / 2];
    swap(arr, start, start + end / 2);
 
    // Partitioning Steps
    index = start;
 
    // Iterate over the range [start, end]
    for (int i = start + 1; i < start + end; i++) {
 
        // Swap if the element is less
        // than the pivot element
        if (arr[i] < pivot) {
            index++;
            swap(arr, i, index);
        }
    }
 
    // Swap the pivot into place
    swap(arr, start, index);
 
    // Recursive Call for sorting
    // of quick sort function
    quicksort(arr, start, index - start);
    quicksort(arr, index + 1, start + end - index - 1);
	}
 
	// Function that merges the two arrays
	int* merge(int* arr1, int n1, int* arr2, int n2)
	{
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0;
    int j = 0;
    int k;
 
    for (k = 0; k < n1 + n2; k++) {
        if (i >= n1) {
            result[k] = arr2[j];
            j++;
        }
        else if (j >= n2) {
            result[k] = arr1[i];
            i++;
        }
 
        // Indices in bounds as i < n1
        // && j < n2
        else if (arr1[i] < arr2[j]) {
            result[k] = arr1[i];
            i++;
        }
 
        // v2[j] <= v1[i]
        else {
            result[k] = arr2[j];
            j++;
        }
    }
    return result;
	}
 
	// Driver Code //
	int main(int argc, char* argv[])
	{
    int number_of_elements;
    int* data = NULL;
    int chunk_size, own_chunk_size;
    int* chunk;
    FILE* file = NULL;
    double time_taken;
    MPI_Status status;
 
    if (argc != 3) {
        printf("Desired number of arguments are not their "
               "in argv....\n");
        printf("2 files required first one input and "
               "second one output....\n");
        exit(-1);
    }
 
    int number_of_process, rank_of_process;
    int rc = MPI_Init(&argc, &argv);
 
    if (rc != MPI_SUCCESS) {
        printf("Error in creating MPI "
               "program.\n "
               "Terminating......\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
 
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);
 
    if (rank_of_process == 0) {
        // Opening the file
        file = fopen(argv[1], "r");
 
        // Printing Error message if any
        if (file == NULL) {
            printf("Error in opening file\n");
            exit(-1);
        }
 
        // Reading number of Elements in file ...
        // First Value in file is number of Elements
        printf(
            "Reading number of Elements From file ....\n");
        fscanf(file, "%d", &number_of_elements);
        printf("Number of Elements in the file is %d \n",
               number_of_elements);
 
        // Computing chunk size
        chunk_size
            = (number_of_elements % number_of_process == 0)
                  ? (number_of_elements / number_of_process)
                  : (number_of_elements / number_of_process
                     - 1);
 
        data = (int*)malloc(number_of_process * chunk_size
                            * sizeof(int));
 
        // Reading the rest elements in which
        // operation is being performed
        printf("Reading the array from the file.......\n");
        for (int i = 0; i < number_of_elements; i++) {
            fscanf(file, "%d", &data[i]);
        }
 
        // Padding data with zero
        for (int i = number_of_elements;
             i < number_of_process * chunk_size; i++) {
            data[i] = 0;
        }
 
        // Printing the array read from file
        printf("Elements in the array is : \n");
        for (int i = 0; i < number_of_elements; i++) {
            printf("%d ", data[i]);
        }
 
        printf("\n");
 
        fclose(file);
        file = NULL;
    }
 
    // Blocks all process until reach this point
    MPI_Barrier(MPI_COMM_WORLD);
 
    // Starts Timer
    time_taken -= MPI_Wtime();
 
    // BroadCast the Size to all the
    // process from root process
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0,
              MPI_COMM_WORLD);
 
    // Computing chunk size
    chunk_size
        = (number_of_elements % number_of_process == 0)
              ? (number_of_elements / number_of_process)
              : number_of_elements
                    / (number_of_process - 1);
 
    // Calculating total size of chunk
    // according to bits
    chunk = (int*)malloc(chunk_size * sizeof(int));
 
    // Scatter the chuck size data to all process
    MPI_Scatter(data, chunk_size, MPI_INT, chunk,
                chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(data);
    data = NULL;
 
    // Compute size of own chunk and
    // then sort them
    // using quick sort
 
    own_chunk_size = (number_of_elements
                      >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (number_of_elements
                            - chunk_size * rank_of_process);
 
    // Sorting array with quick sort for every
    // chunk as called by process
    quicksort(chunk, 0, own_chunk_size);
 
    for (int step = 1; step < number_of_process;
         step = 2 * step) {
        if (rank_of_process % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT,
                     rank_of_process - step, 0,
                     MPI_COMM_WORLD);
            break;
        }
 
        if (rank_of_process + step < number_of_process) {
            int received_chunk_size
                = (number_of_elements
                   >= chunk_size
                          * (rank_of_process + 2 * step))
                      ? (chunk_size * step)
                      : (number_of_elements
                         - chunk_size
                               * (rank_of_process + step));
            int* chunk_received;
            chunk_received = (int*)malloc(
                received_chunk_size * sizeof(int));
            MPI_Recv(chunk_received, received_chunk_size,
                     MPI_INT, rank_of_process + step, 0,
                     MPI_COMM_WORLD, &status);
 
            data = merge(chunk, own_chunk_size,
                         chunk_received,
                         received_chunk_size);
 
            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size
                = own_chunk_size + received_chunk_size;
        }
    }
 
    // Stop the timer
    time_taken += MPI_Wtime();
 
    // Opening the other file as taken form input
    // and writing it to the file and giving it
    // as the output
    if (rank_of_process == 0) {
        // Opening the file
        file = fopen(argv[2], "w");
 
        if (file == NULL) {
            printf("Error in opening file... \n");
            exit(-1);
        }
 
        // Printing total number of elements
        // in the file
        fprintf(
            file,
            "Total number of Elements in the array : %d\n",
            own_chunk_size);
 
        // Printing the value of array in the file
        for (int i = 0; i < own_chunk_size; i++) {
            fprintf(file, "%d ", chunk[i]);
        }
 
        // Closing the file
        fclose(file);
 
        printf("\n\n\n\nResult printed in output.txt file "
               "and shown below: \n");
 
        // For Printing in the terminal
        printf("Total number of Elements given as input : "
               "%d\n",
               number_of_elements);
        printf("Sorted array is: \n");
 
        for (int i = 0; i < number_of_elements; i++) {
            printf("%d ", chunk[i]);
        }
 
        printf(
            "\n\nQuicksort %d ints on %d procs: %f secs\n",
            number_of_elements, number_of_process,
            time_taken);
    }
 
    MPI_Finalize();
    return 0;
	}
 	```
 
source: https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/

- Quick Sort (CUDA)
  
	```   
	#include <time.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <cutil_inline.h>
	#define MAX_THREADS	128 
	#define N		512

	int*	r_values;
	int*	d_values;

	// initialize data set
	void Init(int* values, int i) {
		srand( time(NULL) );
        printf("\n------------------------------\n");
        

        if (i == 0) {
        // Uniform distribution
			printf("Data set distribution: Uniform\n");
			for (int x = 0; x < N; ++x) {
				values[x] = rand() % 100;
				//printf("%d ", values[x]);
			}
		}
        else if (i == 1) {
        	// Gaussian distribution
 			#define MEAN    100
        	#define STD_DEV 5
			printf("Data set distribution: Gaussian\n");
			float r;
			for (int x = 0; x < N; ++x) {
				r  = (rand()%3 - 1) + (rand()%3 - 1) + (rand()%3 - 1);
				values[x] = int( round(r * STD_DEV + MEAN) );
				//printf("%d ", values[x]);
			}
		}
        else if (i == 2) {
        // Bucket distribution
			printf("Data set distribution: Bucket\n");
                int j = 0;
                for (int x = 0; x < N; ++x, ++j) {
					if (j / 20 < 1)
						values[x] = rand() % 20;
					else if (j / 20 < 2)
						values[x] = rand() % 20 + 20;
					else if (j / 20 < 3)
						values[x] = rand() % 20 + 40;
					else if (j / 20 < 4)
						values[x] = rand() % 20 + 60;
					else if (j / 20 < 5)
						values[x] = rand() % 20 + 80;
					if (j == 100)
						j = 0;
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 3) {
        // Sorted distribution
			printf("Data set distribution: Sorted\n");
			/*for (int x = 0; x < N; ++x)
				printf("%d ", values[x]);
			*/
        }
		else if (i == 4) {
        // Zero distribution
			printf("Data set distribution: Zero\n");
			int r = rand() % 100;
			for (int x = 0; x < N; ++x) {
				values[x] = r;
				//printf("%d ", values[x]);
			}
        }
        printf("\n");
    }

     // Kernel function
     __global__ static void quicksort(int* values) {
     #define MAX_LEVELS	300

	int pivot, L, R;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if(L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
				// swap start[idx] and start[idx-1]
				int tmp = start[idx];
				start[idx] = start[idx - 1];
				start[idx - 1] = tmp;

				// swap end[idx] and end[idx-1]
				tmp = end[idx];
				end[idx] = end[idx - 1];
				end[idx - 1] = tmp;
			}
		}
		else
			idx--;
	}
    }
 
     // program main
	int main(int argc, char **argv) {
		printf("./quicksort starting with %d numbers...\n", N);
 		unsigned int hTimer;
 		size_t size = N * sizeof(int);
 	
 		// allocate host memory
 		r_values = (int*)malloc(size);
 	
		// allocate device memory
		cutilSafeCall( cudaMalloc((void**)&d_values, size) );

		// allocate threads per block
        const unsigned int cThreadsPerBlock = 128;
                
		/* Types of data sets to be sorted:
         *      1. Normal distribution
         *      2. Gaussian distribution
         *      3. Bucket distribution
         *      4. Sorted Distribution
         *      5. Zero Distribution
         */

		for (int i = 0; i < 5; ++i) {
			// initialize data set
			Init(r_values, i);

	 		// copy data to device	
			cutilSafeCall( cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice) );

			printf("Beginning kernel execution...\n");

			cutilCheckError( cutCreateTimer(&hTimer) );
 			cutilSafeCall( cudaThreadSynchronize() );
			cutilCheckError( cutResetTimer(hTimer) );
	 		cutilCheckError( cutStartTimer(hTimer) );
	
			// execute kernel
 			quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values);
	 		cutilCheckMsg( "Kernel execution failed..." );

 			cutilSafeCall( cudaThreadSynchronize() );
	 		cutilCheckError( cutStopTimer(hTimer) );
	 		double gpuTime = cutGetTimerValue(hTimer);

 			printf( "\nKernel execution completed in %f ms\n", gpuTime );
 	
		 	// copy data back to host
			cutilSafeCall( cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost) );
 	
		 	// test print
	 		/*for (int i = 0; i < N; i++) {
	 			printf("%d ", r_values[i]);
	 		}
	 		printf("\n");
			*/

			// test
			printf("\nTesting results...\n");
			for (int x = 0; x < N - 1; x++) {
				if (r_values[x] > r_values[x + 1]) {
					printf("Sorting failed.\n");
					break;
				}
				else
					if (x == N - 2)
						printf("SORTING SUCCESSFUL\n");
			}

		}
 	
 		// free memory
		cutilSafeCall( cudaFree(d_values) );
 		free(r_values);
 	
 		cutilExit(argc, argv);
 		cudaThreadExit();
    }
	```
 
source: https://github.com/saigowri/CUDA/blob/master/quicksort.cu

- Merge Sort (MPI)

	```
	void merge(X, n, tmp):
        i = 0
        j = n / 2
        ti = 0
        while i < n / 2 and j < n:
            if X[i] < X[j]:
                tmp[ti] = X[i]
                ti = ti + 1
                i = i + 1
             else:
                tmp[ti] = X[j]
                ti = ti + 1
                j = j + 1
      
        while i < n / 2:
            tmp[ti] = X[i]
            ti = ti + 1
            i = i + 1
      
        while j < n:
            tmp[ti] = X[j]
            ti = ti + 1
            j = j + 1
        copy(tmp, X, n)
	end merge
  

    void mergesort(X, n, tmp):
        if n < 2:
            return
        //Sort the first half
        #pragma omp task (X, n, tmp)
        mergesort(X, n / 2, tmp)
        //Sort the second half
        #pragma omp task (X, n, tmp)
		mergesort(X + (n / 2), n - (n / 2), tmp)
        //wait for both tasks to complete
        #pragma omp taskwait
        //Merge the sorted halves
        merge(X, n, tmp)
    end mergesort

	```

    Source: https://avcourt.github.io/tiny-cluster/2019/03/08/merge_sort.html
  

- Merge Sort (CUDA)
  
        void Device_Merge(d_list, length, elementsPerThread):
            index = blockIdx.x * blockDim.x + threadIdx.x
            for i in 0 to elementsPerThread - 1:
                if (index + i < length):
                    tempList[current_list][elementsPerThread * threadIdx.x + i] = d_list[index + i]
            synchronize_threads()
            for walkLen = 1 to length - 1 by walkLen *= 2:
                my_start = elementsPerThread * threadIdx.x
                my_end = my_start + elementsPerThread
                left_start = my_start
                while left_start < my_end:
                    old_left_start = left_start
                    if left_start > my_end:
                        left_start = len
                        break
                    left_end = left_start + walkLen
                    if left_end > my_end:
                        left_end = len
                    right_start = left_end
                    if right_start > my_end:
                        right_end = len
                    right_end = right_start + walkLen
                    if right_end > my_end:
                        right_end = len
                    solve(tempList, left_start, right_start, old_left_start, my_start, my_end, left_end, right_end, headLoc)
                    left_start = old_left_start + 2 * walkLen
                    current_list = not current_list
            synchronize_threads()
            index = blockIdx.x * blockDim.x + threadIdx.x
            for i in 0 to elementsPerThread - 1:
                if (index + i < length):
                    d_list[index + i] = tempList[current_list][elementsPerThread * threadIdx.x + i]
            synchronize_threads()
            return
        end Device_Merge
      
        void MergeSort(h_list, len, threadsPerBlock, blocks):
            d_list
            allocate_device_memory(d_list, len * sizeof(float))
            copy_input_to_device(d_list, h_list, len * sizeof(float))
            elementsPerThread = ceil(len / float(threadsPerBlock * blocks))
            Device_Merge<<<blocks, threadsPerBlock>>>(d_list, len, elementsPerThread)
            copy_output_to_host(h_list, d_list, len * sizeof(float))
            free_device_memory(d_list)
        end MergeSort
      
        function solve(tempList, left_start, right_start, old_left_start, my_start, my_end, left_end, right_end, headLoc):
            for i = 0 to walkLen - 1:
                if tempList[current_list][left_start] < tempList[current_list][right_start]:
                    tempList[not current_list][headLoc] = tempList[current_list][left_start]
                    left_start = left_start + 1
                    headLoc = headLoc + 1
                    if left_start == left_end:
                        for j = right_start to right_end - 1:
                            tempList[not current_list][headLoc] = tempList[current_list][right_start]
                            right_start = right_start + 1
                            headLoc = headLoc + 1
                else:
                    tempList[not current_list][headLoc] = tempList[current_list][right_start]
                    right_start = right_start + 1
                    if right_start == right_end:
                        for j = left_start to left_end - 1:
                            tempList[not current_list][headLoc] = tempList[current_list][right_start]
                            right_start = right_start + 1
                            headLoc = headLoc + 1
        end solve

    
Source: https://pushkar2196.wordpress.com/2017/04/19/mergesort-cuda-implementation/


- Insertion Sort (MPI)
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

- Insertion Sort (CUDA)
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
