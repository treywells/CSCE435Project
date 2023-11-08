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
	#include "mpi.h"
    #include<stdio.h>
    #include <stdlib.h>
    #include "math.h"
    #include <stdbool.h>
    #define SIZE 1000000

/*
    Divides the array given into two partitions
        - Lower than pivot
        - Higher than pivot
    and returns the Pivot index in the array
*/
int partition(int *arr, int low, int high){
    int pivot = arr[high];
    int i = (low - 1);
    int j,temp;
    for (j=low;j<=high-1;j++){
	if(arr[j] < pivot){
	     i++;
             temp=arr[i];  
             arr[i]=arr[j];
             arr[j]=temp;	
	}
    }
    temp=arr[i+1];  
    arr[i+1]=arr[high];
    arr[high]=temp; 
    return (i+1);
}

/*
    Hoare Partition - Starting pivot is the middle point
    Divides the array given into two partitions
        - Lower than pivot
        - Higher than pivot
    and returns the Pivot index in the array
*/
int hoare_partition(int *arr, int low, int high){
    int middle = floor((low+high)/2);
    int pivot = arr[middle];
    int j,temp;
    // move pivot to the end
    temp=arr[middle];  
    arr[middle]=arr[high];
    arr[high]=temp;

    int i = (low - 1);
    for (j=low;j<=high-1;j++){
        if(arr[j] < pivot){
            i++;
            temp=arr[i];  
            arr[i]=arr[j];
            arr[j]=temp;	
        }
    }
    // move pivot back
    temp=arr[i+1];  
    arr[i+1]=arr[high];
    arr[high]=temp; 

    return (i+1);
}

/*
    Simple sequential Quicksort Algorithm
*/
void quicksort(int *number,int first,int last){
    if(first<last){
        int pivot_index = partition(number, first, last);
        quicksort(number,first,pivot_index-1);
        quicksort(number,pivot_index+1,last);
    }
}

/*
    Functions that handles the sharing of subarrays to the right clusters
*/
int quicksort_recursive(int* arr, int arrSize, int currProcRank, int maxRank, int rankIndex) {
    MPI_Status status;

    // Calculate the rank of the Cluster which I'll send the other half
    int shareProc = currProcRank + pow(2, rankIndex);
    // Move to lower layer in the tree
    rankIndex++;

    // If no Cluster is available, sort sequentially by yourself and return
    if (shareProc > maxRank) {
        MPI_Barrier(MPI_COMM_WORLD);
	    quicksort(arr, 0, arrSize-1 );
        return 0;
    }
    // Divide array in two parts with the pivot in between
    int j = 0;
    int pivotIndex;
    pivotIndex = hoare_partition(arr, j, arrSize-1 );

    // Send partition based on size(always send the smaller part), 
    // Sort the remaining partitions,
    // Receive sorted partition
    if (pivotIndex <= arrSize - pivotIndex) {
        MPI_Send(arr, pivotIndex , MPI_INT, shareProc, pivotIndex, MPI_COMM_WORLD);
	    quicksort_recursive((arr + pivotIndex+1), (arrSize - pivotIndex-1 ), currProcRank, maxRank, rankIndex); 
        MPI_Recv(arr, pivotIndex , MPI_INT, shareProc, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    else {
        MPI_Send((arr + pivotIndex+1), arrSize - pivotIndex-1, MPI_INT, shareProc, pivotIndex + 1, MPI_COMM_WORLD);
        quicksort_recursive(arr, (pivotIndex), currProcRank, maxRank, rankIndex);
        MPI_Recv((arr + pivotIndex+1), arrSize - pivotIndex-1, MPI_INT, shareProc, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
}


int main(int argc, char *argv[]) {
    int unsorted_array[SIZE]; 
    int array_size = SIZE;
    int size, rank;
    // Start Parallel Execution
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank==0){
        // --- RANDOM ARRAY GENERATION ---
        printf("Creating Random List of %d elements\n", SIZE);
        int j = 0;
        for (j = 0; j < SIZE; ++j) {
            unsorted_array[j] =(int) rand() % 1000;
        }
        printf("Created\n");
	}

    // Calculate in which layer of the tree each Cluster belongs
    int rankPower = 0;
    while (pow(2, rankPower) <= rank){
        rankPower++;
    }
    // Wait for all clusters to reach this point 
    MPI_Barrier(MPI_COMM_WORLD);
    double start_timer, finish_timer;
    if (rank == 0) {
	    start_timer = MPI_Wtime();
        // Cluster Zero(Master) starts the Execution and
        // always runs recursively and keeps the left bigger half
        quicksort_recursive(unsorted_array, array_size, rank, size - 1, rankPower);    
    }else{ 
        // All other Clusters wait for their subarray to arrive,
        // they sort it and they send it back.
        MPI_Status status;
        int subarray_size;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // Capturing size of the array to receive
        MPI_Get_count(&status, MPI_INT, &subarray_size);
	    int source_process = status.MPI_SOURCE;     
        int subarray[subarray_size];
        MPI_Recv(subarray, subarray_size, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        quicksort_recursive(subarray, subarray_size, rank, size - 1, rankPower);
        MPI_Send(subarray, subarray_size, MPI_INT, source_process, 0, MPI_COMM_WORLD);
    };
    
    if(rank==0){
        finish_timer = MPI_Wtime();
	    printf("Total time for %d Clusters : %2.2f sec \n",size, finish_timer-start_timer);

        // --- VALIDATION CHECK ---
        printf("Checking.. \n");
        bool error = false;
        int i=0;
        for(i=0;i<SIZE-1;i++) { 
            if (unsorted_array[i] > unsorted_array[i+1]){
		        error = true;
                printf("error in i=%d \n", i);
            }
        }
        if(error)
            printf("Error..Not sorted correctly\n");
        else
            printf("Correct!\n");        
    }
       
    MPI_Finalize();
    // End of Parallel Execution
    return 0;
}
 	```
 
source: https://github.com/triasamo1/Quicksort-Parallel-MPI/blob/master/quicksort_mpi.c

- Quick Sort (CUDA)
  
	```   
	#include<stdio.h>
    #include <cuda.h>
    #include<cuda_runtime.h>

 __global__ static void quicksort(int* values,int N) {
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


int main(){
  int x[20],size,i;
  int *d_x,*d_size,*d_i;

  printf("Enter size of the array: ");
  scanf("%d",&size);

  printf("Enter %d elements: ",size);
  for(i=0;i<size;i++)
    scanf("%d",&x[i]);
  	cudaMalloc((void **)&d_x,sizeof(int)*size);
    cudaMalloc((void **)&d_size,sizeof(int));
    cudaMalloc((void **)&d_i,sizeof(int));

  cudaMemcpy(d_x, &x,  sizeof( int)*size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_size, &size,  sizeof( int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_i, &i,  sizeof( int), cudaMemcpyHostToDevice);

  quicksort<<<1,1,size>>>(d_x,size);
  cudaMemcpy(x, d_x, sizeof(int)*size, cudaMemcpyDeviceToHost);
  printf("Sorted elements: ");
  for(i=0;i<size;i++)
    printf(" %d",x[i]);

 cudaFree(d_x);
 cudaFree(d_size);
 cudaFree(d_i);



  return 0;
}
	```
 
source: https://github.com/GreyVader1993/Cuda-Programs/blob/main/QuickSort.cu

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
