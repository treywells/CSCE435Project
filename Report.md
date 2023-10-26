# CSCE 435 Group project

## 1. Group members:
1. Trey Wells
2. Aaron Weast
3. Jacob Miller
4. David Vilenchouk

---

## 2. _due 10/25_ Project topic

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For the duration of this project, our team plans on communicating via Slack. 

For our algorithms, we plan on implementing various sorting algorithms. The three sorting algorithms we are planning on implementing are Bubble sort, Merge Sort, and Quick sort. 

For each of the algorithms, we are planning on implementating in both OpenMP and CUDA so that we can compare the differences in CPU vs. GPU parallelization. Not only will we be comparing the differences in CPU and GPU speed but we will also be testing the differences in the algorithms on various types of inputs. For example, we might run each algorithm on a completely random input, then on a partially sorted one, then on a completely sorted one. 

## Psuedocode for Algorithms

For example:
- Bubble Sort (MPI)

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


    Source: https://people.cs.pitt.edu/~bmills/docs/teaching/cs1645/lecture_par_sort.pdf

- Bubble Sort (CUDA)

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

    Source: https://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/SLIDES/s19.html

- Quick Sort (MPI)
- Quick Sort (CUDA)
   
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
      '''
source: https://github.com/saigowri/CUDA/blob/master/quicksort.cu
- Merge Sort (MPI)


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
