#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm> // For std::sort
#include <cstdlib>
#include <ctime>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char* sortedInput = "sorted";
const char* randomInput = "random";
const char* reverseSortedInput = "reverse_sorted";
const char* perturbed = "perturbed";


void printArr(int arr[], int n) {
    for (int i = 0; i < n; ++i)
        printf("%d ", arr[i]);
    printf("\n");
}

__device__ int d_size;
__device__ char inputType[16]; // Maximum length for input type

__global__ void generateInputParallel(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);

    if (tid < n) {
        if (inputType[0] == 'r' && inputType[1] == 'a') {
            // Generate random numbers using curand
            arr[tid] = curand(&state) % 10000;
        } else if (inputType[0] == 's' && inputType[1] == 'o') {
            // Assign consecutive values to create a sorted array
            arr[tid] = tid;
        } else if (inputType[0] == 'r' && inputType[1] == 'e') {
            // Assign consecutive values in reverse order to create a reverse sorted array
            arr[tid] = n - tid;
        } else if (inputType[0] == 'p' && inputType[1] == 'e') {
            // Generate sorted array and perturb 1% of elements
            arr[tid] = tid;
            if (tid < n * 0.01) {
                int index = curand(&state) % n;
                int temp = arr[tid];
                arr[tid] = arr[index];
                arr[index] = temp;
            }
        } else {
            // Default to random input if inputType is invalid
            arr[tid] = curand(&state) % 10000;
        }
    }
}
void generateInput(int arr[], int n, const char* inputType, int threads_per_block) {
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("cudaMemcpyToSymbol");
    cudaMemcpyToSymbol(::inputType, inputType, sizeof(char) * 16);
    CALI_MARK_END("cudaMemcpyToSymbol");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    int blocks = n / threads_per_block;
    CALI_MARK_BEGIN("data_init");
    generateInputParallel<<<blocks, threads_per_block>>>(d_arr, n);
    cudaDeviceSynchronize();
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    cudaFree(d_arr);
}


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


void data_init(int arr[], int n, const std::string& inputType) {
    srand(time(NULL));

    if (inputType == "random") {
        // Generate random numbers
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 10000;
        }
    } else if (inputType == "sorted") {
        // Generate sorted array
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }
    } else if (inputType == "reverse_sorted") {
        // Generate reverse-sorted array
        for (int i = 0; i < n; i++) {
            arr[i] = n - i;
        }
    } else if (inputType == "perturbed") {
        // Generate sorted array and perturb 1% of elements
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }

        // Perturb 1% of elements
        for (int i = 0; i < n * 0.01; i++) {
            int index = rand() % n;
            std::swap(arr[i], arr[index]);
        }
    } else {
        std::cerr << "Invalid input type. Using random input by default." << std::endl;
        // Generate random numbers by default
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 10000;
        }
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

__global__ void initialize_reverse_sorted(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        array[tid] = size - tid; // Reverse sorted data
    }
}

__global__ void initialize_sorted(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        array[tid] = tid; // sorted data
    }
}

__global__ void initialize_random(int* array, int size, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    // Generate random numbers and scale them to the desired range
    if (tid < size) {
        array[tid] = static_cast<int>(curand_uniform(&state) * static_cast<float>(INT_MAX));
        // array[tid] = 1;
    }
}

__global__ void initialize_perturbed(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Generate data in parallel with a reverse sorted array
        if (tid % 100 == 0)
            array[tid] = 1; // 1%perturbed
    }
}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");
    
    if (argc != 4) {
        printf("Usage: %s <threads_per_block> <input_size> <input_type>\n", argv[0]);
        return 1;
    }

    int threads_per_block = atoi(argv[1]);
    int input_size = atoi(argv[2]);
    string inputType = argv[3];
    
    const char* inputTypeCStr = inputType.c_str(); // Convert std::string to const char*
    printf("Input Type: %s\n", inputTypeCStr);
    // size_t d_size = input_size * sizeof(int);
    // int *d_array;

    cali::ConfigManager mgr;
    mgr.start();

    int arr[input_size];


    // if (cudaMalloc(&d_array, d_size) != cudaSuccess)
    // {
    //     cout << "D_ARRAY ALLOCATING NOT WORKING!" << endl;
    //     return 0;
    // }

    //  CALI_MARK_BEGIN("data_init");
    // // Generate the data in parallel
    // // Generate in reverse sorted order
    // if (strcmp(inputTypeCStr, sortedInput) == 0) {
    //     printf("sorted input starting\n");
    //     initialize_sorted<<<input_size / threads_per_block, threads_per_block>>>(d_array, input_size);
    //     // input_type = "sorted";
    // }
    // else if (strcmp(inputTypeCStr, randomInput) == 0) {
    //     printf("generating data randomly\n");
    //     initialize_random<<<input_size / threads_per_block, threads_per_block>>>(d_array, input_size, time(NULL));
    //     // input_type = "random";
    // }
    // else if (strcmp(inputTypeCStr, reverseSortedInput) == 0) {
    //     initialize_reverse_sorted<<<input_size / threads_per_block, threads_per_block>>>(d_array, input_size);
    //     // input_type = "reverse_sorted";
    // }
    // else if (strcmp(inputTypeCStr, perturbed) == 0) {
    //     initialize_perturbed<<<input_size / threads_per_block, threads_per_block>>>(d_array, input_size);
    //     // input_type = "perturbed";
    // }

    // cudaDeviceSynchronize();

    // printf("data generated\n");
    
    // cudaMemcpy(arr, d_array, d_size, cudaMemcpyDeviceToHost); 
    
    CALI_MARK_BEGIN("data_init");
    data_init(arr, input_size, inputType);
    // generateInput(arr, input_size, inputTypeCStr, threads_per_block); // 128 threads per block
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
       
        partition<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
        cudaDeviceSynchronize();
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");
        int answer;
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_small");
        CALI_MARK_BEGIN("cudaMemcpyFromSymbol");
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
        CALI_MARK_END("cudaMemcpyFromSymbol");
        CALI_MARK_END("comm_small");
        
        n_t = threads_per_block;
        n_i = answer;
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("cudaMemcpy");
        cudaMemcpy(arr, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        CALI_MARK_END("cudaMemcpy");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
    }
    

    string algorithm = "QuickSort";
    string programmingModel = "CUDA";
    string datatype = "int";
    int sizeOfDatatype = sizeof(int);
    int inputSize = input_size;
    // string inputType = "Random";
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

