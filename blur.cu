// Compute vecotr Sum C = A + B
// Each thread performs one pair-wise addition
// callable from host executed on device
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
	// This variable i will be local to the thread

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// The sequence may not be divisible by the block size:
	// So that means that we need to skip anything bigger than n
	// even if a thread was launched for it.
	if (i < n ) {
		C[i] = A[i] + B[i];
	}
}

void blur(float *A, float* B, float* C, int n ) {
	float *A_d, *B_d, *C_d;
	int size = n * sizeof(float);
	
	// Set up memory on device
	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);

	// Copy the vectors to device
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(C_d, C, size, cudaMemcpyHostToDevice);

	// Execute the Kernel
	// 256 defines the block size and how many threads blocks we'll launch
	// e.g. if n = 1000 we'll launch 4 thread blocks.
	// Execution order will be random as fuck!
	vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

	// Copy the answer back to host memory
	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

	// Free up the memory on the device
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

int main(int argc, char** argv) {
    // Define vector size
    int n = 1000;

    // Allocate and initialize vectors A, B, and C
    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float *C = (float*)malloc(n * sizeof(float));

    // Initialize vectors A and B with some values
    for (int i = 0; i < n; ++i) {
        A[i] = 1.0f; // Set some arbitrary values for demonstration
        B[i] = 2.0f;
    }

    // Call the vecadd function to perform vector addition on the GPU
    vecadd(A, B, C, n);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
