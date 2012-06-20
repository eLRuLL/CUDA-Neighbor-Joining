#include <math.h>
#include <fstream>
#include <stdio.h>
#include <exception>

#define MAX_THREADS 1024

using namespace std;

__global__ void nj_step1(float* mat, float* res,int width)		// Calculate the tree-divergence for every object.
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < width){
		float rpta = 0.0f;
		for(int i=0; i<width; i++){
			if(i<idx)
				rpta += mat[idx*width + i];
			else
				rpta += mat[i*width + idx];
		}
		res[idx] = rpta;
	}
}

__global__ void nj_step2(float* mat_t, float* mat, float* diverg, int width, int* limits) // Calculate a new matrix (Mt) of distances.
{
	int bx = blockIdx.x;
	
	int k = 0;
	int blockfil = 0;
	int blockcol = 0;
	while(limits[k] && limits[k] < bx){
		k++;
	}
	if(k!=0)
		blockfil = k - 1;
	
	if(k!=0)
		blockcol = limits[k - 1];
	
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	
	int i = (blockfil * blockDim.x) + idx;
	int j = (blockcol * blockDim.y) + idy;
	
	mat_t[i*width + j] = mat[i*width + j] - (diverg[i] + diverg[j])/(width-2);
}

int main()
{
	int N; 											// number of elements (the same as the width of the matrix).
	int numblocks;									// number of necessary blocks in the GPU
	int b; 											// dimension of the block
	float* M;										// matrix of distances
	float* Mt;										// temporal matrix for finding the smallest values.
	float* r;										// array of divergences
	
	char buffer[100];
	try{
		printf("Name of the input_file: ");
		scanf("%s",buffer);
		ifstream input(buffer);
		input>>N;									// getting the number of elements.
		printf("%d elements.\n",N);
		
		// garbage
		input.getline(buffer,100);
		input.getline(buffer,100);
		input.getline(buffer,100);
		
		M = new float[N*N];
		r = new float[N];
		
		// Initialize the matrix with 0-values
		for(int i=0; i<N; i++)
			for(int j=0; j<N; j++)
				M[i*N+j]=0;
		
		// Passing data from input to Matrix
		for(int i=1; i<N; i++)
			for(int j=0; j<i; j++)
				input>>M[i*N + j];
				
		// Printing Matrix
		printf("Printing input matrix");
		for(int i=0; i<N; i++){
			for (int j=0; j<N; j++)
				printf("%4.2f ",M[i*N + j]);
			printf("\n");
		}
		printf("----------------------- o ----------------------\n\n");
				
		input.close();
	}catch(exception& e){
		printf("Problem trying to read file.\n");
		return 1;
	}
	
	while(N>2)
	{
		printf("***********************N=%d***********************\n\n",N);
		
		numblocks = ceil((float)N/MAX_THREADS);			// Update the number of blocks for every iteration.
		r = new float[N];
		
		Mt = new float[N*N];							// initializing the temporal Matrix.
		
		float* r_d;										// Allocate divergency array in the device.
		cudaMalloc((void**) &r_d, sizeof(float)*N);
		
		float* M_d;										// Allocate distance matrix in the device and copy.
		cudaMalloc((void**) &M_d, sizeof(float)*N*N);
		cudaMemcpy(M_d,M,sizeof(float)*N*N,cudaMemcpyHostToDevice);
		
		nj_step1<<<numblocks,N>>>(M_d,r_d,N);			// Kernel launch for step 1.
		
		cudaMemcpy(r,r_d,sizeof(float)*N,cudaMemcpyDeviceToHost);	// Copying response array to the Host.
		
		// Printing new divergence matrix.
		for(int i=0; i<N; i++)
			printf("%4.2f ",r[i]);
		printf("\n");
		
		int numblocks = ((N/b)*((N/b)+1))/2.0;			// Number of blocks like a triangular matrix.
		
		int* limits = new int[N/b];
		for(int i=0; i<N/b ; i++)
			limits[i] = (int)((((i+1)*(i+2))/2.0) - 1);
		
		float* Mt_d;
		cudaMalloc((void**) &Mt_d, sizeof(float)*N*N);
		float* limits_d;
		cudaMalloc((void**) &limits_d, sizeof(float)*N/b);
		
		nj_step2<<<numblocks,(b,b)>>>(Mt_d,M_d,r_d,N,limits_d);	// Kernel launch for step 2.
		
		cudaMemcpy(Mt,Mt_d,sizeof(float)*N*N,cudaMemcpyDeviceToHost);	// Copying response matrix to the Host.
		
		// Printing temporal distance matrix (Mt).
		for(int i=0; i<N; i++){
			for(int j=0; j<N; j++)
				printf("%4.2f ",Mt[i*N + j]);
			printf("\n");
		}
		
		// Step 3: Select objects "i" and "j" where M[i][j] is the minimum,
		// Step 4: Create a new object U and delete "i" and "j".
		// Step 5: Calculate distances from "i" to U and "j" to U.
		// Step 6: Calculate the distance between U and the rest.
		
		N = N - 1;
	}
	return 0;
}