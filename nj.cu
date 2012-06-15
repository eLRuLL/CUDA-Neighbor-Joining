#include <math.h>
#include <fstream>
#include <stdio.h>
#include <exception>

#define MAX_THREADS 1024

using namespace std;

__global__ void nj_paso1(float* mat, float* res,int width)
{
	int idx = blockIdx.x + blockDim.x + threadIdx.x;
	float rpta = 0.0;
	for(int i=0; i<width; i++)
	{
		if(i<idx)
			rpta += mat[idx*width + i];
		else
			rpta += mat[i*width + idx];
	}
	res[idx] = rpta;
}

__global__ void nj_paso2(float* mat, float* mat2, float* diverg, int width, int* fronteras)
{
	int bx = blockIdx.x;
	
	int i = 0;
	int currfil = 0;
	int currcol = 0;
	while(fronteras)
	{
		if(bx < fronteras[i])
			currfil = 1;
			currcol = fronteras[i-1];
	}
	
	int idx = blockIdx.x + blockDim.x + threadIdx.x;
	int idy = blockIdx.y + blockDim.y + threadIdx.y;
	
	mat2[idx*width + idy] = mat[idx*width + idy] - (diverg[idx] + diverg[idy])/(width-2);
}

int main()
{
	int N; 											// number of elements (the same as the width of the matrix).
	int b; 											// dimension of the block
	int numblocks = ceil((double)N/MAX_THREADS); 	// number of necessary blocks in the GPU
	double* M;										// matrix of distances
	double* r;										// array of divergences
	
	char buffer[100];
	try{
		printf("Name of the input_file: ");
		scanf("%s",buffer);
		ifstream input(buffer);
		input>>N;								// getting the number of elements.
		printf("%d elements.\n",N);
		
		// garbage
		input.getline(buffer,100);
		input.getline(buffer,100);
		input.getline(buffer,100);
		
		M = new double[N*N];
		r = new double[N];
		
		// Initialize the matrix with 0-values
		for(int i=0; i<N; i++)
			for(int j=0; j<N; j++)
				M[i*N+j]=0;
		
		// Passing data from input to Matrix
		for(int i=1; i<N; i++){
			for(int j=0; j<i; j++){
				input>>M[i*N + j];
				printf("%4.2f ",M[i*N + j]);
			}
			printf("\n");
		}
				
		// Printing Matrix
		for(int i=0; i<N; i++){
			for (int j=0; j<N; j++)
				printf("%4.2f ",M[i*N + j]);
			printf("\n");
		}
				
		input.close();
	}
	catch(exception& e){
		printf("Problem trying to read file.\n");
		return 1;
	}
	
	//nj_paso1<<<numblocks,numthreads>>>(matrix,divergencias,width);//no-arrays no separan memoria.
	//transportar la data al host.
	
	//int numblocks = ((N/b)*((N/b)+1))/2.0;
	//float* matrix_temp;
	
	//int* fronteras = new int[N/b];
	//for(int i=0; i<N/b ; i++)
	//{
//		fronteras[i] = ((i+1)*(i+2))/2.0;
//	}
	
	//nj_paso2<<<numblocks,(b,b)>>>(matrix,matrix_temp,divergencias,width,fronteras);
	return 0;
}