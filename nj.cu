using namespace std;

__global__ nj_paso1(float* mat, float* res,int width)
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

__global__ nj_paso2(float* mat, float* mat2, float* diverg, int width, int* fronteras)
{
	__shared__ int bx = blockIdx.x;
	
	int i = 0;
	int currfil = 0;
	int currcol = 0;
	while(fronteras)
	{
		if(bx < fronteras[i])
			currfil = 1;
			currcol = fronteras[i-1];
	}
	
	//int idx = blockIdx.x + blockDim.x + threadIdx.x;
	//int idy = blockIdx.y + blockDim.y + threadIdx.y;
	
	mat2[idx*width + idy] = mat[idx*width + idy] - (diverg[idx] + diverg[idy])/(n-2);
}

int main()
{
	int N;
	int b; // dimensionalidad del bloque
	int width;
	dim3 numblocks;
	dim3 numthreads;
	float* matrix;
	float* divergencias;
	nj_paso1<<<numblocks,numthreads>>>(matrix,divergencias,width);//no-arrays no separan memoria.
	//transportar la data al host.
	
	int numblocks = ((N/b)*((N/b)+1))/2.0;
	float* matrix_temp;
	
	int* fronteras = new int[N/b];
	for(int i=0; i<N/b ; i++)
	{
		fronteras[i] = ((i+1)*(i+2))/2.0;
	}
	
	nj_paso2<<<numblocks,(b,b)>>>(matrix,matrix_temp,divergencias,width,fronteras);
	return 0;
}