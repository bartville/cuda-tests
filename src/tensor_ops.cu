#include <stdio.h>
#include <math.h>
#include <time.h>


void add(int n, float* x, float* y) {
  for(int i = 0; i < n; ++i)
    y[i] += x[i]; 
}

void add(int x_size, int y_size, int z_size, float*** t1, float*** t2) {
  for(int x = 0; x < x_size; ++x)
    for(int y = 0; y < y_size; ++y)
      for(int z = 0; z < z_size; ++z)
        t1[x][y][z] += t2[x][y][z]; 
}

__global__
void add_gpu(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(int i = index; i < n; i+=stride)
    y[i] += x[i];
}

int main(void) {
  int X = 500;
  int Y = 500;
  int Z = 500;
  int N = X*Y*Z;

  float *v1, *v2;
  cudaMallocManaged(&v1, N*sizeof(float));
  cudaMallocManaged(&v2, N*sizeof(float)); 
  
  float ***t1, ***t2;
  t1 = new float**[X];
  t2 = new float**[X];
  for(int x = 0; x < X; ++x) {
    t1[x] = new float*[Y];
    t2[x] = new float*[Y];
    for(int y = 0; y < Y; ++y) {
      t1[x][y] = new float[Z];
      t2[x][y] = new float[Z];      
    }
  }

  for(int x = 0; x < X; ++x)
    for(int y = 0; y < Y; ++y)
      for(int z = 0; z < Z; ++z) {
        v1[x + y*X + z*X*Y] = 1.0f;
        v2[x + y*X + z*X*Y] = 2.0f;
        t1[x][y][z] = 1.0f;
        t2[x][y][z] = 2.0f;
      }

  // sum
  clock_t t;
  t = clock();
  add(X*Y*Z, v1, v2);
  t = clock() - t;
  printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
  t = clock();
  add(X, Y, Z, t1, t2);
  t = clock() - t;
  printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);  

  int blockSize = 256;
  int numBlocks = (N + blockSize-1) / blockSize;
  t = clock();
  add_gpu<<< numBlocks, blockSize >>>(N, v1, v2);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  t = clock() - t;
  printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);  
  
  
  cudaFree(v1);
  cudaFree(v2);
  
  return 0;
}
  
