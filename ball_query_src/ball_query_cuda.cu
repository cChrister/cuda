/*************************************************************************
    > File Name: ball_query_cuda.cu
    > Author: steve
    > E-mail: yqykrhf@163.com 
    > Created Time: Mon 14 Nov 2022 12:54:37 PM CST
    > Brief: 
 ************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_cuda.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample, 
  	const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float radius2 = radius * radius;
  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
	float x = xyz[k * 3 + 0];
	float y = xyz[k * 3 + 1];
	float z = xyz[k * 3 + 2];
	float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
	if(d2 < radius2) {
	  if (cnt == 0) {
		for (int l = 0; l < nsample; ++l) {
		  idx[l] = k; 
		}
	  }
	}
	idx[cnt] = k;
	++cnt;
	if (cnt > nsample) break;
  }
}

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
    //   fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	  fprintf(stderr, "CUDA kernel failed: %s,  at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); 
      exit(-1);
    }
}