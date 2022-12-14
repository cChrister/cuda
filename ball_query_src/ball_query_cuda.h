/*************************************************************************
    > File Name: ball_query_cuda.h
    > Author: steve
    > E-mail: yqykrhf@163.com 
    > Created Time: Mon 14 Nov 2022 12:54:37 PM CST
    > Brief: 
 ************************************************************************/

#ifndef _BALL_QUERY_CUDA_H
#define _BALL_QUERY_CUDA_H


#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

// Bind with pybind11, to call cuda function
int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample,
		at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

// Function in .cu
void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
		const float* xyz, const float* new_xyz, int *idx);

#endif
