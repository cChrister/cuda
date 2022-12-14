/*************************************************************************
    > File Name: ball_query.cpp
    > Author: steve
    > E-mail: yqykrhf@163.com 
    > Created Time: Mon 14 Nov 2022 12:49:33 PM CST
    > Brief: ball query cpu implementation
    > Log:
    1. Remove warning. Replace `x.type().is_cuda()` by `x.is_cuda()`
     /home/renhaofan/pytorch_cuda/ball_query_src/ball_query.cpp:21:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. 
     Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. 
     If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of
     tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
    2. Remove warning. Replace `.data<float>()` by `.data_ptr<float>()`
    warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]

 ************************************************************************/

#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "ball_query_cuda.h"

extern THCState *state;

// Check data type
#define CHECK_CUDA(x) do { \
  if (!x.is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
	  exit(-1); \
  } \
} while(0)

#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
	  exit(-1); \
  } \
} while(0)

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample,
  at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
  
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor); 
  const float *new_xyz = new_xyz_tensor.data_ptr<float>();
  const float *xyz = xyz_tensor.data_ptr<float>();
  int *idx = idx_tensor.data_ptr<int>();

  ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx);
  return 1;
}




