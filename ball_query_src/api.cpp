/*************************************************************************
    > File Name: api.cpp
    > Author: steve
    > E-mail: yqykrhf@163.com 
    > Created Time: Mon 14 Nov 2022 12:49:33 PM CST
    > Brief: Bind function declared in ball_query.cpp with python inference
 ************************************************************************/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ball_query_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 第一个参数表示的是在python中调用的名称，第二个参数是对应的cpp函数，第三个参数对应的是这个函数的说明
  m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");
}
