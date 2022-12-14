# Introduction
This is simple demo to build custom CUDA operator in torch with C++ backend.
After that, you could call it by python frontend by simple run command
`python test_ball_query.py`
# How to build
Before build, make sure CUDA environment is fine. Take mine for example:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0

$ ll /usr/local/cuda
lrwxrwxrwx 1 root root 9 Oct 15 21:48 /usr/local/cuda -> cuda-11.2/

$ g++ --version
g++ (Ubuntu 8.4.0-1ubuntu1~18.04) 8.4.0
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
## 1. Switch python environment 
```bash
$ conda env create -f environment.yml
$ conda activate pointnet
```
Check python torch env:
```bash
$ python torch_cuda_check.py
True
True
7605
True
True
=====================================
Pytorch version 1.6.0
Pytorch CUDA 10.2
Pytorch cudnn 7605
====================================
```
> Note: Pytorch CUDA 10.2 mismatch the `nvcc --version` mentioned before. In this demo, it will pass and no error. However, it's better to alignd the base CUDA version with python CUDA version.

## 2. Build by setuptools
The corresponding `.h` located in torch/include
```
$ ls ~/.conda/envs/pointnet/lib/python3.7/site-packages/torch/include
ATen  c10  c10d  caffe2  pybind11  TH  THC  THCUNN  torch
```
`#include <THC/THC.h>` used by ball_query_src/ball_query.cpp

After all prepared, the run command
```
$ python setup.py install &>
...
Installed /home/renhaofan/.conda/envs/pointnet/lib/python3.7/site-packages/ballquery-0.0.0-py3.7-linux-x86_64.egg
Processing dependencies for ballquery==0.0.0
Finished processing dependencies for ballquery==0.0.0
``` 
If `Finished processing dependencies for ballquery==0.0.0` occured, compile is finieshed and right.

## 3. Run python frontend
```
$ python test_ball_query.py
```

# Reference
* https://twn29004.top/2022/01/09/pytorchwithcuda/
* https://blog.csdn.net/wqwqqwqw1231/article/details/106902235