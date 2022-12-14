#################################### 
#    File Name: test_ball_query.py
#    Author: steve
#    E-mail: yqykrhf@163.com
#    Created Time: Mon 14 Nov 2022 03:24:54 PM CST
#    Brief: 
####################################


import time
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import math

import ball_query_cuda

# 定义该方法的前向传播和反向传播方法
class BallQuery(Function):
    
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
        
        ball_query_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply

xyz = torch.randn(2, 128, 3).cuda()
new_xyz = xyz

result = ball_query(0.8, 3, xyz, new_xyz)

print(result.shape)
