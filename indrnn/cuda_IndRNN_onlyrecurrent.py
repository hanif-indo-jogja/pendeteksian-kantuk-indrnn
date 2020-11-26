"""
This code is to implement the IndRNN (only the recurrent part) using CUDA for fast computation. The CUDA part is similar the SRU implementation from 
https://github.com/taolei87/sru.
This runs around 32 times faster than the general pytorch implementation on pixel MNIST example (sequence lengeth 784). For longer sequence, 
it will be even more efficient, and vice versa. 
Since this only contains the recurrent part of IndRNN, fully connected layers or convolutional layers are needed before it.
Please cite the following paper if you find it useful.
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN," 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5457-5466. 2018.
@inproceedings{li2018independently,
  title={Independently recurrent neural network (indrnn): Building A longer and deeper RNN},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5457--5466},
  year={2018}
}
"""

import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from torch.nn import Parameter

from cupy.cuda import function as cuda_function
from .pynvrtc_compiler import Program
from collections import namedtuple

IndRNN_CODE = """
extern "C" {

    __forceinline__ __device__ float relu_activation(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    __forceinline__ __device__ float gradient_relu_activation(float x)
    {
        return (x > 0.f) ? 1.f : 0.f;
    }

    __global__ void indrnn_fwd( const float * __restrict__ x,
                            const float * __restrict__ recurrent_weight, 
                            const float * __restrict__ hidden0,
                            const int len, const int batch, const int hidden_size, 
                            float * __restrict__ hidden)
    {
        int ncols = batch * hidden_size;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;       
        const float current_recurrent_weight = *(recurrent_weight + (col % hidden_size));
        float current_hidden = *(hidden0 + col);
        const float *xp = x + col;
        float *hp = hidden + col;

        for (int row = 0; row < len; ++row)
        {
            current_hidden = relu_activation(current_hidden * current_recurrent_weight + (*xp));
            *hp = current_hidden;
            xp += ncols;
            hp += ncols;            
        }
    }

    __global__ void indrnn_bwd(const float * __restrict__ x,
                             const float * __restrict__ recurrent_weight, const float * __restrict__ hidden0,
                             const float * __restrict__ hidden,
                            const float * __restrict__ grad_hidden, 
                            const int len, const int batch, const int hidden_size, 
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_recurrent_weight, float * __restrict__ grad_hidden0)
    {    
        int ncols = batch * hidden_size;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;        
        const float current_recurrent_weight = *(recurrent_weight + (col % hidden_size));
        float grad_current_recurrent_weight = 0;
        float current_hidden = 0;
        
        const float *xp = x + col + (len - 1) * ncols;
        const float *hp = hidden + col + (len - 1) * ncols;      
        float *gxp = grad_x + col + (len - 1) * ncols;
        const float *ghp = grad_hidden + col + (len - 1) * ncols;
        

        for (int row = len - 1; row >= 0; --row)
        {        
            const float prev_hidden_val = (row > 0) ? (*(hp - ncols)) : (*(hidden0 + col));

            float grad_hidden_before_activation = (*ghp) * gradient_relu_activation(
                prev_hidden_val * current_recurrent_weight + (*xp));
            grad_current_recurrent_weight += grad_hidden_before_activation * prev_hidden_val;
            *gxp = grad_hidden_before_activation;

            xp -= ncols;
            hp -= ncols;
            gxp -= ncols;
            ghp -= ncols;        
        }

        float *grad_rwp = grad_recurrent_weight + (col % hidden_size);
        *grad_rwp += grad_current_recurrent_weight;
        *(grad_hidden0 + col) = current_hidden;
    }
}
"""

class IndRNN_Compute_GPU(Function):

    _IndRNN_PROG = Program(IndRNN_CODE, 'indrnn_prog.cu')
    _IndRNN_PTX = _IndRNN_PROG.compile()
    _DEVICE2FUNC = {}

    @staticmethod
    def compile_functions():
        device = torch.cuda.current_device()
        mod = cuda_function.Module()
        mod.load(bytes(IndRNN_Compute_GPU._IndRNN_PTX.encode()))
        fwd_func = mod.get_function('indrnn_fwd')
        bwd_func = mod.get_function('indrnn_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        IndRNN_Compute_GPU._DEVICE2FUNC[device] = (current_stream, fwd_func, bwd_func)
        return current_stream, fwd_func, bwd_func

    @staticmethod
    def get_functions():
        res = IndRNN_Compute_GPU._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else IndRNN_Compute_GPU.compile_functions()

    @staticmethod
    def forward(ctx, x, recurrent_weight, hidden0):
        length = x.size(0)
        batch = x.size(-2)
        hidden_size = x.size(-1)
        ncols = batch * hidden_size
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block+1
        
        size = (length, batch, hidden_size)
        hidden = x.new(*size)

        stream, fwd_func, _ = IndRNN_Compute_GPU.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            x.contiguous().data_ptr(),
            recurrent_weight.contiguous().data_ptr(),
            hidden0.contiguous().data_ptr(),
            length,
            batch,
            hidden_size,
            hidden.contiguous().data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream = stream
        )

        ctx.save_for_backward(x, hidden, recurrent_weight, hidden0)#
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden):
        x, hidden, recurrent_weight, hidden0 = ctx.saved_tensors
        length = x.size(0)
        batch = x.size(-2)
        hidden_size = x.size(-1)
        ncols = batch * hidden_size
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        grad_x = x.new(*x.size())
        grad_recurrent_weight = x.new(hidden_size).zero_()
        grad_hidden0 = x.new(batch, hidden_size)  

        stream, _, bwd_func = IndRNN_Compute_GPU.get_functions()
        FUNC = bwd_func
        FUNC(args=[
            x.contiguous().data_ptr(),
            recurrent_weight.contiguous().data_ptr(),
            hidden0.contiguous().data_ptr(),
            hidden.contiguous().data_ptr(),
            grad_hidden.contiguous().data_ptr(),
            length,
            batch,
            hidden_size,
            grad_x.contiguous().data_ptr(),
            grad_recurrent_weight.contiguous().data_ptr(),
            grad_hidden0.contiguous().data_ptr()],
            block = (thread_per_block, 1, 1), grid = (num_block, 1, 1),
            stream = stream
        )

        return grad_x, grad_recurrent_weight, grad_hidden0


class IndRNN_onlyrecurrent(nn.Module):
    def __init__(self, hidden_size):
        super(IndRNN_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_weight = Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "recurrent_weight" in name:
                nn.init.uniform_(weight, a=0, b=1)

    def forward(self, input):
        assert input.dim() == 3        
        
        hidden0 = input.data.new(input.size(-2),input.size(-1)).zero_()
                    
        IndRNN_Compute = IndRNN_Compute_GPU().apply
        return IndRNN_Compute(input, self.recurrent_weight, hidden0)
