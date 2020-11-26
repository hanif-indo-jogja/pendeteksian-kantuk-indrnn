from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.nn import Parameter
from .cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN

class Batch_norm_overtime(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_overtime, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.bn(x.clone())
        x = x.permute(2, 0, 1)
        return x

class Linear_overtime_module(nn.Module):
    def __init__(self, input_size, hidden_size,bias=True):
        super(Linear_overtime_module, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        y = x.contiguous().view(-1, self.input_size)
        y = self.fc(y)
        y = y.view(x.size()[0], x.size()[1], self.hidden_size)
        return y

class Dropout_overtime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p=0.5, training=False):
        output = input.clone()
        noise = input.data.new(input.size(-2), input.size(-1))  # torch.ones_like(input[0])
        if training:
            noise.bernoulli_(1 - p).div_(1 - p)
            noise = noise.unsqueeze(0).expand_as(input)
            output.mul_(noise)
        ctx.save_for_backward(noise)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        noise, = ctx.saved_tensors
        if ctx.training:
            return grad_output.mul(noise), None, None
        else:
            return grad_output, None, None