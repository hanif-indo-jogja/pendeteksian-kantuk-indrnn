import torch
import torch.nn as nn
import numpy as np
from .cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN


args = None
U_bound = 0
U_lowbound = None

def init(anArgs, aU_bound):
    global args
    global U_bound
    global U_lowbound

    args = anArgs
    U_bound = aU_bound
    U_lowbound = np.power(10, (np.log10(1.0 / args.MAG) / args.seq_len))

from .utils import Batch_norm_overtime, Linear_overtime_module, Dropout_overtime
BN = Batch_norm_overtime
Linear_overtime = Linear_overtime_module
dropout_overtime = Dropout_overtime.apply


class IndRNNwithBN(nn.Sequential):
    def __init__(self, hidden_size, seq_len):
        super(IndRNNwithBN, self).__init__()  
        self.add_module('indrnn1', IndRNN(hidden_size))         
        self.add_module('norm1', BN(hidden_size, args.seq_len))

class IndrnnPlainnet(nn.Module):
    def __init__(self, input_size, outputclass):
        super(IndrnnPlainnet, self).__init__()
        hidden_size = args.hidden_size
        
        self.dense_input_list = nn.ModuleList()
        dense_input = Linear_overtime(input_size, hidden_size)
        self.dense_input_list.append(dense_input)
        for x in range(args.num_layers - 1):
            dense_input = Linear_overtime(hidden_size, hidden_size)
            self.dense_input_list.append(dense_input)   

        self.RNNs = nn.ModuleList()
        for x in range(args.num_layers):
            rnn = IndRNNwithBN(hidden_size=hidden_size, seq_len=args.seq_len)
            self.RNNs.append(rnn)         
            
        self.classifier = nn.Linear(hidden_size, outputclass, bias=True)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'recurrent_weight' in name: 
                param.data.uniform_(U_lowbound,U_bound)    
            if ('fc' in name) and 'weight' in name:
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if ('norm' in name)  and 'weight' in name:
                param.data.fill_(1)
            if 'bias' in name:
                param.data.fill_(0.0)


    def forward(self, input):
        rnnoutputs = {}    
        rnnoutputs['outlayer-1'] = input
        for x in range(len(self.RNNs)):
            rnnoutputs['dilayer%d' % x] = self.dense_input_list[x](rnnoutputs['outlayer%d' % (x - 1)])    
            rnnoutputs['outlayer%d' % x] = self.RNNs[x](rnnoutputs['dilayer%d' % x])          
            if args.dropout > 0:
                rnnoutputs['outlayer%d' % x] = dropout_overtime(
                                            rnnoutputs['outlayer%d'%x],args.dropout,self.training) 
        temp = rnnoutputs['outlayer%d' % (len(self.RNNs) - 1)][-1]
        output = self.classifier(temp)
        return output                
