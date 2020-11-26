import argparse
import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
import numpy as np

# Set the random seed manually for reproducibility.
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import indrnn.opts as opts    
parser = argparse.ArgumentParser(description='pytorch action')
opts.train_opts(parser)
args = parser.parse_args()

outputclass = 2
indim = 4
batch_size = args.batch_size
seq_len = args.seq_len

if args.U_bound == 0:
    U_bound = np.power(10, (np.log10(args.MAG) / args.seq_len))   
else:
    U_bound = args.U_bound

import indrnn.Indrnn_plainnet as Indrnn_network
Indrnn_network.init(anArgs = args, aU_bound = U_bound)

model = Indrnn_network.IndrnnPlainnet(indim, outputclass)

model.cuda()
criterion = nn.CrossEntropyLoss()

#Adam with lr 2e-4 works fine.
learning_rate = args.learning_rate

param_decay = []
param_nodecay = []
for name, param in model.named_parameters():
    if 'recurrent_weight' in name or 'bias' in name:
        param_nodecay.append(param)      
    else:
        param_decay.append(param)      
            
optimizer = torch.optim.Adam([
        {'params': param_nodecay},
        {'params': param_decay, 'weight_decay': args.decayfactor}
    ], lr = learning_rate) 

from data_gen import DataHandler, EvalDataHandler, TestDataHandler
dh_train = DataHandler(batch_size)
dh_eval = EvalDataHandler(batch_size)
dh_test = TestDataHandler()
num_train_batches = int(np.ceil(dh_train.get_dataset_size() / (batch_size + 0.0)))
num_eval_batches = int(np.ceil(dh_eval.get_dataset_size() / (batch_size + 0.0)))
x, y = dh_train.get_batch()

seq_len = x.shape[1]
if seq_len != args.seq_len:
    print('error seq_len')
    assert False


def train(num_train_batches):
    model.train()
    tacc = 0
    count = 0
    for batchi in range(0,num_train_batches):
        inputs,targets = dh_train.get_batch()
        
        inputs = inputs.transpose(1, 0, 2)
        inputs = torch.from_numpy(inputs).cuda()

        targets = torch.from_numpy(np.int64(targets)).cuda()

        model.zero_grad()
        if args.constrain_U:
            clip_weight(model, U_bound)
        output = model(inputs)
        loss = criterion(output, targets)

        pred = output.data.max(1)[1] # get the index of the max log-probability
        accuracy = pred.eq(targets.data).cpu().sum()    
            
        loss.backward()
        optimizer.step()
        
        tacc = tacc + accuracy.numpy() / (0.0 + targets.size(0))#loss.data.cpu().numpy()#accuracy
        count += 1

    # print ("training accuracy: ", tacc/(count+0.0)  )
    return tacc / (count + 0.0)
         
def eval(dh,num_batches):
    model.eval()
  
    tacc = 0
    count = 0
    while (True):  
        inputs, targets = dh.get_batch()
        inputs = inputs.transpose(1, 0, 2)
        inputs = torch.from_numpy(inputs).cuda()
        targets = torch.from_numpy(np.int64(targets)).cuda()
            
        output = model(inputs)
        output = output.detach()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        accuracy = pred.eq(targets.data).cpu().sum()        
        tacc += accuracy.numpy()
        count += 1
        if count == num_batches:
            break
  
    # print ("test accuracy: ", tacc / (count * targets.data.size(0) + 0.0)  )
  
    return tacc / (count * targets.data.size(0) + 0.0)

def test(dh):
    model.eval()

    tacc = 0

    inputs, targets = dh.get_all()
    inputs = inputs.transpose(1, 0, 2)
    inputs = torch.from_numpy(inputs).cuda()
    targets = torch.from_numpy(np.int64(targets)).cuda()
    
    output = model(inputs)
    output = output.detach()
    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum()        
    tacc = accuracy.numpy()
  
    return tacc / (targets.data.size(0) + 0.0)

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
      if 'recurrent_weight' in name:
        param.data.clamp_(-clip, clip)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     

print()
start_time = time.time()
acc = 0
prev_acc = 0
last_high_acc = 0
epoch_with_high_acc = 0
best_model = None
best_opti = None
patience_threshold = 10
patience = 0
num_epoch = args.epoch_count
for epochi in range(num_epoch):
    train(num_train_batches)
    acc = eval(dh_eval, num_eval_batches)

    print('>>>>>>>>>>>>>>>> Epoch {0} <<<<<<<<<<<<<<<<'.format(epochi + 1))
    print('Accuracy: ', acc)
    print()

    if (acc > last_high_acc):
        best_model = copy.deepcopy(model.state_dict())   
        best_opti = copy.deepcopy(optimizer.state_dict()) 

        prev_acc = last_high_acc
        last_high_acc = acc
        epoch_with_high_acc = epochi + 1

        print('====== CHANGE CURRENT HIGHEST ACCURACY ======')
        print('Previous Highest Accuracy  : {0}'.format(prev_acc))
        print('Current Highest Accuracy   : {0}'.format(last_high_acc))
        print('\n')

        patience = 0
    elif (patience > patience_threshold):
        model.load_state_dict(best_model)
        optimizer.load_state_dict(best_opti)
    else:
        patience += 1

model.load_state_dict(best_model)
optimizer.load_state_dict(best_opti)
  
elapsed = time.time() - start_time

test_acc = test(dh_test)

print('')
print('====== FINAL HIGHEST ACCURACY ======')
print('Epoch                        : {0}'.format(epoch_with_high_acc))
print('Test Accuracy                : {0}'.format(test_acc))
print('Training Time                : {0} seconds'.format(elapsed))
print('')   

save_name = 'indrnn_plainnet_drowsiness_model' 
with open(save_name, 'wb') as f:
    torch.save(model.state_dict(), f)

print('===>>> Training End <<<===')
 