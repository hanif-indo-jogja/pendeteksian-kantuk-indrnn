import numpy as np
import os
import time
import json
import torch
import argparse
from indrnn import opts

from drowsiness_detector import DrowsinessDetector

class MeanStd():
    def __init__(self):
        self.freq_mean      = 0
        self.freq_stddev    = 0
        self.amp_mean       = 0
        self.amp_stddev     = 0
        self.dur_mean       = 0
        self.dur_stddev     = 0
        self.vel_mean       = 0
        self.vel_stddev     = 0

def get_train_mean_std():
    json_mean_std = []
    with open('train_mean_std.json', 'r') as f_handle:
        json_mean_std = json.loads(f_handle.read())

    train_mean_std = MeanStd()

    train_mean_std.freq_mean    = np.float32(json_mean_std['freq_mean'])
    train_mean_std.freq_stddev  = np.float32(json_mean_std['freq_stddev'])

    train_mean_std.amp_mean     = np.float32(json_mean_std['amp_mean'])
    train_mean_std.amp_stddev   = np.float32(json_mean_std['amp_stddev'])

    train_mean_std.dur_mean     = np.float32(json_mean_std['dur_mean'])
    train_mean_std.dur_stddev   = np.float32(json_mean_std['dur_stddev'])

    train_mean_std.vel_mean     = np.float32(json_mean_std['vel_mean'])
    train_mean_std.vel_stddev   = np.float32(json_mean_std['vel_stddev'])

    return train_mean_std

def z_score_normalization(blinks, mean_std):
    blinks[:,:,0] = ((blinks[:,:,0].astype('float32') - np.float32(mean_std.freq_mean)) /
        np.float32(mean_std.freq_stddev))

    blinks[:,:,1] = ((blinks[:,:,1] - mean_std.amp_mean) /
        mean_std.amp_stddev)

    blinks[:,:,2] = ((blinks[:,:,2] - mean_std.dur_mean) /
        mean_std.dur_stddev)

    blinks[:,:,3] = ((blinks[:,:,3] - mean_std.vel_mean) /
        mean_std.vel_stddev)

def do_test(blinks, labels):
    parser = argparse.ArgumentParser(description='pytorch action')
    opts.train_opts(parser)
    args = parser.parse_args()

    args.seq_len = 30
    args.hidden_size = 512
    args.dropout = 0.5
    args.bn_location = 'bn_after'
    args.num_layers = 2
    args.u_lastlayer_ini = True

    outputclass = 2
    indim = 4

    if args.U_bound == 0:
        U_bound = np.power(10, (np.log10(args.MAG) / args.seq_len))   
    else:
        U_bound = args.U_bound

    import indrnn.Indrnn_plainnet as Indrnn_network
    Indrnn_network.init(anArgs = args, aU_bound = U_bound)

    model = Indrnn_network.IndrnnPlainnet(indim, outputclass)

    model.load_state_dict(torch.load('indrnn_plainnet_drowsiness_model'))

    model.cuda()
    model.eval()

    tacc = 0

    inputs, targets = blinks, labels
    inputs = inputs.transpose(1, 0, 2)
    inputs = torch.from_numpy(inputs).cuda()
    targets = torch.from_numpy(np.int64(targets)).cuda()
    
    output = model(inputs)
    output = output.detach()
    pred = output.data.max(1)[1]
    accuracy = pred.eq(targets.data).cpu().sum()        
    tacc = accuracy.numpy()
  
    return tacc / (targets.data.size(0) + 0.0)

def test_drowsiness_detector_accuracy(blinks_path, labels_path):
    blinks = np.load(blinks_path).astype('float32')
    labels = np.load(labels_path)

    train_mean_std = get_train_mean_std()

    z_score_normalization(blinks, train_mean_std)

    return do_test(blinks, labels)


##########
## MAIN ##
##########

blinks_path = 'output/preprocessed_blink_features/BlinksTest_w30_s2.npy'
labels_path = 'output/preprocessed_blink_features/LabelsTest_w30_s2.npy'

start_time = time.time()
accuracy = test_drowsiness_detector_accuracy(blinks_path, labels_path)

elapsed = time.time() - start_time

print('================ Total Accuracy ===============')
print('total accuracy   : {0}%'.format(accuracy))
print('Test Time        : {0} seconds'.format(elapsed))
print()