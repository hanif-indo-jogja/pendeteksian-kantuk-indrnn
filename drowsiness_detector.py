import numpy as np
import torch
import torch.nn as nn
import argparse
from indrnn import opts    

seed = 100
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    print("WARNING: CUDA not available")

class DrowsinessDetector():
    def __init__(self, seq_len, hidden_size, dropout, num_layers):
        parser = argparse.ArgumentParser(description='pytorch action')
        opts.train_opts(parser)
        args = parser.parse_args()

        args.seq_len = seq_len
        args.hidden_size = hidden_size
        args.dropout = dropout
        args.bn_location = 'bn_after'
        args.num_layers = num_layers
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

        device = torch.device("cuda")
        model.to(device)
        model.eval()

        self.model = model

    def detect(self, data):
        data = data.astype('float32')
        data = data.transpose(1,0,2)
        device = torch.device("cuda")
        data = torch.from_numpy(data).to(device)

        output = self.model(data)

        _, predicted = torch.max(output, 1)
        predicted = np.array([predicted[j].cpu().numpy() for j in range(len(predicted))])
        # print('Predicted Class: ', ' '.join('%5s' % predicted))

        return predicted

 






