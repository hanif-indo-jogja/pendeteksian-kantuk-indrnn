import argparse
import numpy as np


def train_opts(parser):
  parser.add_argument('--learning_rate', type=float, default=2e-4,help='learning_rate')
  parser.add_argument('--epoch_count', type=int, default=120, help='epoch_count')
  parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
  parser.add_argument('--seq_len', type=int, default=784)
  parser.add_argument('--hidden_size', type=int, default=512)
  #parser.add_argument('--use_weightdecay_nohiddenW', action='store_true', default=False)
  parser.add_argument('--decayfactor', type=float, default=1e-4,help='lr')
  parser.add_argument('--opti', type=str, default='adam')

  # parser.add_argument('--ini_in2hid', type=float, default=0.002)

  parser.add_argument('--constrain_U', action='store_true', default=False)
  parser.add_argument('--MAG', type=float, default=5.0)
  parser.add_argument('--U_bound', type=float, default=0.0)
  
  parser.add_argument('--dropout', type=float, default=0.1)

  parser.add_argument('--num_layers', type=int, default=6)