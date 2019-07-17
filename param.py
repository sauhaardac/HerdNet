import argparse
import torch

params = {}

# Network training hyperparams
params['lr']        = 0.0005
params['gamma']     = 0.98
params['lmbda']     = 0.95
params['eps_clip']  = 0.1
params['K_epoch']   = 3
params['T_horizon'] = 20
params['device'] = torch.device('cpu')
