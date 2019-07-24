import argparse
import torch
import numpy as np
import warnings
# warnings.filterwarnings("ignore")

np.seterr(all='raise')

params = {}

# Verbosity
params['print_interval'] = 10  # print every 10 episodes

# Network training hyperparams
params['transfer']  = False
params['lr']        = 0.0005
params['gamma']     = 0.98
params['lmbda']     = 0.95
params['eps_clip']  = 0.1
params['K_epoch']   = 3
params['T_horizon'] = 20

# Time
params['t0'] = 0.
params['tf'] = 20.
params['dt'] = 0.05
T = np.arange( params['t0'], params['tf'], params['dt'])
params['T'] = np.reshape(T, (len(T),-1))
params['nt'] = len(params['T'])

# Plot Parameters
params['render'] = False
params['agentcolor'] = 'r'
params['birdcolor'] = 'b'
params['stop_marker'] = 'x'
params['start_marker'] = 's'
params['pdcolor'] = 'g'
params['centroidcolor'] = 'k'
params['fn_plots'] = 'plots.pdf'

# Simulation Parameters
params['env_name'] = 'boids-v0'
params['ep_len'] = 250
params['num_birds'] = 6
params['num_agents'] = 1
params['num_dims'] = 2
params['plim'] = 1
params['vlim'] = 1
params['min_dist_constraint'] = 0.3
params['r_comm'] = 1.
params['r_des'] = 0.8
params['kx'] = 2
params['kv'] = 2
params['lambda_a'] = 0.1
params['n'] = params['num_birds'] + params['num_agents']
params['a_u'] = 1.

# Training defaults
torch.set_default_tensor_type(torch.cuda.FloatTensor)
params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
