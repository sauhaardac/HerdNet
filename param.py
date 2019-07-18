import argparse
import torch

params = {}

# Verbosity
params['print_interval'] = 10  # print every 10 episodes

# Network training hyperparams
params['lr']        = 0.0005
params['gamma']     = 0.98
params['lmbda']     = 0.95
params['eps_clip']  = 0.1
params['K_epoch']   = 3
params['T_horizon'] = 20

# Simulation Parameters
params['env_name'] = 'boids-v0'
params['ep_len'] = 250
params['render'] = False
params['num_birds'] = 6
params['num_agents'] = 1
params['num_dims'] = 2
params['plim'] = 1
params['vlim'] = 1
params['min_dist_constraint'] = 0.3
params['agentcolor'] = 'blue'
params['birdcolor'] = 'green'
params['r_comm'] = 1.
params['r_des'] = 0.8
params['kx'] = 2
params['kv'] = 2
params['lambda_a'] = 0.1
params['dt'] = 0.05
params['n'] = params['num_birds'] + params['num_agents']
params['a_u'] = 1.

# Training defaults
torch.set_default_tensor_type(torch.cuda.FloatTensor)
params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")