from tensorboardX import SummaryWriter
import os
import torch
from param import params
import numpy as np

def get_P(X, i):
    """ Get position of agent i for all time

    Parameters:
        X (np.array): state over all time numpy array
        i (int): Agent # to return
    """

    state_idx = 2 * params['num_dims'] * i
    return X[:, state_idx : state_idx + params['num_dims']]

def get_P_bar(X):
    """ Get centroid position

    Parameters:
        X (np.array): state over all time numpy array
    """

    p_bar = np.zeros([params['nt'], params['num_dims']])

    for i in range(params['num_birds']):
        p_bar += get_P(X, i)

    return p_bar / params['num_birds']

def set_xd():
    params['pd'] = np.zeros((len(params.get('T')), 2))
    params['pd'][:,0] = np.cos( 2*np.pi*np.squeeze(params['T'])/params['T'][-1])
    params['pd'][:,1] = np.sin( 2*np.pi*np.squeeze(params['T'])/params['T'][-1])
    params['vd'] = np.gradient(params.get('pd'), params.get('dt'), axis = 0)
    params['ad'] = np.gradient(params.get('vd'), params.get('dt'), axis = 0)
    params['jd'] = np.gradient(params.get('ad'), params.get('dt'), axis = 0)

class Logger:
    def __init__(self):
        self.run_name = input('Run/Test Name (Descriptive): ')
        self.writer = SummaryWriter(f"logs/{self.run_name}")
        if not os.path.isdir(f"saves/{self.run_name}"):
            os.mkdir(f"saves/{self.run_name}")

    def episode_score(self, score, ep):
        # add 20 to make the score always positive!
        self.writer.add_scalar('data/epscore', score, ep)

    def save_model(self, score, weights, ep):
        torch.save(weights, f"saves/{self.run_name}/{score:2.2f}-{ep}.save")
