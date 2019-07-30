from param import params
import gym_boids
import gym
from net.ppo import PPO
from torch.distributions import Categorical
import torch
import numpy as np
import util
import sys
import plot


def run_network(infer, x):
    """ Runs the network to determine action given current state

    Parameters:
        infer (dict): dictionary of infering variables
        x (np.array): current state to input into neural network

    Returns:
        prob (torch.FloatTensor): output of the network
        m (torch.FloatTensor): categorical distribution of output
        u (torch.FloatTensor): actual action selected by the network

    """

    prob = infer['model'].pi(torch.from_numpy(x).float().to(params['device']))
    m = Categorical(prob)
    u = m.sample().item()
    return prob, m, u


def transform_state(x, i):
    """ Transforms the state to make the infering set more varied.

    Shifts the position state in a circle so the agents are forced to
    track a point rather than simply move towards a goal point. This
    is a harder problem to learn.

    Params:
        x (np.array): current state
        i (int): iteration # within the episode

    Returns:
        x_transformed (np.array): augmented/transformed state

    """

    return x


def infer(path, label):
    """ Trains an RL model.

    First initializes environment, logging, and machine learning model. Then iterates
    through epochs of infering and prints score intermittently.
    """

    util.set_xd()

    infer = {}
    infer['env'] = gym.make(params['env_name'])
    infer['env'].init(params)
    infer['model'] = PPO(
        params, infer['env'].observation_space.shape[0]).to(params['device'])
    infer['model'].load_state_dict(torch.load(path))

    x = transform_state(infer['env'].reset(), 0)

    for i in range(1, params['nt']):
        prob, m, u = run_network(infer, x)
        state, _ = infer['env'].step(u)
        x_prime = transform_state(state, i)
        x = x_prime

    x = np.array(infer['env'].get_x())
    plot.plot_SS(x, params['T'], title=f"State Space after {label}")


if __name__ == '__main__':
    paths = ['0.76-2720.save']
    for i in range(len(paths)):
        infer(f'saves/maxr/{paths[i]}', 'Training')
    plot.save_figs()
