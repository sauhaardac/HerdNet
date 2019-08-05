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
        prob (torch.FloatTensor): output of the network, raw softmax distribution
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

    x_transformed = x.copy()
    
    for agent_idx in range(params['n']):
        x_transformed[2 * agent_idx * params['num_dims'] :
                (2 * agent_idx + 1) * params['num_dims'] - 1] -= params['pd'][i, 0]

        x_transformed[2 * agent_idx * params['num_dims'] + 1 :
                (2 * agent_idx + 1) * params['num_dims']] -= params['pd'][i, 1]

        x_transformed[(2 * agent_idx + 1) * params['num_dims']:
                2 * (agent_idx + 1) * params['num_dims'] - 1] -= params['vd'][i, 0]

        x_transformed[(2 * agent_idx + 1) * params['num_dims'] + 1:
                2 * (agent_idx + 1) * params['num_dims']] -= params['vd'][i, 1]

    return x_transformed

def tm(x, i):
    a = x.copy()
    goal_i = params['num_birds']
    cur_i = params['num_birds'] + i

    a[2 * cur_i * 2: (2 * cur_i + 1) * 2] = x[2 * goal_i * 2: (2 * goal_i + 1) * 2]
    a[2 * goal_i * 2: (2 * goal_i + 1) * 2] = x[2 * cur_i * 2: (2 * cur_i + 1) * 2]

    return a

def infer(path):
    """ Trains an RL model.

    First initializes environment, logging, and machine learning model. Then iterates
    through epochs of infering and prints score intermittently.
    """

    util.set_xd()

    infer = {}
    infer['env'] = gym.make(params['env_name'])
    infer['env'].init(params)
    infer['model'] = PPO(params, infer['env'].observation_space.shape[0]).to(params['device'])
    infer['model'].load_state_dict(torch.load(path))

    x = transform_state(infer['env'].reset(), 0)

    for i in range(1, params['nt']):
        joined_prob, joined_u = [], []

        for j in range(params['num_agents']):
            prob, m, u = run_network(infer, tm(x, j))
            joined_prob.append(prob)
            joined_u.append(u)
        
        step, acc = infer['env'].step(joined_u)
        x_prime = transform_state(step, i)

    x = np.array(infer['env'].get_x())
    plot.plot_SS(x, params['T'])
    plot.plot_error(x, params['T'])

if __name__ == '__main__':
    infer('saves/notidiot/0.82-4820.save')
    plot.save_figs()
