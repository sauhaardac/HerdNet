from param import params
import gym_boids
import gym
from net.ppo import PPO
from torch.distributions import Categorical
import torch
import numpy as np
from util import Logger
import sys

params['render'] = True

def run_network(train, x):
    """ Runs the network to determine action given current state

    Parameters:
        train (dict): dictionary of training variables
        x (np.array): current state to input into neural network

    Returns:
        prob (torch.FloatTensor): output of the network, raw softmax distribution
        m (torch.FloatTensor): categorical distribution of output
        u (torch.FloatTensor): actual action selected by the network

    """

    prob = train['model'].pi(torch.from_numpy(x).float().to(params['device']))
    m = Categorical(prob)
    u = m.sample().item()
    return prob, m, u

def transform_state(x, i):
    """ Transforms the state to make the training set more varied.

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
        idx = i / (params['ep_len'] / (2 * np.pi))

        x_transformed[2 * agent_idx * params['num_dims'] :
                (2 * agent_idx + 1) * params['num_dims'] - 1] -= np.cos(idx)

        x_transformed[2 * agent_idx * params['num_dims'] + 1 :
                (2 * agent_idx + 1) * params['num_dims']] -= np.sin(idx)

        x_transformed[(2 * agent_idx + 1) * params['num_dims']:
                2 * (agent_idx + 1) * params['num_dims'] - 1] -= -np.sin(idx)

        x_transformed[(2 * agent_idx + 1) * params['num_dims'] + 1:
                2 * (agent_idx + 1) * params['num_dims']] -= np.cos(idx)

    return x_transformed

def episode(train):
    """ Runs one episode of training

    Parameters:
        train (dict): dictionary of training variables

    Returns:
        episode_score (float): average reward during the episode
    """
        
    episode_score = 0
    x = transform_state(train['env'].reset(), 0)

    trajx, trajy, centroidx, centroidy = [], [], [], []

    for i in range(1, params['ep_len']):
        prob, m, u = run_network(train, x)
        state, = train['env'].step(u)
        x_prime = transform_state(state, i)
        episode_score += calculate_reward(x_prime)

        trajx.append(np.cos(i / (params['ep_len']/(2 * np.pi))))
        trajy.append(np.sin(i / (params['ep_len']/(2 * np.pi))))

        x, y = [], []
        for agent_idx in range(params['num_birds']):
            x.append(x_prime[2 * agent_idx * params['num_dims'] : (2 * agent_idx + 1) * params['num_dims'] - 1])
            y.append(x_prime[2 * agent_idx * params['num_dims'] + 1 : (2 * agent_idx + 1) * params['num_dims']])

        centroidx.append(np.mean(x))
        centroidy.append(np.mean(y))

        train['env'].render((trajx[-200:], trajy[-200:]), (centroidx[-200:], centroidy[-200:]))

        x = x_prime

    episode_score /= params['ep_len']

    return episode_score

def train():
    """ Trains an RL model.

    First initializes environment, logging, and machine learning model. Then iterates
    through epochs of training and prints score intermittently.
    """

    train = {}
    train['env'] = gym.make(params['env_name'])
    train['env'].init(params)
    train['model'] = PPO(params, train['env'].observation_space.shape[0]).to(params['device'])
    train['model'].load_state_dict(torch.load(sys.argv[1]))

    # logger = Logger()

    score = 0.0

    for n_epi in range(10**6):
        ep_score = episode(train)
        # logger.episode_score(ep_score, n_epi)
        score += ep_score

        print(f"Episode #{n_epi:5d} | Avg Score : {score / params['print_interval']:2.2f}")
        score = 0.0

    env.close()

if __name__ == '__main__':
    train()
