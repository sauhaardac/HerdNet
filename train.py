from param import params
import gym_boids
import gym
from net.ppo import PPO
from torch.distributions import Categorical
import torch
import numpy as np
from util import Logger
import sys

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

def calculate_reward(x):
    """ Calculates the reward function i.e. how good is the current state

    Parameters:
        x (np.array): current state

    Returns:
        reward (float): how good the state is

    """

    sum_x = np.zeros(params['num_dims'])
    for i in range(params['num_birds']):
        sum_x += x[2 * i * params['num_dims'] : (2 * i + 1) * params['num_dims']]

    reward = (2 - np.linalg.norm(sum_x/params['num_birds'])) / 2

    return reward

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

    x_transformed = np.concatenate([x, np.zeros(3)])
    
    for agent_idx in range(params['n']):
        idx = i / (params['ep_len'] / (2 * np.pi))

        x_transformed[2 * agent_idx * params['num_dims'] :
                (2 * agent_idx + 1) * params['num_dims'] - 1] -= np.cos(idx)

        x_transformed[2 * agent_idx * params['num_dims'] + 1 :
                (2 * agent_idx + 1) * params['num_dims']] -= np.sin(idx)

    x_transformed[-1] = -np.sin(idx)
    x_transformed[-2] = np.cos(idx)
    x_transformed[-3] = -np.cos(idx)
    # x_transformed[-4] = 0  # uneccessary for now

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

    for i in range(1, params['ep_len']):
        prob, m, u = run_network(train, x)
        x_prime = transform_state(train['env'].step(u), i)
        reward = calculate_reward(x)  # custom reward function given state
        train['model'].put_data((x, u, reward, x_prime, prob[u].item(), False))
        episode_score += reward
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
    train['model'] = PPO(params, 3 + train['env'].observation_space.shape[0]).to(params['device'])

    if params['transfer']:
        train['model'].load_state_dict(torch.load(sys.argv[1]))

    logger = Logger()

    score = 0.0

    for n_epi in range(10**6):
        ep_score = episode(train)
        logger.episode_score(ep_score, n_epi)
        score += ep_score

        if n_epi % params['print_interval'] == 0 and n_epi != 0:
            print(f"Episode #{n_epi:5d} | Avg Score : {score / params['print_interval']:2.2f}")

            if n_epi >= 0:
                logger.save_model(score/params['print_interval'], train['model'].state_dict(), n_epi)

            score = 0.0

        train['model'].train_net()

    env.close()

if __name__ == '__main__':
    train()
