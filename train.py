"""Trains a RL Model to do Leader-Follower Maneuvers."""

from param import params
import gym_boids
import gym
from net.ppo import PPO
from torch.distributions import Categorical
import torch
import numpy as np
from util import Logger
import util
import sys
import random


def run_network(train, x):
    """Run the network to determine action given current state.

    Parameters:
        train (dict): dictionary of training variables
        x (np.array): current state to input into neural network

    Returns:
        (tuple): tuple containing:

            prob (torch.FloatTensor): output of the network
            m (torch.FloatTensor): categorical distribution of output
            u (torch.FloatTensor): actual action selected by the network

    """
    prob = train['model'].pi(torch.from_numpy(x).float().to(
        params['device']))

    m = Categorical(prob)
    u = m.sample().item()
    return prob, m, u


def calculate_reward(x, acc, hyper):
    """Calculates the reward function i.e. how good is the current state.

    Parameters:
        x (np.array): current state
        acc (np.array): current centroid acceleration

    Returns:
        reward (float): how good the state is

    """
    coordinates = [np.array([1, 1]), np.array([-1, -1])]

    avg_dist = 0
    avg_vel = 0
    for coordinate in coordinates:
        min_dist = 1000000000000
        min_vel = 1000000000000
        for i in range(params['num_agents']):
            dist = np.linalg.norm(util.get_p(x, i) - coordinate) 
            if dist < min_dist:
                min_dist = dist
                min_vel = np.linalg.norm(util.get_v(x, i))
        avg_dist += min_dist / params['num_agents']
        avg_vel += min_dist / params['num_agents']

    return 1 - np.power(hyper['coeff'] * avg_dist + hyper['coeff2'] * avg_vel, hyper['power']) / hyper['div'], avg_dist


def episode(train, hyper):
    """ Runs one episode of training

    Parameters:
        train (dict): dictionary of training variables

    Returns:
        episode_score (float): average reward during the episode

    """
    episode_score = 0

    my_i = np.random.uniform(low=0.0, high=2 * np.pi)
    goal = [np.cos(my_i), np.sin(my_i)]

    x = transform_state(train['env'].reset(), goal)

    for i in range(1, params['ep_len']):
        prob, m, u = run_network(train, x)
        step, acc = train['env'].step(u)
        x_prime = transform_state(step, goal)

        reward, goodness = calculate_reward(x, acc, hyper)  # custom reward function given state
        train['model'].put_data((x, u, reward, x_prime, prob[u].item(), False))
        episode_score += goodness
        x = x_prime

    episode_score /= params['ep_len']

    return episode_score


def transform_state(x, goal):
    """Transforms the state to make the training set more varied.

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


def train(hyper, log=False, num_eps=100):
    """ Trains an RL model.

    First initializes environment, logging, and machine learning model,
    then iterates through epochs of training and prints the score
    intermittently.
    """

    train = {}
    train['env'] = gym.make(params['env_name'])
    train['env'].init(params)
    train['model'] = PPO(
        params,
        train['env'].observation_space.shape[0], out=4**params['num_agents']).to(
        params['device'])

    if params['transfer']:
        train['model'].load_state_dict(torch.load(sys.argv[1]))

    if log:
        logger = Logger()

    score = 0.0
    ep_score = 0
    last_score = 10000 

    for n_epi in range(num_eps):
        ep_score = episode(train, hyper)
        score += ep_score

        if log:
            logger.episode_score(ep_score, n_epi)
            if n_epi % params['print_interval'] == 0 and n_epi != 0:
                print(f"Episode #{n_epi:5d} | Avg Score :" +
                      f"{score / params['print_interval']:2.2f}")

                if n_epi >= 0:
                    logger.save_model(
                        score / params['print_interval'],
                        train['model'].state_dict(),
                        n_epi)

                score = 0.0
        elif n_epi % 10 == 0 and n_epi != 0:
            last_score = score / 10
            score = 0.0

        train['model'].train_net()

    train['env'].close()
    return last_score


if __name__ == '__main__':
    from hyperopt import hp, tpe, fmin
    space = [hp.uniform('power', 1/24, 1), hp.uniform('div', 1, 10), hp.uniform('coeff', 0, 1), hp.uniform('coeff2', 0, 1),]
    best = fmin(train, {'power' : space[0], 'div' : space[1], 'coeff': space[2], 'coeff2' : space[3]}, algo=tpe.suggest, max_evals=40)
    # best = {'div': 1.5425666248304248, 'power': 0.3526938268834692}
    train(best, log=True, num_eps=5000)
