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


def calculate_reward(x, acc, kr, pr, std):
    """Calculates the reward function i.e. how good is the current state.

    Parameters:
        x (np.array): current state
        acc (np.array): current centroid acceleration

    Returns:
        reward (float): how good the state is

    """
    coordinates = [(1,1)]

    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-2, 2, 0.05)

    X, Y = np.meshgrid(X, Y)
    Z = X*0
    Z_des = X*0

    for ax, ay in coordinates:
        ay -= np.sqrt(3)/2
        Z_des = np.maximum(Z_des, np.exp(-((X-ax)**2 / std + (Y-ay)**2 / std)))

    for i in range(params['num_agents']):
        ax, ay = util.get_p(x, i)
        Z = np.maximum(Z, np.exp(-((X-ax)**2 / std + (Y-ay)**2 / std)))

    return 1 - np.power(np.linalg.norm(Z - Z_des), pr) * kr, np.linalg.norm(util.get_p(x, 0) - np.array([1, 1]))


def episode(train, ep_num, kr, pr, std):
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

        reward, goodness = calculate_reward(x, acc, kr, pr, std)  # custom reward function given state
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


def train(args, log=False, num_eps=100):
    """ Trains an RL model.

    First initializes environment, logging, and machine learning model,
    then iterates through epochs of training and prints the score
    intermittently.
    """

    kr, pr, std = args
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
        ep_score = episode(train, n_epi, kr, pr, std)
        std *= 0.99
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
    # from hyperopt import hp, tpe, fmin
    # space = [hp.uniform('kr', 1/24, 1), hp.uniform('pr', 1/3, 1), hp.uniform('std', 0.5, 2)]
    # best = fmin(train, space, algo=tpe.suggest, max_evals=40)
    # print(best)
    best = {'kr': 0.21183340208383372, 'pr': 0.49020013438473126, 'std': 1.289503679565993}
    # best = {'kr': 0.7179338426650235, 'pr': 0.9504521969659028, 'std': 1.9015236564373323}
    # best = {'kr': 0.9005977992525029, 'pr': 0.4791564901464124, 'std': 1.1123498688314886}
    train((best['kr'], best['pr'], best['std']), log=True, num_eps=5000)
