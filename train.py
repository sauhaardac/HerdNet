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


def calculate_reward(x, acc):
    """Calculates the reward function i.e. how good is the current state.

    Parameters:
        x (np.array): current state
        acc (np.array): current centroid acceleration

    Returns:
        reward (float): how good the state is

    """
    coordinates = [(0,0),
                   (np.cos(-np.pi/6), np.sin(-np.pi/6)),
                   (np.cos(-5*np.pi/6), np.sin(-5*np.pi/6)),
                   (np.cos(np.pi/2), np.sin(np.pi/2))]

    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-2, 2, 0.05)

    X, Y = np.meshgrid(X, Y)
    Z = X*0
    Z_des = X*0

    for ax, ay in coordinates:
        ay -= np.sqrt(3)/2
        Z_des = np.maximum(Z_des, np.exp(-((X-ax)**2 / 1 + (Y-ay)**2 / 1)))

    for i in range(params['num_agents']):
        ax, ay = util.get_p(x, i)
        Z = np.maximum(Z, np.exp(-((X-ax)**2 / 1 + (Y-ay)**2 / 1)))

    return 2.6 - np.linalg.norm(Z - Z_des) / 15


def episode(train, ep_num):
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

        reward = calculate_reward(x, acc)  # custom reward function given state
        train['model'].put_data((x, u, reward, x_prime, prob[u].item(), False))
        episode_score += reward
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


def train():
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

    logger = Logger()

    score = 0.0

    for n_epi in range(10**6):
        ep_score = episode(train, n_epi)
        logger.episode_score(ep_score, n_epi)
        score += ep_score

        if n_epi % params['print_interval'] == 0 and n_epi != 0:
            print(f"Episode #{n_epi:5d} | Avg Score :" +
                  f"{score / params['print_interval']:2.2f}")

            if n_epi >= 0:
                logger.save_model(
                    score / params['print_interval'],
                    train['model'].state_dict(),
                    n_epi)

            score = 0.0

        train['model'].train_net()

    env.close()


if __name__ == '__main__':
    train()
