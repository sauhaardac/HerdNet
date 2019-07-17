from ppo import PPO
import torch
import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import gym_boids
from torch.distributions import Categorical
import sys
import datetime

# Unit test
def train_cartpole(params):
    """
    Unit test the PPO with a simple CartPole example.
    
    Returns:
    bool:Test success (did the model learn anything)
    """

    env = gym.make('CartPole-v1')
    model = PPO(params, 4, out=2)
    score = 0.0
    print_interval = 20

    initial = 0

    for n_epi in range(101):
        s = env.reset()
        done = False
        save = 0
        while not done:
            for t in range(params['T_horizon']):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi == 0:
            save = score
            score = 0.0

        if n_epi == 30:
            env.close()
            return  score > save

def test_ppo():
    params = {}

    params['lr']        = 0.0005
    params['gamma']     = 0.98
    params['lmbda']     = 0.95
    params['eps_clip']  = 0.1
    params['K_epoch']   = 3
    params['T_horizon'] = 20

    params['device'] = torch.device('cpu')

    assert train_cartpole(params)
