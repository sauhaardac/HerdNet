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


class PPO(nn.Module):
    def __init__(self, params, inp, out=4):
        super(PPO, self).__init__()
        self.params = params
        self.data = []
        self.__net_architecture(inp, out)
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['lr'])

    def __net_architecture(self, inp, out):
        self.fc1   = nn.Linear(inp, 128)
        self.fc2   = nn.Linear(128, 256)
        self.fc_pi = nn.Linear(256, out)
        self.fc_v  = nn.Linear(256,1)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.params['device']), torch.tensor(a_lst).to(self.params['device']), \
                                          torch.tensor(r_lst).to(self.params['device']), torch.tensor(s_prime_lst, dtype=torch.float).to(self.params['device']), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.params['device']), torch.tensor(prob_a_lst).to(self.params['device'])
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.params['K_epoch']):
            td_target = r + self.params['gamma'] * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.params['gamma'] * self.params['lmbda'] * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.params['device'])

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
