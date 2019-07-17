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

torch.set_default_tensor_type(torch.cuda.FloatTensor)

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, inp):
        super(PPO, self).__init__()
        self.data = []
        
        # self.c1 = nn.Conv2d(3, 128, 3, stride=2, padding=1)
        # self.p1 = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Linear(inp, 128)
        self.fc2   = nn.Linear(128, 256)
        self.fc_pi = nn.Linear(256,4)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
                                          torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(device), torch.tensor(prob_a_lst).to(device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('boids-v0')

    param = {'render' : True, 'num_birds' : 6, 'num_agents' : 1, 'num_dims' : 2, 'plim' : 1, 'vlim' : 1, 'min_dist_constraint' : 0.3,
            'agentcolor' : 'blue', 'birdcolor' : 'green', 'r_comm' : 1., 'r_des' : 0.8, 'kx' : 2, 'kv' : 2,
            'lambda_a' : 0.1, 'dt' : 0.05}

    param['n'] = param['num_birds'] + param['num_agents']
    param['a_u'] = 1.
    param['ep_len'] = 250

    env.init(param)

    model = PPO(env.observation_space.shape[0]).to(device)
    score = 0.0
    epi_score = 0.0
    max_epi_score = 0.0

    print_interval = 10

    for n_epi in range(10000):
        done = False
        x = env.reset()
        for i in range(param['ep_len']):
            prob = model.pi(torch.from_numpy(x).float().to(device))
            m = Categorical(prob)
            u = m.sample().item()

            x_prime = env.step(u)

            if(n_epi > 1000):
                env.render()

            sum_x = np.zeros(param['num_dims'])
            for i in range(param['num_birds']):
                sum_x += x_prime[2 * i * param['num_dims'] : (2 * i + 1) * param['num_dims']]

            reward = (2 - np.linalg.norm(sum_x/param['num_birds'])) / 2

            score += reward
            
            model.put_data((x, u, reward, x_prime, prob[u].item(), False))

            x = x_prime

            if done:
                break

        if n_epi%print_interval==0 and n_epi!=0:
            torch.save(model.state_dict(), f"saves/{score/print_interval}-{n_epi}.save")
            print("# of episode :{}, avg score : {:.3f}".format(n_epi, score/(print_interval * param['ep_len'])))
            env.save_epi(f"logs/episode-{n_epi}-score-{score/(print_interval * param['ep_len'])}.pkl")
            score = 0.0

        model.train_net()

    env.close()

if __name__ == '__main__':
    main()
