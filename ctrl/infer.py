import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym_boids
from torch.distributions import Categorical
import sys
import datetime

program_start_time = datetime.datetime.now()
# num_birds = int(sys.argv[1])
save = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, inp):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(inp, 128)
        self.fc2   = nn.Linear(128, 256)
        self.fc_pi = nn.Linear(256,4)
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

    for i in range(10):# save in saves:
        model.load_state_dict(torch.load(save))
        s = env.reset()
        done = False
        for i in range(600):
            prob = model.pi(torch.from_numpy(s).float().to(device))
            m = Categorical(prob)
            a = m.sample().item()
            s_prime = env.step(a)
            env.render()
            s = s_prime

        score = 0

    env.close()

if __name__ == '__main__':
    main()
