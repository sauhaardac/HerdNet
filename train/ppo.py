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
