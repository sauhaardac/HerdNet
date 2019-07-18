from param import params
import gym_boids
import gym
from net.ppo import PPO
from torch.distributions import Categorical
import torch
import numpy as np

def run_network(train, x):
    prob = train['model'].pi(torch.from_numpy(x).float().to(params['device']))
    m = Categorical(prob)
    u = m.sample().item()
    return prob, m, u

def calculate_reward(x):
    sum_x = np.zeros(params['num_dims'])
    for i in range(params['num_birds']):
        sum_x += x[2 * i * params['num_dims'] : (2 * i + 1) * params['num_dims']]

    reward = (2 - np.linalg.norm(sum_x/params['num_birds'])) / 2

    return reward

def episode(train):
    episode_score = 0
    x = train['env'].reset()

    for i in range(params['ep_len']):
        prob, m, u = run_network(train, x)
        x_prime = train['env'].step(u)  # propagate step through environment
        reward = calculate_reward(x)  # custom reward function given state
        train['model'].put_data((x, u, reward, x_prime, prob[u].item(), False))
        episode_score += reward
        x = x_prime

    return episode_score / params['ep_len']

def main():
    train = {}
    train['env'] = gym.make(params['env_name'])
    train['env'].init(params)
    train['model'] = PPO(params, train['env'].observation_space.shape[0]).to(params['device'])

    score = 0.0

    for n_epi in range(10**6):
        score += episode(train)

        torch.save(model.state_dict(), f"saves/{score/print_interval}-{n_epi}.save")

        if n_epi % params['print_interval'] == 0 and n_epi != 0:
            print(f"Episode #{n_epi:5d} | Avg Score : {score / params['print_interval']:2.2f}")
            score = 0.0

        train['model'].train_net()

    env.close()

if __name__ == '__main__':
    main()
