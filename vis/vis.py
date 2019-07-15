import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm 
from numpy.random import rand
from numpy.linalg import norm
from math import exp
import pickle
import sys
import random

def get_p(param, x, agent_idx):
    """Returns the position of the specified agent given the state vector"""

    return x[2 * agent_idx * param['num_dims'] : (2 * agent_idx + 1) * param['num_dims']]

def get_v(param, x, agent_idx):
    """Returns the velocity of the specified agent given the state vector"""

    return x[(2 * agent_idx + 1) * param['num_dims']: 2 * (agent_idx + 1) * param['num_dims']]


def init_plot(param):
    """Initializes the Matplotlib visualization, necessary for visualization"""

    plt.ion()
    fig = plt.figure(figsize=(10, 10), dpi=80)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    boidplot = plt.quiver([], [], [], [], color=param['birdcolor'])
    agentplot = plt.quiver([], [], [], [], color=param['agentcolor'])

    return {'fig' : fig, 'boidplot' : boidplot, 'agentplot' : agentplot}

def plot_boids(param, x, plot):
    """Update the plot with the current state given a pre-initialized plot"""

    fig = plot['fig']
    boidplot = plot['boidplot']
    agentplot = plot['agentplot']

    plotx = []
    ploty = []
    plotu = []
    plotv = []

    for i in range(param['num_birds']):
        p_i = get_p(param, x, i)
        plotx.append(p_i[0])
        ploty.append(p_i[1])

        v_i = get_v(param, x, i) / 100
        plotu.append(v_i[0])
        plotv.append(v_i[1])

    
    boidplot.set_offsets(np.array([plotx, ploty]).T)
    boidplot.set_UVC(plotu, plotv)

    plotx = []
    ploty = []
    plotu = []
    plotv = []
    for i in range(param['num_agents']):
        p_i = get_p(param, x, i + param['num_birds'])
        plotx.append(p_i[0])
        ploty.append(p_i[1])

        v_i = get_v(param, x, i + param['num_birds']) / 100
        plotu.append(v_i[0])
        plotv.append(v_i[1])

    agentplot.set_offsets(np.array([plotx, ploty]).T)
    agentplot.set_UVC(plotu, plotv)

    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == '__main__':
    param = {'render' : True, 'num_birds' : 6, 'num_agents' : 1, 'num_dims' : 2, 'plim' : 1, 'vlim' : 1, 'min_dist_constraint' : 0.3,
            'agentcolor' : 'blue', 'birdcolor' : 'green', 'r_comm' : 1., 'r_des' : 0.8, 'kx' : 2, 'kv' : 2,
            'lambda_a' : 0.1, 'dt' : 0.05}

    param['n'] = param['num_birds'] + param['num_agents']
    param['a_u'] = 1.

    plot = init_plot(param)

    states = pickle.load(open('logs/episode-1571.pkl', 'rb'))

    import time
    for x in states:
        plot_boids(param, x, plot)
