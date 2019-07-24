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

class BoidsEnv(gym.Env):
    def __init__(self):
        """Empty constructor function to satisfy gym requirements"""

        pass

    def init(self, param):
        """Real constructor with params input"""

        self.param = param 
        self.x = [init_state(param)]
        self.observation_space = spaces.Box(-5, 5, self.x[-1].shape)

        if param['render']:
            self.plot = init_tracking_plot(param)

    def step(self, u):
        """Simulate step in environment"""

        dxdt = get_f(self.param, self.x[-1]) + get_g(self.param, self.x[-1], u)

        sum_a = np.zeros(self.param['num_dims'])
        for i in range(self.param['num_birds']):
            sum_a += dxdt[(2 * i + 1) * self.param['num_dims'] : 2 * (i + 1) * self.param['num_dims']]

        sum_a /= self.param['num_birds'] / self.param['dt']

        self.x.append(self.x[-1] + dxdt * self.param['dt'])

        return self.x[-1], sum_a

    def get_x(self):
        return self.x

    def save_epi(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.x, handle)

    def reset(self):
        """Reset enviornment"""

        self.x = [init_state(self.param)]
        return self.x[-1]
    
    def render(self, goal = None, centroid = None):
        """Render enviornment"""

        if(goal is None or centroid is None):
            plot_boids(self.param, self.x[-1], self.plot)
        else:
            plot_tracking_boids(self.param, self.x[-1], self.plot, goal, centroid)

####################
# HELPER FUNCTIONS #
####################

def get_p(param, x, agent_idx):
    """Returns the position of the specified agent given the state vector"""

    return x[2 * agent_idx * param['num_dims'] : (2 * agent_idx + 1) * param['num_dims']]

def get_v(param, x, agent_idx):
    """Returns the velocity of the specified agent given the state vector"""

    return x[(2 * agent_idx + 1) * param['num_dims']: 2 * (agent_idx + 1) * param['num_dims']]

def get_min_dist(param, x):
    """Returns the minimum distance between any pair of agents in the given state vector"""

    min_dist = sys.maxsize
    for i in range(param['num_agents'] + param['num_birds']):
        p_i = get_p(param, x, i)
        for j in range(i+1, param['num_agents'] + param['num_birds']):
            p_j = get_p(param, x, j)
            dist = norm(p_j - p_i)
            if dist < min_dist:
                min_dist = dist;
    return min_dist

def init_state(param):
    """Initializes the state of the system"""

    x0 = np.zeros(2 * param['n'] * param['num_dims'])

    min_dist = 0
    while min_dist < param['min_dist_constraint']:
        for i in range(param['num_birds']):
            p_i = rand(param['num_dims']) * param['plim'] - param['plim'] / 2
            v_i = rand(param['num_dims']) * param['vlim'] - param['vlim'] / 2

            idx = i * 2 * param['num_dims']
            x0[idx : idx + 2 * param['num_dims']] = np.concatenate([p_i, v_i])

        for i in range(param['num_agents']):
            p_i = rand(param['num_dims']) * param['plim'] - param['plim'] / 2
            v_i = rand(param['num_dims']) * param['vlim'] - param['vlim'] / 2

            idx = (i + param['num_birds']) * 2 * param['num_dims']
            x0[idx : idx + 2 * param['num_dims']] = np.concatenate([p_i, v_i])

        min_dist = get_min_dist(param, x0)

    return x0

def init_plot(param):
    """Initializes the Matplotlib visualization, necessary for visualization"""

    plt.ion()
    fig = plt.figure(figsize=(10, 10), dpi=80)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    boidplot = plt.quiver([], [], [], [], color=param['birdcolor']);
    agentplot = plt.quiver([], [], [], [], color=param['agentcolor']);
    theta = np.linspace(-np.pi, np.pi, 200)
    circleplot, = plt.plot(np.sin(theta), np.cos(theta))

    return {'fig' : fig, 'boidplot' : boidplot, 'agentplot' : agentplot, 'circleplot' : circleplot}

def init_tracking_plot(param):
    """Initializes the Matplotlib visualization, necessary for visualization"""

    plt.ion()
    fig = plt.figure(figsize=(10, 10), dpi=80)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    boidplot = plt.quiver([], [], [], [], color=param['birdcolor'])
    agentplot = plt.quiver([], [], [], [], color=param['agentcolor'])
    circleplot, = plt.plot([], [])

    centroidtraj, = plt.plot([], [])
    goaltraj, = plt.plot([], [])

    return {'fig' : fig, 'boidplot' : boidplot, 'agentplot' : agentplot,
            'circleplot' : circleplot, 'centroidtraj' : centroidtraj, 'goaltraj' : goaltraj}

def plot_tracking_boids(param, x, plot, goal, centroid):
    """Update the plot with the current state given a pre-initialized plot"""

    fig = plot['fig']
    boidplot = plot['boidplot']
    agentplot = plot['agentplot']
    # circleplot = plot['circleplot']
    centroidtraj = plot['centroidtraj']
    goaltraj = plot['goaltraj']

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

    centroidtraj.set_data(*centroid)
    goaltraj.set_data(*goal)

    fig.canvas.draw()
    fig.canvas.flush_events()

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

def get_adjacency(param, x):
    """Get the adjacency matrix between agents given the state vector"""

    A = np.zeros([param['n'], param['n']])
    for i in range(param['n']):
        p_i = get_p(param, x, i)
        for j in range(i, param['n']):
            p_j = get_p(param, x, j)
            dist = norm(p_j - p_i)
            if dist < param['r_comm']:
                A[i, j] = exp(-param['lambda_a'] * dist)
            else:
                A[i, j] = 0
            A[j, i] = A[i, j]
    return A


def get_f(param, x):
    """Get f(x), the drift dynamics of the system with no control input"""

    A = get_adjacency(param, x)

    f = np.zeros(2 * param['n'] * param['num_dims'])

    for i in range(param['num_birds']):
        p_i = get_p(param, x, i)
        v_i = get_v(param, x, i)
        a_i = np.zeros(param['num_dims'])

        for j in range(param['n']):
            if i == j:
                continue

            p_j = get_p(param, x, j)
            v_j = get_v(param, x, j)
            r_ij = p_j - p_i

            a_i += A[i, j] * (param['kv'] * (v_j - v_i))
            a_i += param['kx'] * r_ij * (1 - param['r_des']/norm(r_ij))

        idx = i * 2 * param['num_dims']
        f[idx : idx + 2 * param['num_dims']] = np.concatenate([v_i, a_i])

    return f

def get_g(param, x, u):

    # Define control matrix
    G = {0 : np.array([0,  param['a_u']]),
         1 : np.array([0, -param['a_u']]),
         2 : np.array([ param['a_u'], 0]),
         3 : np.array([-param['a_u'], 0])}

    g = np.zeros(2 * param['n'] * param['num_dims'])
    for i in range(param['num_birds'], param['n']):
        v_i = get_v(param, x, i)
        idx = i * 2 * param['num_dims']
        g[idx : idx + 2 * param['num_dims']] = np.concatenate([v_i, G[u]])

    return g


########################
# SIMULATION UNIT TEST #
########################

if __name__ == '__main__':
    param = {'render' : True, 'num_birds' : 6, 'num_agents' : 1, 'num_dims' : 2, 'plim' : 1, 'vlim' : 1, 'min_dist_constraint' : 0.3,
            'agentcolor' : 'blue', 'birdcolor' : 'green', 'r_comm' : 1., 'r_des' : 0.8, 'kx' : 2, 'kv' : 2,
            'lambda_a' : 0.1, 'dt' : 0.05}

    param['n'] = param['num_birds'] + param['num_agents']
    param['a_u'] = 1.

    plot = init_plot(param)
    x = init_state(param)

    import time
    for i in range(1000):
        x += (get_f(param, x) + get_g(param, x, random.randint(0, 3))) * param['dt']
        plot_boids(param, x, plot)
