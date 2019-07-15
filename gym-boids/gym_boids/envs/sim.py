import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm 
from numpy.random import rand
from numpy.linalg import norm
from math import exp
import sys

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
    boidplot, = plt.plot([], [], 'o', c=param['birdcolor'])
    agentplot, = plt.plot([], [], 'go', c=param['agentcolor'])

    return {'fig' : fig, 'boidplot' : boidplot, 'agentplot' : agentplot}

def plot_boids(param, x, plot):
    """Update the plot with the current state given a pre-initialized plot"""

    fig = plot['fig']
    boidplot = plot['boidplot']
    agentplot = plot['agentplot']

    plotx = []
    ploty = []
    for i in range(param['num_birds']):
        p_i = get_p(param, x, i)
        plotx.append(p_i[0])
        ploty.append(p_i[1])

    boidplot.set_data(plotx, ploty)

    plotx = []
    ploty = []
    for i in range(param['num_agents']):
        p_i = get_p(param, x, i + param['num_birds'])
        plotx.append(p_i[0])
        ploty.append(p_i[1])

    agentplot.set_data(plotx, ploty)

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
        a_v = np.zeros(param['num_dims'])
        a_x = np.zeros(param['num_dims'])

        for j in range(param['n']):
            if i == j:
                continue

            p_j = get_p(param, x, j)
            v_j = get_v(param, x, j)
            r_ij = p_j - p_i

            if j == param['n'] - 1:
                print(f"p_j: {p_j}\nv_j: {v_j}\nr_ij: {r_ij}\nA[i, j]: {A[i, j]}")

            a_v += A[i, j] * (param['kv'] * (v_j - v_i))
            a_x += param['kx'] * r_ij * (1 - param['r_des']/norm(r_ij))

        print(f"a_v: {norm(a_v)}, a_x: {norm(a_x)}")
        a_i = a_v + a_x


        idx = i * 2 * param['num_dims']
        f[idx : idx + 2 * param['num_dims']] = np.concatenate([v_i, a_i])

    for i in range(param['num_birds'] + 1, param['n']):
        v_i = get_v(param, x, i)
        idx = i * 2 * param['num_dims']
        f[idx : idx + 2 * param['num_dims']] = np.concatenate([v_i, np.zeros(param['num_dims'])])

    return f

def get_g(param, u):
    g = np.zeros(2 * param['n'] * param['num_dims'])
    for i in range(param['num_birds'] + 1, param['n']):
        idx = i * 2 * param['num_dims']
        f[idx : idx + 2 * param['num_dims']] = np.concatenate([v_i, np.zeros(param['num_dims'])])
