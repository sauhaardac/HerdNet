from matplotlib.backends.backend_pdf import PdfPages
from param import params
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import util
import numpy as np
import os


def plot_error(x, T, title="Errors"):
    fig, ax = plt.subplots()

    ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([-1, 1])

    pbar = util.get_P_bar(x)
    vbar = np.gradient(pbar, params['dt'], axis=0)
    abar = np.gradient(vbar, params['dt'], axis=0)

    pe = pbar - params['pd']
    ve = vbar - params['vd']
    ae = abar - params['ad']

    plt.plot(T, pe[:, 0], label='pe_x')
    plt.plot(T, pe[:, 1], label='pe_y')
    plt.plot(T, ve[:, 0], label='ve_x', alpha=0.7)
    plt.plot(T, ve[:, 1], label='ve_y', alpha=0.7)
    plt.plot(T, ae[:, 0], label='ae_x', alpha=0.3)
    plt.plot(T, ae[:, 1], label='ae_y', alpha=0.3)

    plt.legend()
    plt.title(title)


def plot_SS(X, T, title='State Space'):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim([-2, 2])
    ax.set_xlim([-2, 2])
    # plt.axis('equal')

    for i in range(params['n']):
        P_i = util.get_P(X, i)
        if i < params['num_birds']:
            color = params['birdcolor']
        else:
            color = params['agentcolor']

        if i == 0:
            label = 'Bird'
        elif i == params['n'] - 1:
            label = 'Controlled Agent'
        else:
            label = '_nolegend_'

        ax.plot(P_i[:, 0], P_i[:, 1], color=color, alpha=0.3, label=label)
        ax.scatter(P_i[0, 0], P_i[0, 1], color=color,
                   marker=params['start_marker'], alpha=0.3, label='_nolegend_')
        ax.scatter(P_i[-1, 0], P_i[-1, 1], color=color,
                   marker=params['stop_marker'], alpha=0.3, label='_nolegend_')

    ax.plot(params.get('pd')[:, 0], params.get('pd')[:, 1],
            color=params.get('pdcolor'), label='Desired Trajectory')

    P_bar = util.get_P_bar(X)
    ax.plot(P_bar[:, 0], P_bar[:, 1], color=params.get(
        'centroidcolor'), label='Bird Centroid')
    ax.plot(P_bar[0, 0], P_bar[0, 1], color=params.get('centroidcolor'),
            marker=params.get('start_marker'), label='_nolegend_')
    ax.plot(P_bar[-1, 0], P_bar[-1, 1], color=params.get('centroidcolor'),
            marker=params.get('stop_marker'), label='_nolegend_')

    plt.title(title)
    plt.legend()


def test():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
# ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.title("Error vs. Time")
    plt.plot(range(len(p_0)), p_0, label='p_0 error')
    plt.plot(range(len(p_1)), p_1, label='p_1 error')
    plt.plot(range(len(v_0)), v_0, label='v_0 error')
    plt.plot(range(len(v_1)), v_1, label='v_1 error')
    plt.legend()

    plt.ylim([-0.5, 0.5])

    plt.savefig('fig.png', dpi=300)


def save_figs():
    fn = os.path.join(os.getcwd(), params.get('fn_plots'))

    pp = PdfPages(fn)
    for i in plt.get_fignums():
        pp.savefig(plt.figure(i))
        plt.close(plt.figure(i))
    pp.close()
