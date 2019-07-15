import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sim import *

class BoidsEnv(gym.Env):
    def __init__(self, param):
        """Empty constructor function to satisfy gym requirements"""

        pass

    def init(self, param):
        """Real constructor with params input"""

        self.param = param 
        self.x = [init_state(param)]

        if param['render']:
            self.plot = init_plot(param)

    def step(self, u):
        """Simulate step in environment"""

        dxdt = get_f(self.param, self.x[-1]) + get_g(self.param, u)
        x.append(self.x[-1] + dxdt * self.param['dt'])

    def reset(self):
        """Reset enviornment"""

        self.x = [init_state(self.param)]
        return self.x
    
    def render(self):
        """Render enviornment"""

        plot_boids(self.param, x, self.plot)


####################################################################################################
#                                          SIMULATION UNIT TEST                                    #
####################################################################################################

if __name__ == '__main__':
    param = {'render' : True, 'num_birds' : 6, 'num_agents' : 1, 'num_dims' : 2, 'plim' : 1, 'vlim' : 1, 'min_dist_constraint' : 0.3,
            'agentcolor' : 'blue', 'birdcolor' : 'green', 'r_comm' : 1., 'r_des' : 0.8, 'kx' : 2, 'kv' : 2,
            'lambda_a' : 0.1, 'dt' : 0.05}

    param['n'] = param['num_birds'] + param['num_agents']

    plot = init_plot(param)
    x = init_state(param)

    import time
    for i in range(1000):
        x += get_f(param, x) * param['dt']
        plot_boids(param, x, plot)
