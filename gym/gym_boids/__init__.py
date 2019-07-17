from gym.envs.registration import register

register(
    id='boids-v0',
    entry_point='gym_boids.envs:BoidsEnv',
)
