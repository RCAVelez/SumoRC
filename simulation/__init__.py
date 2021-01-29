from gym.envs.registration import register

register(
    id='RCSumo-v0',
    entry_point='simulation.envs:simulationEnv'
)
