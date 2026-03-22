import gymnasium as gym

ENV_ID = "LunarLander-v3"


def make_env(render=False, seed=None):
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
