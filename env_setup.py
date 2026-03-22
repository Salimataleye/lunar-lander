import gymnasium as gym

def make_env(render=False):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return env