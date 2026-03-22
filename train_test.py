import numpy as np
from env_setup import make_env

def train(agent, num_episodes=500):
    env = make_env(render=False)
    rewards_log = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total_reward += reward
            done = terminated or truncated

        rewards_log.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards_log[-50:])
            print(f"Episode {episode + 1} | Avg Reward (last 50): {avg:.2f}")

    env.close()
    return rewards_log


def test(agent, num_episodes=10):
    env = make_env(render=True)

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Test Episode {episode + 1} | Total Reward: {total_reward:.2f}")

    env.close()