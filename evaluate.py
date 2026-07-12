import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

MODEL_PATH = "ppo_cartpole"
N_EVAL_EPISODES = 20
RENDER = True

env = gym.make("CartPole-v1", render_mode="human" if RENDER else None)

model = PPO.load(MODEL_PATH)

episode_rewards = []

for episode in range(1, N_EVAL_EPISODES + 1):
    state, info = env.reset(seed=episode)
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        action, _states = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode {episode:2d} | reward = {total_reward}")

env.close()

episode_rewards = np.array(episode_rewards)
print("\n--- Summary over", N_EVAL_EPISODES, "episodes ---")
print(f"Mean reward:   {episode_rewards.mean():.1f}")
print(f"Std deviation: {episode_rewards.std():.1f}")
print(f"Min / Max:     {episode_rewards.min()} / {episode_rewards.max()}")
print("(Compare this to the random baseline: ~10-50 per episode)")