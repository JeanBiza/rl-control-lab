import gymnasium as gym
import random
import numpy as np

random.seed(42)
np.random.seed(42)


env = gym.make("CartPole-v1")
env.action_space.seed(42)

state, info = env.reset(seed=42)
print("Initial state:", state)
print("  -> [cart position, cart velocity, pole angle, pole angular velocity]\n")

total_reward = 0
step_count = 0
terminated = False
truncated = False

while not terminated and not truncated:
    action = env.action_space.sample()

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    print(f"Step {step_count:3d} | action={action} | state={state.round(3)} | reward={reward}")

print(f"\nEpisode ended at step {step_count}")
print(f"Total accumulated reward: {total_reward}")
print("(remember: +1 for every step the pole stayed within the allowed range)")

env.close()