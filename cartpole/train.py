from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=20_000)

model.save("ppo_cartpole")
env.close()