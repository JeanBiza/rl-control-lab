from stable_baselines3 import SAC
import gymnasium as gym

env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=20_000)

model.save("sac_pendulum")
env.close()