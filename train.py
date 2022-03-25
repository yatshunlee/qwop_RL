from game_env import qwopEnv
from stable_baselines3 import DQN


if __name__ == '__main__':
    env = qwopEnv()

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000, log_interval=4)
    model.save("qwop")