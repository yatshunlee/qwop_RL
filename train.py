import time
from game_env import qwopEnv
from stable_baselines3 import DQN


if __name__ == '__main__':
    env = qwopEnv()

    model = DQN("MlpPolicy", env, learning_rate=0.001, verbose=1)

    t = time.time()
    model.learn(total_timesteps=8000, log_interval=4)
    model.save("qwop3")
    
    print("Time taken:", time.time()-t)
    env.close()