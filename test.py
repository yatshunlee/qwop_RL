from game_env import qwopEnv
from stable_baselines3 import DQN

if __name__ == '__main__':
    env = qwopEnv()

    model = DQN.load("qwop3")

    obs = env.reset()
    while True:
        if env.gameover:
            obs = env.reset()
        else:
            action, _states = model.predict(obs, deterministic=True)
            # return obs, reward, done, info from step function
            env.step(action)