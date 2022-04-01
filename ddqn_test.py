import torch
import numpy as np
from ddqn_train import DDQN
from game_env_2t import qwopEnv
import gym

if __name__ == '__main__':
    PATH = None
    hidden_dim = 64

    env = qwopEnv()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDQN(action_dim=env.action_space.n,
                 state_dim=env.observation_space.shape[0],
                 hidden_dim=hidden_dim).to(device)

    # torch.load() -> dict and load_state_dict into the model
    model.load_state_dict(torch.load(PATH))

    state = env.reset()
    for i_episode in range(10):
        observation = env.reset()
        while not env.gameover:
            env.render()
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = model(state,"online")
                # values = model(state, "target")
            action = np.argmax(values.cpu().numpy())
            observation, reward, done, info = env.step(action)
            total+=reward
            if done:
                print(reward)
                break
    env.close()