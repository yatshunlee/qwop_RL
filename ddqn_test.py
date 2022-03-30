import torch
import numpy as np
from ddqn_train import QNetwork
from game_env import qwopEnv

if __name__ == '__main__':
    PATH = 'Q1.pth'
    hidden_dim = 64
    num_test_episodes = 5

    env = qwopEnv()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = QNetwork(action_dim=env.action_space.n,
                     state_dim=env.observation_space.shape[0],
                     hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(PATH))

    state = env.reset()
    while True:
        if env.gameover:
            obs = env.reset()
        else:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = model(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)