import torch
import numpy as np
from ddqn_train import DDQN
from game_env import qwopEnv

if __name__ == '__main__':
    PATH = 'checkpoints/2022-04-03T14-20-52/qwop_ddqn_50.chkpt'
    hidden_dim = 64*2

    env = qwopEnv()
    env.MAX_DURATION = 720
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDQN(action_dim=env.action_space.n,
                 state_dim=env.observation_space.shape[0],
                 hidden_dim=hidden_dim).to(device)

    # torch.load() -> dict and load_state_dict into the model
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model'])

    state = env.reset()
    for i_episode in range(1):
        observation = env.reset()

        while not env.gameover:
            env.render()
            state = torch.Tensor(state).to(device)

            with torch.no_grad():
                values = model(state,"online")

            action = np.argmax(values.cpu().numpy())
            observation, reward, done, info = env.step(action)

            if done:
                break

    # env.close()