import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import random, copy, collections, datetime, os, time
from game_env import qwopEnv
from logger import MetricLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQN(nn.Module):
    """
    The Double Deep Q-Network has as input a state s and
    outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
    :param: state_dim: for input layer
    :param: hidden_dim: for every hidden layer
    :param: action_dim: for output layer
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        When doing update by forward, it takes:
        :param: input: all state of each observation
        :param: model: online or target
        :return: Q_values of all actions given state from online/target
        """

        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class Rabbit:
    def __init__(self, state_dim, action_dim, hidden_dim, save_dir):
        self.log_interval = 4
        self.save_dir = save_dir

        # FOR ACT
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = DDQN(self.action_dim,
                        self.state_dim,
                        self.hidden_dim).to(device=device)

        # - training parameter
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999999
        self.exploration_rate_min = 0.05
        self.current_step = 0

        self.save_net_every = 5e5 # no. of exp between saving network

        # FOR CACHE AND RECALL
        self.memory = collections.deque(maxlen=100000) # truncated list w/ maxlen
        self.batch_size = 32

        # FOR LEARN
        self.burnin = 5e2 # min. experiences before training (learning start)
        self.learn_every = 3 # update every learn_every of experiences
        self.sync_every = 1e4 # synv every sync_every of experiences
        # - td_estimate and td_target
        self.gamma = 0.99
        # - update_Q_online
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and
        update the Q value.

        :param: state(body_state), dimension = (state_dim)
        :return: action_idx for rabbit to take action
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
            # env.action_space.sample()
        # EXPLOIT
        else:
            state = state.__array__()
            state = torch.tensor(state,
                                 dtype=torch.float32).to(device=device)
            state = state.unsqueeze(0)

            # argmax from online
            action_values = self.net(state, 'online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease the exploration rate until the min.
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate,
                                    self.exploration_rate_min)

        self.current_step += 1
        return action_idx

    def cache(self, state, action, reward, next_state, done):
        """
        Store the experience to self.memory (replay buffer)
        Experience contains of following params
        """
        state = state.__array__()
        next_state = next_state.__array__()

        state = torch.tensor(state,
                             dtype=torch.float32).to(device=device)
        next_state = torch.tensor(next_state,
                                  dtype=torch.float32).to(device=device)
        action = torch.tensor([action]).to(device=device)
        reward = torch.tensor([reward],
                              dtype=torch.float32).to(device=device)
        done = torch.tensor([done]).to(device=device)

        experience = (state, next_state, action, reward, done)
        self.memory.append(experience)

    def recall(self):
        """Retrieve a batch of experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(),\
               reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """Return TD estimate"""
        # TD_estimate = Q*_online(s,a)
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    # Use the decorator disable gradient calculation of td_target
    @torch.no_grad()
    def td_target(self, next_state, reward, done):
        """Return TD target"""
        # a = argmax_a (Q_online(s',a))
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)

        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        Backpropagate the loss to update the parameters.
        Update for parameter_online:
        parameter_online <- parameter_online + alpha * d/dtheta(TD_est - TD_target)
        :return loss: the average of batch losses
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Periodically copy parameter_online to parameter_target."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        """Save checkpoint"""
        num = int(self.current_step // self.save_net_every)
        save_path = (
            self.save_dir / f"qwop_ddqn_{num}.chkpt"
        )
        torch.save(
            dict(model=self.net_state_dict(), exploration_rate=self.exploration_rate),
            save_path
        )
        print(f"Saved to {save_path} at step {self.current_step}.")

    def learn(self):
        # sync Q target every sync_every steps
        if self.current_step % self.sync_every == 0:
            self.sync_Q_target()
        # save current net every save_net_every steps
        if self.current_step % self.save_net_every == 0:
            self.save()
        # do nothing before burning in
        if self.current_step < self.burnin:
            return None, None
        # learn every learn_every steps
        if self.current_step % self.learn_every != 0:
            return None, None
        # log interval
        if self.current_step % self.log_interval == 0:
            print('current timestep:', self.current_step)

        # sample from memory
        state, next_state, action, reward, done = self.recall()

        # get TD estimate
        td_est = self.td_estimate(state, action)
        # get TD target
        td_tgt = self.td_target(next_state, reward, done)
        # backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


if __name__ == '__main__':
    env = qwopEnv()
    env.reset()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)

    rabbit = Rabbit(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=64*2,
        save_dir=save_dir,
    )

    logger = MetricLogger(save_dir=save_dir)

    episodes = 3000
    s = time.time()

    for ep in range(episodes):
        state = env.reset()

        while True:
            # get action based on state from agent
            action = rabbit.act(state)
            # performs action in env
            next_state, reward, done, info = env.step(action)
            # remember
            rabbit.cache(state, action, reward, next_state, done)
            # learn
            q, loss = rabbit.learn()
            # logging
            logger.log_step(reward, loss, q)
            # update state
            state = next_state
            # check if the game end
            if done:
                break

        logger.log_episode()

        if ep % 20 == 0:
            logger.record(episode=ep, epsilon=rabbit.exploration_rate, step=rabbit.current_step)

    print('It took',time.time()-s,'to complete 1000 episodes')