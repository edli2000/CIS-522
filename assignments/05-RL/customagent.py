import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQNet, self).__init__()
        self.fc1 = nn.Linear(in_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    """
    Class for the agent being trained to play the RL game
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        gamma: float = 0.99,
        lr: float = 0.001,
        eps_init: float = 1.0,
        eps_final: float = 0.01,
        eps_decay: float = 0.995,
        batch_size: int = 4,
        max_mem: int = 10000,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lr = lr
        self.eps = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_mem)

        self.nnet = DQNet(self.observation_space.shape[0], self.action_space.n).to(
            self.device
        )
        self.target_nnet = DQNet(
            self.observation_space.shape[0], self.action_space.n
        ).to(self.device)
        self.target_nnet.load_state_dict(self.nnet.state_dict())
        self.target_nnet.eval()

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.last_obs = None
        self.last_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Function for computing the actions of the agent

        Args:
            observation (gym.spaces.Box): state of environment
        Returns:
            gym.spaces.Discrete: a sampling of actions for the agent
        """
        if np.random.rand() < self.eps:
            return self.action_space.sample()
        else:
            observation = torch.tensor([observation], dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                return int(torch.argmax(self.nnet(observation)).item())

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Function for training the agent for the RL game

        Args:
            observation (gym.spaces.Box): state of environment
            reward (float): amount of reward
            terminated (bool): whether the game has terminated
            truncated (bool): whether the game is truncated
        Returns:
            None
        """
        if len(self.memory) < self.batch_size:
            return

        self.memory.append(
            (
                observation,
                reward,
                self.last_obs,
                self.last_action,
                terminated or truncated,
            )
        )
        self.last_obs = observation
        self.last_action = self.act(observation)

        if len(self.memory) < self.batch_size:
            return

        sample = random.sample(self.memory, self.batch_size)
        next_obs, rewards, obs, actions, dones = zip(*sample)

        next_obs = torch.tensor(np.stack(next_obs), dtype=torch.float32).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        obs = torch.tensor(np.stack(obs), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        qvals = self.nnet(obs).gather(1, actions)
        next_qvals = self.target_nnet(next_obs).detach().max(1)[0].unsqueeze(1)
        tgt_qvals = rewards + (self.gamma * next_qvals * (1 - dones))

        loss = self.criterion(qvals, tgt_qvals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_final, self.eps_decay * self.eps)

        if terminated or truncated:
            self.target_nnet.load_state_dict(self.nnet.state_dict())
