import gymnasium as gym
import numpy as np


class Agent:
    """
    Class for the agent being trained to play the RL game
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        gamma: float = 0.99,
        lr: float = 0.1,
        min_eps: float = 0.01,
        eps: float = 1.0,
        eps_decay: float = 0.95,
        batch_size: int = 64,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lr = lr
        self.min_eps = min_eps
        self.eps = eps
        self.eps_decay = eps_decay
        self.q_table = np.zeros((self.observation_space.shape[0], self.action_space.n))
        self.batch_size = batch_size
        self.prev_obs = None
        self.prev_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Function for computing the actions of the agent

        Args:
            observation (gym.spaces.Box): state of environment
        Returns:
            gym.spaces.Discrete: a sampling of actions for the agent
        """
        if observation in self.q_table:
            action = np.argmax(self.q_table[observation])
            return action
        else:
            return self.action_space.sample()

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
        next_obs = gym.spaces.Box
        if self.prev_action is not None:
            obs_idx = np.ravel_multi_index(
                observation.astype(int), self.observation_space.shape
            )
            prev_action_idx = int(self.prev_action)
            next_obs_idx = np.ravel_multi_index(
                next_obs.astype(int), self.observation_space.shape
            )

            prev_qval = self.q_table[obs_idx, prev_action_idx]
            max_qval = np.max(self.q_table[next_obs_idx])
            next_qval = (1 - self.lr) * prev_qval + self.lr * (
                reward + self.gamma * max_qval
            )
            self.q_table[obs_idx, prev_action_idx] = next_qval

            self.eps = max(self.min_eps, self.eps_decay * self.eps)
