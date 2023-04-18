import gymnasium as gym


class Agent:
    """
    Class for the agent being trained to play the RL game
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Function for computing the actions of the agent

        Args:
            observation (gym.spaces.Box): state of environment
        Returns:
            gym.spaces.Discrete: a sampling of actions for the agent
        """
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
        pass
