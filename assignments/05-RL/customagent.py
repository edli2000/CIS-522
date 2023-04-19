import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from contextlib import contextmanager


@contextmanager
def evaluate(nnet):
    """Switch from train to evaluate"""
    is_training = nnet.training
    try:
        nnet.eval()
        yield nnet
    finally:
        if is_training:
            nnet.train()


class MLPAgent(nn.Module):
    def __init__(self, num_features, num_actions, hidden=64):
        """
        A simple MLP backbone for a PPO agent,
        where the input features are all numerical,
        and the output is the choice within n actions.
        """
        super().__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.hidden = hidden

        self.extractor_net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def sample(self, state):
        state = self.extractor_net(state)
        logits = self.actor_net(state)
        distr = torch.distributions.Categorical(logits=logits)
        return distr.sample()

    def score(self, state, action):
        state = self.extractor_net(state)
        logits = self.actor_net(state)
        distr = torch.distributions.Categorical(logits=logits)
        return (
            self.critic_net(state).flatten(),
            distr.log_prob(action.flatten()),
            distr.entropy(),
        )


class Agent:
    """
    Class for the agent being trained to play the RL game
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.gamma = 0.99
        self.lamb = 0.9
        self.entropy = 0.9
        self.lr = 1e-3
        self.train_epochs = 6
        self.explore_epochs = 4
        self.batch_size = 64
        self.value_func = 0.1
        self.value_func_clip = 10.0
        self.ppo_clip = 0.5
        self.hidden = 128

        self.n_obs = observation_space.shape[0]
        self.n_actions = action_space.n
        self.step_ctr = 0
        self.episode_ctr = 0
        self.cache = []

        self.agent = MLPAgent(self.n_obs, self.n_actions, hidden=self.hidden)
        self.prev_agent = MLPAgent(self.n_obs, self.n_actions, hidden=self.hidden)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr)
        for param in self.prev_agent.parameters():
            param.requires_grad_(False)
        self.reset_buff()

    def reset_buff(self):
        self.cache = [([], [], []) for _ in range(self.explore_epochs)]

    def update_state(self, obs, action):
        idx = self.episode_ctr % self.explore_epochs
        self.cache[idx][0].append(obs)
        self.cache[idx][1].append(torch.tensor(action))

    def update_reward(self, reward):
        idx = self.episode_ctr % self.explore_epochs
        self.cache[idx][2].append(reward)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Function for computing the actions of the agent

        Args:
            observation (gym.spaces.Box): state of environment
        Returns:
            gym.spaces.Discrete: a sampling of actions for the agent
        """
        obs = torch.tensor(np.array(observation), dtype=torch.float32)
        with torch.no_grad(), evaluate(self.prev_agent):
            action = self.prev_agent.sample(obs).item()
        self.update_state(obs, action)
        return action

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
        self.update_reward(reward)
        if truncated or terminated:
            self.episode_ctr += 1
            if self.episode_ctr % self.explore_epochs == 0:
                sample = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                    "values": [],
                    "log_probs": [],
                    "adv": [],
                    "returns": [],
                }

                with torch.no_grad(), evaluate(self.prev_agent):
                    self.prev_agent.to(device=None)
                    for _, (states, actions, rewards) in enumerate(self.cache):
                        states = states[: len(actions)]
                        stacked_states = (
                            torch.stack(states)
                            if isinstance(states[0], torch.Tensor)
                            else tuple(torch.stack(t) for t in zip(*states))
                        )
                        states = (
                            stacked_states.to(device=None)
                            if isinstance(stacked_states, torch.Tensor)
                            else tuple(t.to(device=None) for t in stacked_states)
                        )
                        stacked_actions = (
                            torch.stack(actions)
                            if isinstance(actions[0], torch.Tensor)
                            else tuple(torch.stack(t) for t in zip(*actions))
                        )
                        actions = (
                            stacked_actions.to(device=None)
                            if isinstance(stacked_actions, torch.Tensor)
                            else tuple(t.to(device=None) for t in stacked_actions)
                        )
                        values, log_probs, _ = self.prev_agent.score(states, actions)
                        values = values.flatten()
                        log_probs = log_probs.flatten()

                        rewards = torch.tensor(
                            rewards, device=None, dtype=torch.float32
                        )
                        with torch.no_grad():
                            old_device = rewards.device
                            rewards = rewards.detach().clone().cpu()
                            values = values.detach().clone().cpu()
                            adv = torch.zeros(len(rewards), device="cpu")
                            adv[-1] = rewards[-1] - values[-1]
                            for idx in reversed(range(len(rewards) - 1)):
                                delta = (
                                    rewards[idx]
                                    + self.gamma * values[idx + 1]
                                    - values[idx]
                                )
                                adv[idx] = delta + self.gamma * self.lamb * adv[idx + 1]
                            adv = adv.to(device=old_device)

                        returns = values + adv
                        sample["states"].append(states)
                        sample["actions"].append(actions)
                        sample["rewards"].append(rewards)
                        sample["values"].append(values)
                        sample["log_probs"].append(log_probs)
                        sample["adv"].append(adv)
                        sample["returns"].append(returns)

                sample["states"] = (
                    torch.cat(sample["states"])
                    if isinstance(sample["states"][0], torch.Tensor)
                    else tuple(torch.cat(t) for t in zip(*sample["states"]))
                )
                sample["actions"] = (
                    torch.cat(sample["actions"])
                    if isinstance(sample["actions"][0], torch.Tensor)
                    else tuple(torch.cat(t) for t in zip(*sample["actions"]))
                )
                sample["rewards"] = torch.cat(sample["rewards"])
                sample["values"] = torch.cat(sample["values"])
                sample["log_probs"] = torch.cat(sample["log_probs"])
                sample["adv"] = torch.cat(sample["adv"])
                sample["returns"] = torch.cat(sample["returns"])
                self.agent.to(device=None)
                self.agent.train()

                for _ in range(self.train_epochs):
                    batch_indices = torch.split(
                        torch.randperm(len(sample["rewards"]), device=None),
                        split_size_or_sections=self.batch_size,
                    )
                    for batch_idx in batch_indices:
                        batch_states = (
                            sample["states"][batch_idx]
                            if isinstance(sample["states"], torch.Tensor)
                            else tuple(t[batch_idx] for t in sample["states"])
                        )
                        batch_actions = (
                            sample["actions"][batch_idx]
                            if isinstance(sample["actions"], torch.Tensor)
                            else tuple(t[batch_idx] for t in sample["actions"])
                        )
                        batch_prev_log_probs = sample["log_probs"][batch_idx]
                        batch_prev_adv = sample["adv"][batch_idx]
                        batch_prev_values = sample["values"][batch_idx]
                        batch_prev_returns = sample["returns"][batch_idx]
                        (
                            batch_new_values,
                            batch_new_log_probs,
                            batch_new_entropy,
                        ) = self.agent.score(batch_states, batch_actions)

                        clip_vals = batch_prev_values + (
                            batch_new_values - batch_prev_values
                        ).clamp(min=-self.value_func_clip, max=self.value_func_clip)
                        vf_loss = (
                            torch.maximum(
                                (batch_new_values - batch_prev_returns) ** 2,
                                (clip_vals - batch_prev_returns) ** 2,
                            ).mean()
                            * 0.5
                        )
                        ratio = torch.exp(batch_new_log_probs - batch_prev_log_probs)
                        clip_ratio = ratio.clamp(
                            min=1.0 - self.ppo_clip, max=1.0 + self.ppo_clip
                        )
                        policy_obj = torch.minimum(
                            ratio * batch_prev_adv, clip_ratio * batch_prev_adv
                        )
                        policy_loss = -policy_obj.mean()
                        entropy_loss = -batch_new_entropy.mean()
                        loss = (
                            policy_loss
                            + self.value_func * vf_loss
                            + self.entropy * entropy_loss
                        )

                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                self.reset_buff()
                self.prev_agent.load_state_dict(self.agent.state_dict())
