import numpy as np
from typing import Dict, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrl.data import ListStorage, ReplayBuffer
from tqdm import tqdm
import copy

from reduction_env import ReductionEnvConfig, ReductionEnvironment


class ActorNetwork(nn.Module):
    def __init__(
        self, basis_dim: int, action_history_size: int, action_dim: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_history_size = action_history_size

        self.basis_processor = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(basis_dim ** 2, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256)
        )
        self.meta_processor = nn.Sequential(
            nn.Embedding(action_dim + 1, 32, action_dim),
            nn.Flatten(-2, -1),
            nn.Linear(action_history_size * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64)
        )
        self.actor = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        basis = tensordict["basis"]
        basis_features = self.basis_processor(basis)

        action_history = tensordict["action_history"].to(torch.long)
        action_history[(action_history == -1)] = self.action_dim
        meta_features = self.meta_processor(action_history)

        combined = torch.cat([basis_features, meta_features], dim=1)

        return self.actor(combined)


class CriticNetwork(nn.Module):
    """Critic network for DDPG - estimates Q-values"""

    def __init__(
        self, basis_dim: int, action_history_size: int, action_dim: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_history_size = action_history_size

        self.basis_processor = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(basis_dim ** 2, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256)
        )
        self.meta_processor = nn.Sequential(
            nn.Embedding(action_dim + 1, 8, action_dim),
            nn.Flatten(-2, -1),
            nn.Linear(action_history_size * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64)
        )
        self.action_processor = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.critic = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 + 64 + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, tensordict: TensorDict, action: torch.Tensor) -> torch.Tensor:
        basis = tensordict["basis"]
        basis_features = self.basis_processor(basis)

        action_history = tensordict["action_history"].to(torch.long)
        action_history[(action_history == -1)] = self.action_dim
        meta_features = self.meta_processor(action_history)

        # Process the action (one-hot encoded or action probabilities)
        action_features = self.action_processor(action)

        combined = torch.cat(
            [basis_features, meta_features, action_features], dim=1)

        return self.critic(combined)


class OUNoise:
    """Ornstein-Uhlenbeck process for action exploration"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.standard_normal(len(x))
        self.state = x + dx
        return torch.FloatTensor(self.state)


class DDPGConfig:
    def __init__(self, env_config: ReductionEnvConfig = None, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=64, buffer_size=100000,
                 dropout_p=0.2, noise_sigma=0.2):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau  # For soft target network updates
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dropout_p = dropout_p
        self.noise_sigma = noise_sigma
        self.env_config = env_config if env_config is not None else ReductionEnvConfig()


class DDPGAgent(nn.Module):
    def __init__(self, ddpg_config: DDPGConfig) -> None:
        super().__init__()
        self.ddpg_config = ddpg_config
        self.action_dim = self.ddpg_config.env_config.actions_n

        # Actor and critic networks
        self.actor = ActorNetwork(
            self.ddpg_config.env_config.basis_dim,
            self.ddpg_config.env_config.action_history_size,
            self.action_dim,
            ddpg_config.dropout_p
        )

        self.critic = CriticNetwork(
            self.ddpg_config.env_config.basis_dim,
            self.ddpg_config.env_config.action_history_size,
            self.action_dim,
            ddpg_config.dropout_p
        )

        # Target networks for stability
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Freeze target networks with respect to optimizers to avoid
        # unnecessary computations
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=ddpg_config.actor_lr)
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=ddpg_config.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(max_size=ddpg_config.buffer_size)
        )

        # Noise process for exploration
        self.noise = OUNoise(self.action_dim, sigma=ddpg_config.noise_sigma)

    def store_transition(self, state, action, reward, done, next_state):
        # Convert to appropriate tensor types
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        # Handle action tensor
        if isinstance(action, int):
            action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
            action_tensor[action] = 1.0
        else:
            action_tensor = action

        # Create TensorDict for storage
        td = TensorDict({
            "state": {
                "basis": state["basis"],
                "action_history": state["action_history"]
            },
            "action": action_tensor,
            "reward": reward,
            "done": done,
            "next_state": {
                "basis": next_state["basis"],
                "action_history": next_state["action_history"]
            }
        }, batch_size=[])

        self.replay_buffer.add(td)

    def get_action(self, state: TensorDict) -> Tuple[int, torch.Tensor]:
        """Get action from the actor network, optionally with added noise for exploration"""
        with torch.no_grad():
            action_probs = self.actor(state)

        # Add noise for exploration during training
        if self.training:
            noise = self.noise.sample()
            action_probs = action_probs + noise.to(action_probs.device)
            action_probs = torch.clamp(action_probs, 1e-7, 1)
            # Renormalize to ensure it's a valid probability distribution
            action_probs = action_probs / action_probs.sum()

        # Sample discrete action based on probabilities
        dist = torch.distributions.Categorical(probs=action_probs)
        action_idx = dist.sample().item()

        return action_idx, action_probs

    def soft_update_target_networks(self):
        """Soft update of target networks using polyak averaging"""
        tau = self.ddpg_config.tau

        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)

        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)

    def update(self, device: Union[torch.device, str]) -> None:
        if len(self.replay_buffer) < self.ddpg_config.batch_size:
            return None

        # Is there a nicer way to do the following?
        batch = self.replay_buffer.sample(
            self.ddpg_config.batch_size).to(device)
        states = batch["state"].flatten(0, 1)
        next_states = batch["next_state"].flatten(0, 1)
        actions = batch["action"].flatten(0, 1)
        rewards = batch["reward"].flatten(0, 1)
        dones = batch["done"].flatten(0, 1)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions).flatten()
            target_q = rewards + self.ddpg_config.gamma * target_q * (~dones)

        # Update critic
        current_q = self.critic(states, actions).flatten()
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor (policy)
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update_target_networks()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q_value': current_q.mean().item()
        }

    def train_step(self, batch: TensorDict, device: Union[torch.device, str]) -> Dict[str, float]:
        self.train()

        # Unpack batch data
        basis = batch['basis']  # [batch_size, n_dim, n_dim]
        shortest_vector = batch['shortest_vector']  # [batch_size, n_dim]

        # Reset environment
        env = ReductionEnvironment(self.ddpg_config.env_config)
        state, _ = env.reset(options={'basis': basis.squeeze(
        ), 'shortest_vector': shortest_vector.squeeze()})
        self.noise.reset()  # Reset exploration noise

        done = False
        total_reward = 0

        state = TensorDict({
            "basis": state["basis"].unsqueeze(0).to(device),
            "action_history": state["action_history"].unsqueeze(0).to(device)
        }, batch_size=[])

        # Run episode
        step_count = 0
        max_steps = self.ddpg_config.env_config.max_steps

        while not done and step_count < max_steps:
            step_count += 1

            # Get action
            action_idx, action_probs = self.get_action(state)

            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            total_reward += reward

            next_state = TensorDict({
                "basis": next_state["basis"].unsqueeze(0).to(device),
                "action_history": next_state["action_history"].unsqueeze(0).to(device)
            }, batch_size=[])

            # Store transition in replay buffer
            self.store_transition(state, action_probs,
                                  reward, done, next_state)

            # Move to next state
            state = next_state

            # Update networks
            update_info = self.update(device)

        metrics = {'total_reward': total_reward, 'steps': step_count}
        if update_info is not None:
            metrics.update(update_info)

        return metrics

    def evaluate(self, dataloader: DataLoader, device: Union[torch.device, str]) -> Dict[str, float]:
        self.eval()
        env = ReductionEnvironment(self.ddpg_config.env_config)

        with torch.no_grad():
            total_reward = 0.0
            total_steps = 0
            success_count = 0
            num_samples = len(dataloader)

            for batch in tqdm(dataloader, dynamic_ncols=True):
                # Unpack batch data
                basis = batch['basis'].squeeze()
                shortest_vector = batch['shortest_vector'].squeeze()

                state, info = env.reset(
                    options={'basis': basis, 'shortest_vector': shortest_vector})
                log_defect_history = [info["log_defect"]]
                done = False
                episode_reward = 0
                steps = 0

                state = TensorDict({
                    "basis": state["basis"].unsqueeze(0).to(device),
                    "action_history": state["action_history"].unsqueeze(0).to(device)
                }, batch_size=[])

                while not done and steps < self.ddpg_config.env_config.max_steps:
                    # Get action (without exploration noise during evaluation)
                    action_idx, _ = self.get_action(state)

                    next_state, reward, terminated, truncated, info = env.step(
                        action_idx)
                    next_state = TensorDict({
                        "basis": next_state["basis"].unsqueeze(0).to(device),
                        "action_history": next_state["action_history"].unsqueeze(0).to(device)
                    }, batch_size=[])

                    log_defect_history.append(info["log_defect"])
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                    state = next_state

                total_reward += episode_reward
                total_steps += steps

                # Check success (similar to the PPO implementation)
                final_log_defect = log_defect_history[-1]
                if final_log_defect - batch["lll_log_defect"] < 1e-3:
                    success_count += 1

            return {
                'avg_reward': total_reward / num_samples,
                'avg_steps': total_steps / num_samples,
                'success_rate': success_count / num_samples
            }
